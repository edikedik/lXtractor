"""
A module defining core `lXtractor`'s data structures:

#. :class:`ChainSequence`
#. :class:`ChainStructure`
#. :class:`Chain`

... and methods to manipulate these objects.
"""
from __future__ import annotations

import logging
import operator as op
import typing as t
from collections import namedtuple, abc
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from dataclasses import dataclass
from functools import partial
from io import TextIOBase
from itertools import starmap, chain, zip_longest, tee, filterfalse, repeat
from pathlib import Path

import biotite.structure as bst
import numpy as np
import pandas as pd
from more_itertools import unzip, first_true, split_into, collapse, nth
from toolz import curry, keyfilter, keymap, valmap
from tqdm.auto import tqdm

from lXtractor.core.alignment import Alignment
from lXtractor.core.base import (
    AminoAcidDict,
    AbstractChain,
    Ord,
    AlignMethod,
    SeqReader,
)
from lXtractor.core.config import Sep, DumpNames, SeqNames, MetaNames
from lXtractor.core.exceptions import (
    MissingData,
    AmbiguousMapping,
    InitError,
    NoOverlap,
    LengthMismatch,
)
from lXtractor.core.segment import Segment
from lXtractor.core.structure import GenericStructure, _validate_chain, PDB_Chain
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.seq import mafft_align, map_pairs_numbering, read_fasta
from lXtractor.util.structure import filter_selection
from lXtractor.variables.base import Variables

T = t.TypeVar("T")

LOGGER = logging.getLogger(__name__)


def topo_iter(
    start_obj: T, iterator: abc.Callable[[T], abc.Iterator[T]]
) -> abc.Generator[list[T], None, None]:
    """
    Iterate over sequences in topological order.

    >>> n = 1
    >>> it = topo_iter(n, lambda x: (x + 1 for n in range(x)))
    >>> next(it)
    [2]
    >>> next(it)
    [3, 3]

    :param start_obj: Starting object.
    :param iterator: A callable accepting a single argument of the same type as
        the `start_obj` and returning an iterator over objects with the same
        type, representing the next level.
    :return: A generator yielding lists of objects obtained using `iterator`
        and representing topological levels with the root in `start_obj`.
    """

    def get_level(objs: abc.Iterable[T]) -> abc.Iterator[T]:
        return chain.from_iterable(map(iterator, objs))

    curr_level = list(iterator(start_obj))

    while True:
        yield curr_level
        curr_level = list(get_level(curr_level))
        if not curr_level:
            return


def _parse_children(children):
    if children:
        if not isinstance(children, ChainList):
            return ChainList(children)
        return children
    return ChainList([])


class ChainSequence(Segment):
    """
    A class representing polymeric sequence of a single entity (chain).

    The sequences are stored internally as a dictionary `{seq_name => seq}`
    and must all have the same length.
    Additionally, `seq_name` must be a valid field name:
    something one could use in namedtuples.
    If unsure, please use :func:`lXtractor.util.misc.is_valid_field_name`
    for testing.

    A single gap-less primary sequence (:meth:`seq1`) is mandatory
    during the initialization.
    We refer to the sequences other than :meth:`seq1` as "maps."
    To view the standard sequence names supported by :class:`ChainSequence`,
    use the :meth:`flied_names` property.

    The sequence can be a part of a larger one. The child-parent relationships
    are indicated via :attr:`parent` and attr:`children`, where the latter
    entails any sub-sequence. A preferable way to create subsequences is
    the :meth:`spawn_child` method.

    >>> seqs = {
    ...     'seq1': 'A' * 10,
    ...     'A': ['A', 'N', 'Y', 'T', 'H', 'I', 'N', 'G', '!', '?']
    ... }
    >>> cs = ChainSequence(1, 10, 'CS', seqs=seqs)
    >>> cs
    CS|1-10
    >>> assert len(cs) == 10
    >>> assert 'A' in cs and 'seq1' in cs
    >>> assert cs.seq1 == 'A' * 10

    """

    __slots__ = ()

    def iter_children(self) -> abc.Generator[list[ChainSequence]]:
        """
        Iterate over a child tree in topological order.

        >>> s = ChainSequence(1, 10, 'CS', seqs={'seq1': 'A' * 10})
        >>> ss = s.spawn_child(1, 5, 'CS_')
        >>> sss = ss.spawn_child(1, 3, 'CS__')
        >>> list(s.iter_children())
        [[CS_|1-5<-(CS|1-10)], [CS__|1-3<-(CS_|1-5<-(CS|1-10))]]

        :return: a generator over child tree levels, starting from
            the :attr:`children` and expanding such attributes over
            :class:`ChainSequence` instances within this attribute.
        """
        return topo_iter(self, lambda x: x.children)

    @property
    def fields(self) -> tuple[str, ...]:
        """
        :return: Names of the currently stored sequences.
        """
        return tuple(self._seqs.keys())

    @classmethod
    def field_names(cls) -> SeqNames:
        """
        :return: The default sequence's names.
        """
        return SeqNames

    @classmethod
    def meta_names(cls) -> dataclass:
        """
        :return: defaults names of the :attr:`meta` fields.
        """
        return MetaNames

    @property
    def numbering(self) -> abc.Sequence[int]:
        """
        :return: the primary sequence's (:meth:`seq1`) numbering.
        """
        if SeqNames.enum in self:
            return self[SeqNames.enum]
        return list(range(self.start, self.end + 1))

    @property
    def seq1(self) -> str:
        """
        :return: the primary sequence.
        """
        return self[SeqNames.seq1]

    @property
    def seq3(self) -> t.Sequence[str]:
        # TODO: remove and defer to subclasses representing concrete seqs.
        """
        :return: the three-letter codes of a primary sequence.
        """
        if SeqNames.seq3 not in self:
            mapping = AminoAcidDict()
            return [mapping[x] for x in self.seq1]
        return self[SeqNames.seq3]

    @property
    def categories(self) -> list[str]:
        """
        :return: A list of categories associated with this object.

        Categories are kept under "category" field in :attr:`meta`
        as a ","-separated list of strings. For instance, "domain,family_x".
        """
        cat = self.meta.get(MetaNames.category)
        return cat.split(",") if cat else []

    def _setup_and_validate(self):
        super()._setup_and_validate()

        if SeqNames.seq1 not in self:
            raise MissingData(f"Requires {SeqNames.seq1} in `seqs`")

        if not isinstance(self.seq1, str):
            try:
                self[SeqNames.seq1] = "".join(self.seq1)
            except Exception as e:
                raise InitError(
                    f"Failed to convert {SeqNames.seq1} "
                    f"from type {type(self.seq1)} to str "
                    f"due to: {e}"
                ) from e

        self.meta[MetaNames.id] = self.id
        self.meta[MetaNames.name] = self.name
        self.children: ChainList[ChainSequence] = _parse_children(self.children)

    def map_numbering(
        self,
        other: str | tuple[str, str] | ChainSequence | Alignment,
        align_method: AlignMethod = mafft_align,
        save: bool = True,
        name: t.Optional[str] = None,
        **kwargs,
    ) -> list[None | int]:
        """
        Map the :meth:`numbering`: of another sequence onto this one.
        For this, align primary sequences and relate their numbering.

        >>> s = ChainSequence.from_string('XXSEQXX', name='CS')
        >>> o = ChainSequence.from_string('SEQ', name='CSO')
        >>> s.map_numbering(o)
        [None, None, 1, 2, 3, None, None]
        >>> assert 'map_CSO' in s
        >>> a = Alignment([('CS1', 'XSEQX'), ('CS2', 'XXEQX')])
        >>> s.map_numbering(a, name='map_aln')
        [None, 1, 2, 3, 4, 5, None]
        >>> assert 'map_aln' in s

        :param other: another chain sequence.
        :param align_method: a method to use for alignment.
        :param save: save the numbering as a sequence.
        :param name: a name to use if `save` is ``True``.
        :param kwargs: passed to `func:map_pairs_numbering`.
        :return: a list of integers with ``None`` indicating gaps.
        """

        if isinstance(other, str):
            other = ChainSequence.from_string(other)
        elif isinstance(other, tuple):
            name = other[0]
            other = ChainSequence.from_string(other[1], name=name)

        if isinstance(other, ChainSequence):
            mapping = filter(
                lambda x: x[0] is not None,
                map_pairs_numbering(
                    self.seq1,
                    self.numbering,
                    other.seq1,
                    other.numbering,
                    align=True,
                    align_method=align_method,
                    **kwargs,
                ),
            )
            if not name:
                name = f"map_{other.name}"
        elif isinstance(other, Alignment):
            aligned_aln = other.align((name, self.seq1))
            aligned_other = aligned_aln[name]
            aligned_other_num = [
                i for (i, c) in enumerate(aligned_other, start=1) if c != "-"
            ]
            mapping = map_pairs_numbering(
                self.seq1,
                self.numbering,
                aligned_other,
                aligned_other_num,
                align=True,
                align_method=align_method,
                **kwargs,
            )
            if not name:
                name = SeqNames.map_aln
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        mapped_numbering = [x[1] for x in mapping]
        if save:
            self[name] = mapped_numbering

        return mapped_numbering

    def map_boundaries(
        self, start: Ord, end: Ord, map_name: str, closest: bool = False
    ) -> tuple[namedtuple, namedtuple]:
        """
        Map the provided boundaries onto sequence.

        A convenient interface for common task where one wants to find sequence
        elements corresponding to arbitrary boundaries.

        >>> s = ChainSequence.from_string('XXSEQXX', name='CS')
        >>> s.add_seq('NCS', list(range(10, 17)))
        >>> s.map_boundaries(1, 3, 'i')
        (Item(i=1, seq1='X', NCS=10), Item(i=3, seq1='S', NCS=12))
        >>> s.map_boundaries(5, 12, 'NCS', closest=True)
        (Item(i=1, seq1='X', NCS=10), Item(i=3, seq1='S', NCS=12))

        :param start: Some orderable object.
        :param end: Some orderable object.
        :param map_name: Use this sequence to search for boundaries.
            It is assumed that ``map_name in self is True``.
        :param closest: If true, instead of exact mapping, search for
            the closest elements.
        :return: a tuple with two items corresponding to mapped
            `start` and `end`.
        """
        if closest:
            mapping = list(filterfalse(lambda x: x is None, self[map_name]))
            map_min, map_max = min(mapping), max(mapping)
            reverse_start = start > map_max
            reverse_end = end > map_min
            _start, _end = starmap(
                lambda c, r: self.get_closest(map_name, c, reverse=r),
                [(start, reverse_start), (end, reverse_end)],
            )
            if _start is None or _end is None:
                raise AmbiguousMapping(
                    f"Failed mapping {(start, end)}->{(_start, _end)} "
                    f"using map {map_name}"
                )
        else:
            _start, _end = self.get_item(map_name, start), self.get_item(map_name, end)

        return _start, _end

    def relate(
        self,
        other: ChainSequence,
        map_name: str,
        link_name: str,
        link_points_to: str = "i",
        save: bool = True,
        map_name_in_other: str | None = None,
    ) -> list[t.Any]:
        """
        Relate mapping from this sequence with `other` via some common
        "link" sequence.

        The "link" sequence is a part of the `other` pointing to some sequence
        within this instance.

        To provide an example, consider the case of transferring the mapping
        to alignment positions `aln_map`. To do this, the `other` must
        be mapped to some sequence within this instance -- typically to
        the canonical numbering -- via some stored `map_canonical` sequence.

        Thus, one would use::

            this.transfer_to(
                other, map_name=aln_map,
                link_name=map_canonical, link_name_points_to="i")

        In the example below, we transfer `map_some` sequence from
        `s` to `o` via sequence `L` pointing to the primary sequence of `s`::

          seq1    : A B C D   ---|
          map_some: 9 8 7 6      | --> 9 8 None 6 (map transferred to `o`)
                    | | | |      |
          seq1    : X Y Z R      |
          L       : A B X D   ---|

        >>> s = ChainSequence.from_string('ABCD', name='CS')
        >>> s.add_seq('map_some', [9, 8, 7, 6])
        >>> o = ChainSequence.from_string('XYZR', name='XY')
        >>> o.add_seq('L', ['A', 'B', 'X', 'D'])
        >>> assert 'L' in o
        >>> s.relate(o, map_name='map_some', link_name='L', link_points_to='seq1')
        [9, 8, None, 6]
        >>> assert o['map_some'] == [9, 8, None, 6]

        :param other: An arbitrary chain sequence.
        :param map_name: The name of the sequence to transfer.
        :param link_name: The name of the "link" sequence that connects
            `self` and `other`.
        :param link_points_to: Values within this instance the "link" sequence
            points to.
        :param save: Store the obtained sequence within the `other`.
        :param map_name_in_other: The name of the mapped sequence to store
            within the `other`. By default, the `map_name` is used.
        :return: The mapped sequence.
        """
        mapping = self.get_map(link_points_to)
        mapped = list(
            map(
                lambda x: x if x is None else x._asdict()[map_name],
                (mapping.get(x) for x in other[link_name]),
            )
        )
        if save:
            other.add_seq(map_name_in_other or map_name, mapped)
        return mapped

    def coverage(
        self,
        map_names: abc.Sequence[str] | None = None,
        save: bool = True,
        prefix: str = "cov",
    ) -> dict[str, float]:
        """
        Calculate maps' coverage, i.e., the number of non-empty elements.

        :param map_names: optionally, provide the sequence of map names
            to calculate the coverage for.
        :param save: save the results to :attr:`meta`
        :param prefix: if `save` is ``True``, format keys f"{prefix}_{name}"
            for the :attr:`meta` dictionary.
        :return:
        """
        df = self.as_df()
        map_names = map_names or self.fields[3:]
        size = len(df)
        cov = {f"{prefix}_{n}": (~df[n].isna()).sum() / size for n in map_names}
        if save:
            self.meta.update(cov)
        return cov

    def get_map(self, key: str) -> dict[t.Any, namedtuple]:
        """
        Obtain the mapping of the form "key->item(seq_name=*,...)".

        >>> s = ChainSequence.from_string('ABC', name='CS')
        >>> s.get_map('i')
        {1: Item(i=1, seq1='A'), 2: Item(i=2, seq1='B'), 3: Item(i=3, seq1='C')}
        >>> s.get_map('seq1')
        {'A': Item(i=1, seq1='A'), 'B': Item(i=2, seq1='B'), 'C': Item(i=3, seq1='C')}

        :param key: map name.
        :return: `dict` mapping key values to items.
        """
        keys = (x.i for x in self) if key == "i" else self[key]
        return dict(zip(keys, iter(self)))

    def get_item(self, key: str, value: t.Any) -> namedtuple:
        """
        Get a specific item. Same as :meth:`get_map`, but uses `value`
        to retrieve the needed item immediately.

        **(!) Use it when a single item is needed.** For multiple queries
        for the same sequence, please use :meth:`get_map`.

        >>> s = ChainSequence.from_string('ABC', name='CS')
        >>> s.get_item('seq1', 'B').i
        2

        :param key: map name.
        :param value: sequence value of the sequence under the `key` name.
        :return: an item correpsonding to the desired sequence element.
        """
        return self.get_map(key)[value]

    def get_closest(
        self, key: str, value: Ord, *, reverse: bool = False
    ) -> t.Optional[namedtuple]:
        """
        Find the closest item for which item.key ``>=/<=`` value.
        By default, the search starts from the sequence's beginning,
        and expands towards the end until the first element for which
        the retrieved `value` >= the provided `value`.
        If the `reverse` is ``True``, the search direction is reversed,
        and the comparison operator becomes ``<=``

        >>> s = ChainSequence(1, 4, 'CS', seqs={'seq1': 'ABCD', 'X': [5, 6, 7, 8]})
        >>> s.get_closest('seq1', 'D')
        Item(i=4, seq1='D', X=8)
        >>> s.get_closest('X', 0)
        Item(i=1, seq1='A', X=5)
        >>> assert s.get_closest('X', 0, reverse=True) is None

        :param key: map name.
        :param value: map value. Must support comparison operators.
        :param reverse: reverse the sequence order and the comparison operator.
        :return: The first relevant item or `None` if no relevant items
            were found.
        """

        def pred(kv: tuple[Ord | None, t.Any]) -> bool:
            if kv[0] is None:
                return False
            if reverse:
                return kv[0] <= value
            return kv[0] >= value

        items = self.get_map(key).items()
        if reverse:
            items = reversed(items)
        result = first_true(items, default=None, pred=pred)
        if result:
            return result[1]
        return None

    # @lru_cache
    def as_df(self) -> pd.DataFrame:
        """
        :return: The pandas DataFrame representation of the sequence where
            each column correspond to a sequence or map.
        """
        return pd.DataFrame(iter(self))

    # @lru_cache
    def as_np(self) -> np.ndarray:
        """
        :return: The numpy representation of a sequence as matrix.
            This is a shortcut to :meth:`as_df` and getting `df.values`.
        """
        return self.as_df().values

    def spawn_child(
        self,
        start: int,
        end: int,
        name: str | None = None,
        category: str | None = None,
        *,
        map_from: t.Optional[str] = None,
        map_closest: bool = False,
        deep_copy: bool = False,
        keep: bool = True,
    ) -> ChainSequence:
        """
        Spawn the sub-sequence from the current instance.

        Child sequence's boundaries must be within this sequence's boundaries.

        Uses :meth:`Segment.sub` method.

        >>> s = ChainSequence(
        ...     1, 4, 'CS',
        ...     seqs={'seq1': 'ABCD', 'X': [5, 6, 7, 8]}
        ... )
        >>> child1 = s.spawn_child(1, 3, 'Child1')
        >>> assert child1.id in s.children
        >>> s.children
        [Child1|1-3<-(CS|1-4)]

        :param start: Start of the sub-sequence.
        :param end: End of the sub-sequence.
        :param name: Spawned child sequence's name.
        :param map_from: Optionally, the map name the boundaries correspond to.
        :param map_closest: Map to closest `start`, `end` boundaries
            (see :meth:`map_boundaries`).
        :param deep_copy: Deep copy inherited sequences.
        :param keep: Save child sequence within :attr:`children`.
        :return: Spawned sub-sequence.
        """
        if map_from:
            start, end = map(
                lambda x: x._asdict()["i"],
                self.map_boundaries(start, end, map_from, map_closest),
            )

        name = name or self.name

        # TODO: there is no point in transferring meta info whatsoever --> make new
        child = self.sub(start, end, deep_copy=deep_copy, handle_mode="self")
        child.name = name
        child.meta[MetaNames.name] = name
        child.meta[MetaNames.id] = child.id

        if category:
            add_category(child, category)

        if keep:
            self.children.append(child)

        return child

    @classmethod
    def from_file(
        cls,
        inp: Path | TextIOBase | abc.Iterable[str],
        reader: SeqReader = read_fasta,
        start: t.Optional[int] = None,
        end: t.Optional[int] = None,
        name: t.Optional[str] = None,
        meta: dict[str, t.Any] | None = None,
        **kwargs,
    ) -> ChainSequence:
        """
        Initialize chain sequence from file.

        :param inp: Path to a file or file handle or iterable over file lines.
        :param reader: A function to parse the sequence from `inp`.
        :param start: Start coordinate of a sequence in a file.
            If not provided, assumed to be 1.
        :param end: End coordinate of a sequence in a file.
            If not provided, will evaluate to the sequence's length.
        :param name: Name of a sequence in `inp`.
            If not provided, will evaluate to a sequence's header.
        :param meta: Meta-info to add for the sequence.
        :param kwargs: Additional sequences other than `seq1`
            (as used during initialization via `seq` attribute).
        :return: Initialized chain sequence.
        """
        seqs = list(reader(inp))
        if not seqs:
            raise MissingData("No sequences in provided inp")
        if len(seqs) > 1:
            raise ValueError("Input contains more than one sequence")

        seq = seqs.pop()

        start = start or 1
        end = end or start + len(seq[1]) - 1

        if name is None:
            name = seq[0]

        return cls(start, end, name, meta=meta, seqs={SeqNames.seq1: seq[1], **kwargs})

    @classmethod
    def from_string(
        cls,
        s: str,
        start: t.Optional[int] = None,
        end: t.Optional[int] = None,
        name: t.Optional[str] = None,
        meta: dict[str, t.Any] | None = None,
        **kwargs,
    ) -> ChainSequence:
        """
        Initialize chain sequence from string.

        :param s: String to init from.
        :param start: Start coordinate (default=1).
        :param end: End coordinate(default=len(s)).
        :param name: Name of a new chain sequence.
        :param meta: Meta info of a new sequence.
        :param kwargs: Additional sequences other than `seq1`
            (as used during initialization via `seq` attribute).
        :return: Initialized chain sequence.
        """
        start = start or 1
        end = end or start + len(s) - 1

        return cls(start, end, name, meta=meta, seqs={SeqNames.seq1: s, **kwargs})

    @classmethod
    def from_df(
        cls,
        df: Path | pd.DataFrame,
        name: t.Optional[str] = None,
        meta: dict[str, t.Any] | None = None,
    ):
        """
        Init sequence from a data frame.

        :param df: Path to a tsv file or a pandas DataFrame.
        :param name: Name of a new chain sequence.
        :param meta: Meta info of a new chain sequence.
        :return: Initialized chain sequence.
        """
        if isinstance(df, Path):
            df = pd.read_csv(df, sep="\t")
        if "i" not in df.columns:
            raise InitError('Must contain the "i" column')
        assert len(df) >= 1
        start, end = df["i"].iloc[0], df["i"].iloc[-1]
        seqs = {col: list(df[col]) for col in df.columns if col != "i"}
        return cls(start, end, name, meta=meta, seqs=seqs)

    @classmethod
    def read(
        cls,
        base_dir: Path,
        *,
        dump_names: DumpNames = DumpNames,
        search_children: bool = False,
    ) -> ChainSequence:
        """
        Initialize chain sequence from dump created using :meth:`write`.

        :param base_dir: A path to a dump dir.
        :param search_children: Recursively search for child segments and
            populate the :attr:`children`
        :param dump_names: A container (dataclass) with filenames.
        :return: Initialized chain sequence.
        """
        files = get_files(base_dir)
        dirs = get_dirs(base_dir)

        if dump_names.sequence not in files:
            raise InitError(f"{dump_names.sequence} must be present")

        if dump_names.meta in files:
            df = pd.read_csv(
                files[dump_names.meta], sep=r"\s+", names=["Title", "Value"]
            )
            meta = dict(zip(df["Title"], df["Value"]))
            if MetaNames.name in meta:
                name = meta[MetaNames.name]
            else:
                name = "UnnamedSequence"
        else:
            meta, name = {}, "UnnamedSequence"

        df = pd.read_csv(files[dump_names.sequence], sep="\t")
        seq = cls.from_df(df, name)
        seq.meta = meta

        if dump_names.variables in files:
            seq.variables = Variables.read(files[dump_names.variables]).sequence

        if search_children and dump_names.segments_dir in dirs:
            for path in (base_dir / dump_names.segments_dir).iterdir():
                child = ChainSequence.read(
                    path, dump_names=dump_names, search_children=True
                )
                child.parent = seq
                seq.children.append(child)

        return seq

    def write_seq(
        self, path: Path, fields: t.Optional[t.Container[str]] = None, sep: str = "\t"
    ) -> t.NoReturn:
        """
        Write the sequence (and all its maps) as a table.

        :param path: Write destination file.
        :param fields: Optionally, names of sequences to dump.
        :param sep: Table separator. Please use the default to avoid ambiguities
            and keep readability.
        :return: Nothing.
        """
        self.as_df().drop_duplicates().to_csv(
            path, index=False, columns=fields, sep=sep
        )

    def write_meta(self, path: Path, sep="\t") -> t.NoReturn:
        """
        Write meta information as {key}{sep}{value} lines.

        :param path: Write destination file.
        :param sep: Separator between key and value.
        :return: Nothing.
        """
        items = (
            f"{k}{sep}{v}"
            for k, v in self.meta.items()
            if isinstance(v, (str, int, float))
        )
        path.write_text("\n".join(items))

    def write(
        self,
        base_dir: Path,
        *,
        dump_names: DumpNames = DumpNames,
        write_children: bool = False,
    ) -> t.NoReturn:
        """
        Dump this chain sequence. Creates `sequence.tsv` and `meta.tsv`
        in `base_dir` using :meth:`write_seq` and :meth:`write_meta`.

        :param base_dir: Destination directory.
        :param dump_names: A container (dataclass) with filenames.
        :param write_children: Recursively write children.
        :return: Nothing.
        """
        base_dir.mkdir(exist_ok=True, parents=True)
        self.write_seq(base_dir / dump_names.sequence)
        if self.meta:
            self.write_meta(base_dir / dump_names.meta)
            if self.variables:
                self.variables.write(base_dir / dump_names.variables)
        if write_children:
            for child in self.children:
                child_dir = base_dir / dump_names.segments_dir / child.name
                child.write(
                    child_dir, dump_names=dump_names, write_children=write_children
                )


class ChainStructure:
    """
    A structure of a single chain.

    Typical usage workflow:

    1) Use :meth:`GenericStructure.read <lXtractor.core.structure.
    GenericStructure.read>` to parse the file.
    2) Split into chains using :meth:`split_chains <lXtractor.core.structure.
    GenericStructure.split_chains>`.
    3) Initialize :class:`ChainStructure` from each chain via
    :meth:`from_structure`.


    .. code-block:: python

        s = GenericStructure.read(Path("path/to/structure.cif"))
        chain_structures = [
            ChainStructure.from_structure(c) for c in s.split_chains()
        ]

    Two main containers are:

    1) :attr:`seq` -- a :class:`ChainSequence` of this structure,
    also containing meta info.
    2) :attr:`pdb` -- a container with pdb id, pdb chain id,
    and the structure itself.
    """

    __slots__ = ("pdb", "seq", "parent", "variables", "children")

    def __init__(
        self,
        pdb_id: str,
        pdb_chain: str,
        pdb_structure: GenericStructure,
        seq: ChainSequence | None = None,
        parent: ChainStructure | None = None,
        children: abc.Sequence[ChainStructure]
        | ChainList[ChainStructure]
        | None = None,
        variables: t.Optional[Variables] = None,
    ):
        """
        `pdb_id`, `pdb_chain`, and `pdb_structure` are wrapped into a
        :attr:`pdb`: -- a :class:`PDB_Chain` container.

        :param pdb_id: Four-letter PDB code.
        :param pdb_chain: PDB Chain code.
        :param pdb_structure: Parsed generic structure with a single chain.
        :param seq: Chain sequence of a structure. If not provided, will use
            :meth:`get_sequence <lXtractor.core.structure.
            GenericStructure.get_sequence>`.
        :param parent: Specify parental structure.
        :param children: Specify structures descended from this one.
            This contained is used to record sub-structures obtained via
            :meth:`spawn_child`.
        :param variables: Variables associated with this structure.
        :raise InitError: If invalid (e.g., multi-chain structure) is provided.
        """
        #: A container with PDB ID, PDB Chain, and parsed structure.
        self.pdb: PDB_Chain = PDB_Chain(pdb_id, pdb_chain, pdb_structure)
        _validate_chain(self.pdb)

        #: Sequence of this structure.
        self.seq: ChainSequence | None = seq

        #: Parent of this structure.
        self.parent: ChainStructure | None = parent

        #: Variables assigned to this structure. Each should be of a
        #: :class:`lXtractor.variables.base.StructureVariable`.
        self.variables: Variables = variables or Variables()

        #: Any sub-structures descended from this one,
        #: preferably using :meth:`spawn_child`.
        self.children: ChainList[ChainStructure] = _parse_children(children)

        if self.seq is None:
            seq1, seq3, num = map(list, unzip(self.pdb.structure.get_sequence()))
            seqs = {SeqNames.seq3: seq3, SeqNames.enum: num}
            self.seq = ChainSequence.from_string(
                "".join(seq1), name=f"{pdb_id}{Sep.chain}{pdb_chain}", **seqs
            )

        self.seq.meta[MetaNames.pdb_id] = pdb_id
        self.seq.meta[MetaNames.pdb_chain] = pdb_chain

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def iter_children(self) -> abc.Generator[list[ChainStructure]]:
        """
        Iterate :attr:`children` in topological order.

        See :meth:`ChainSequence.iter_children` and :func:`topo_iter`.
        """
        return topo_iter(self, lambda x: x.children)

    @property
    def id(self) -> str:
        """
        :return: Unique identifier of a structure.
        """
        return f"{self.__class__.__name__}({self.seq})"

    @property
    def array(self) -> bst.AtomArray:
        """
        :return: The ``AtomArray`` object (a shortcut for
            ``.pdb.structure.array``).
        """
        return self.pdb.structure.array

    @property
    def meta(self) -> dict[str, str]:
        """
        :return: Meta info of a :attr:`seq`.
        """
        return self.seq.meta

    @property
    def categories(self) -> list[str]:
        """
        :return: A list of categories encapsulated within
            :attr:`ChainSequence.meta`.
        """
        return self.seq.categories

    @classmethod
    def from_structure(
        cls,
        structure: bst.AtomArray | GenericStructure,
        pdb_id: t.Optional[str] = None,
    ) -> ChainStructure:
        """
        :param structure: An `AtomArray` or `GenericStructure`,
            corresponding to a single protein chain.
        :param pdb_id: PDB identifier of a structure
            (Chain ID will be inferred from the `AtomArray`).
        :return: Initialized chain structure.
        """
        if isinstance(structure, bst.AtomArray):
            structure = GenericStructure(structure)

        chain_id = structure.array.chain_id[0]

        if pdb_id is None:
            pdb_id = structure.pdb_id

        return cls(pdb_id, chain_id, structure)

    def superpose(
        self,
        other: ChainStructure,
        res_id: abc.Sequence[int] | None = None,
        atom_names: abc.Sequence[abc.Sequence[str]] | abc.Sequence[str] | None = None,
        map_name_self: str | None = None,
        map_name_other: str | None = None,
        mask_self: np.ndarray | None = None,
        mask_other: np.ndarray | None = None,
        inplace: bool = False,
        rmsd_to_meta: bool = True,
    ) -> tuple[ChainStructure, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Superpose some other structure to this one.
        It uses func:`biotite.structure.superimpose` internally.

        The most important requirement is both structures (after all optional
        selections applied) having the same number of atoms.

        :param other: Other chain structure (mobile).
        :param res_id: Residue positions within this or other chain structure.
            If ``None``, use all available residues.
        :param atom_names: Atom names to use for selected residues. Two options
            are available:

            1) Sequence of sequences of atom names. In this case, atom names
            are given per selected residue (`res_id`), and the external
            sequence's length must correspond to the number of residues in the
            `res_id`. Note that if no `res_id` provided, the sequence must
            encompass all available residues.

            2) A sequence of atom names. In this case, it will be used
            to select atoms for each available residues. For instance, use
            ``atom_names=["CA", "C", "N"]`` to select backbone atoms.

        :param map_name_self: Use this map to map `res_id` to real numbering
            of this structure.
        :param map_name_other: Use this map to map `res_id` to real numbering
            of the `other` structure.
        :param mask_self: Per-atom boolean selection mask to pick fixed atoms
            within this structure.
        :param mask_other: Per-atom boolean selection mask to pick mobile atoms
            within the `other` structure. Note that `mask_self` and
            `mask_other` take precedence over other selection specifications.
        :param inplace: Apply the transformation to the mobile structure
            inplace, mutating `other`. Otherwise, make a new instance:
            same as `other`, but with transformed atomic coordinates of
            a :attr:`pdb.structure`.
        :param rmsd_to_meta: Write RMSD to the :attr:`meta` of `other` as "rmsd
        :return: A tuple with (1) transformed chain structure,
            (2) transformation RMSD, and (3) transformation matrices
            (see func:`biotite.structure.superimpose` for details).
        """

        def _get_mask(c: ChainStructure, map_name: str) -> np.ndarray:
            if res_id is None:
                return np.ones_like(c.array, bool)
            _res_id = (
                res_id
                if not map_name or res_id is None
                else [
                    c.seq.get_item(map_name, x)._asdict()[SeqNames.enum] for x in res_id
                ]
            )
            return filter_selection(c.array, _res_id, atom_names)

        match atom_names:
            case [str(), *_]:
                if res_id is not None:
                    atom_names = [atom_names] * len(res_id)
            case [[str(), *_], *_]:
                if len(res_id) != len(atom_names):
                    raise LengthMismatch(
                        "When specifying `atom_names` per residue, the number of "
                        f"residues must match the number of atom name groups; "
                        f"Got {len(res_id)} residues and {len(atom_names)} "
                        "atom names groups."
                    )

        if mask_self is None:
            mask_self = _get_mask(self, map_name_self)
        if mask_other is None:
            mask_other = _get_mask(other, map_name_other)

        superposed, rmsd, transformation = self.pdb.structure.superpose(
            other.pdb.structure, mask_self=mask_self, mask_other=mask_other
        )

        if inplace:
            other.pdb.structure = superposed
        else:
            other = ChainStructure(
                other.pdb.id,
                other.pdb.chain,
                superposed,
                other.seq,
                parent=other.parent,
                children=other.children,
                variables=other.variables,
            )

        if rmsd_to_meta:
            # TODO: Is this correct?
            map_name = map_name_other or other.seq.name
            other.seq.meta[f"rmsd_{map_name}"] = rmsd

        return other, rmsd, transformation

    def spawn_child(
        self,
        start: int,
        end: int,
        name: t.Optional[str] = None,
        *,
        map_from: t.Optional[str] = None,
        map_closest: bool = True,
        keep_seq_child: bool = False,
        keep: bool = True,
        deep_copy: bool = False,
    ) -> ChainStructure:
        """
        Create a sub-structure from this one.
        `Start` and `end` have inclusive boundaries.

        :param start: Start coordinate.
        :param end: End coordinate.
        :param name: The name of the spawned sub-structure.
        :param map_from: Optionally, the map name the boundaries correspond to.
        :param map_closest: Map to closest `start`, `end` boundaries
            (see :meth:`map_boundaries`).
        :param keep_seq_child: Keep spawned sub-sequence within
            :attr:`ChainSequence.children`. Beware that it's best to use
            a single object type for keeping parent-children relationships
            to avoid duplicating information.
        :param keep: Keep spawned substructure in :attr:`children`.
        :param deep_copy: Deep copy spawned sub-sequence and sub-structure.
        :return: New chain structure -- a sub-structure of the current one.
        """

        if start > end:
            raise ValueError(f"Invalid boundaries {start, end}")

        name = name or self.seq.name

        seq = self.seq.spawn_child(
            start,
            end,
            name,
            map_from=map_from,
            map_closest=map_closest,
            deep_copy=deep_copy,
            keep=keep_seq_child,
        )
        structure = None
        if self.pdb.structure:
            enum_field = seq.field_names().enum
            start, end = seq[enum_field][0], seq[enum_field][-1]
            structure = self.pdb.structure.sub_structure(start, end)

        child = ChainStructure(self.pdb.id, self.pdb.chain, structure, seq, self)
        if keep:
            self.children.append(child)
        return child

    @classmethod
    def read(
        cls,
        base_dir: Path,
        *,
        dump_names: DumpNames = DumpNames,
        search_children: bool = False,
    ) -> ChainStructure:
        """
        Read the chain structure from a file disk dump.

        :param base_dir: An existing dir containing structure,
            structure sequence, meta info, and (optionally) any sub-structure
            segments.
        :param dump_names: File names container.
        :param search_children: Recursively search for sub-segments and
            populate :attr:`children`.
        :return: An initialized chain structure.
        """

        files = get_files(base_dir)
        dirs = get_dirs(base_dir)
        variables = None

        bname = dump_names.structure_base_name
        stems = {p.stem: p.name for p in files.values()}
        if bname not in stems:
            raise InitError(
                f"{base_dir} must contain {bname}.fmt "
                f'where "fmt" is supported structure format'
            )
        structure = GenericStructure.read(base_dir / stems[bname])

        seq = ChainSequence.read(base_dir, dump_names=dump_names, search_children=False)
        pdb_id = seq.meta.get(MetaNames.pdb_id, "UnkPDB")
        chain_id = seq.meta.get(MetaNames.pdb_chain, "UnkChain")

        if dump_names.variables in files:
            variables = Variables.read(files[dump_names.variables]).structure

        cs = ChainStructure(pdb_id, chain_id, structure, seq, variables=variables)

        if search_children and dump_names.segments_dir in dirs:
            for path in (base_dir / dump_names.segments_dir).iterdir():
                child = ChainStructure.read(
                    path, dump_names=dump_names, search_children=True
                )
                child.parent = cs
                cs.children.append(child)

        return cs

    def write(
        self,
        base_dir: Path,
        fmt: str = "cif",
        *,
        dump_names: DumpNames = DumpNames,
        write_children: bool = False,
    ) -> t.NoReturn:
        """
        Dump chain structure to disk.

        :param base_dir: A writable dir to save files to.
        :param fmt: The format of the structure to use
            -- any format supported by `biotite`.
        :param dump_names: File names container.
        :param write_children: Recursively write :attr:`children`.
        :return: Nothing.
        """

        base_dir.mkdir(exist_ok=True, parents=True)

        self.seq.write(base_dir)
        self.pdb.structure.write(base_dir / f"{dump_names.structure_base_name}.{fmt}")
        if self.variables:
            self.variables.write(base_dir / dump_names.variables)

        if write_children:
            for child in self.children:
                child_dir = base_dir / DumpNames.segments_dir / child.seq.name
                child.write(child_dir, fmt, dump_names=dump_names, write_children=True)


class Chain(AbstractChain):
    """
    A container, encompassing a :class:`ChainSequence` and possibly many
    :class:`ChainStructure`'s corresponding to a single protein chain.

    A typical use case is when one wants to benefit from the connection
    of structural and sequential data, e.g., using single full canonical
    sequence as :attr:`seq` and all the associated structures within
    :attr:`structures`. In this case, this data structure makes it easier
    to extract, annotate, and calculate variables using canonical sequence
    mapped to the sequence of a structure.

    Typical workflow:

        1) Initialize from some canonical sequence.
        2) Add structures and map their sequences.
        3) ???
        4) Do something useful, like calculate variables using canonical
        sequence's positions.

    .. code-block:: python

        c = Chain.from_sequence((header, seq))
        for s in structures:
            c.add_structure(s)

    """

    __slots__ = ("seq", "structures", "parent", "children")

    def __init__(
        self,
        seq: ChainSequence,
        structures: abc.Iterable[ChainStructure] | None = None,
        parent: Chain | None = None,
        children: abc.Sequence[Chain] | None = None,
    ):
        """

        :param seq: A chain sequence.
        :param structures: Chain structures corresponding to a single protein
            chain specified by `seq`.
        :param parent: A parent chain this chain had descended from.
        :param children: A collection of children.
        """
        #: A chain sequence.
        self.seq: ChainSequence = seq

        #: A collection of structures corresponding to :attr:`seq`.
        if structures is None:
            structures = ChainList([])
        else:
            if not isinstance(structures, ChainList):
                structures = ChainList(structures)
        self.structures: ChainList[ChainStructure] = structures

        #: A parent chain this chain had descended from.
        self.parent = parent

        #: A collection of children preferably obtained using
        #: :meth:`spawn_child`.
        self.children: ChainList[Chain] = _parse_children(children)

    @property
    def id(self) -> str:
        """
        :return: Unique identifier: same as :attr:`seq`'s id.
        """
        return self.seq.id

    @property
    def meta(self) -> dict[str, str]:
        """
        :return: A :attr:`seq`'s :attr:`ChainSequence.meta`.
        """
        return self.seq.meta

    @property
    def categories(self) -> list[str]:
        """
        :return: A list of categories from :attr:`seq`'s
            :attr:`ChainSequence.meta`.
        """
        return self.seq.categories

    def __repr__(self) -> str:
        return self.id

    def __str__(self) -> str:
        return self.id

    # def __getitem__(self, key: str | int) -> Chain:
    #     if isinstance(key, str):
    #         return self.children[key]
    #     if isinstance(key, int):
    #         return list(self.children.values())[key]
    #     else:
    #         raise TypeError('Wrong key type')

    # def __contains__(self, item: Chain) -> bool:
    #     return item in self.children

    def iter_children(self) -> abc.Generator[list[Chain]]:
        """
        Iterate :attr:`children` in topological order.

        See :meth:`ChainSequence.iter_children` and :func:`topo_iter`.

        :return: Iterator over levels of a child tree.
        """
        return topo_iter(self, lambda x: x.children)

    @t.overload
    def from_seq(
        self,
        inp: Path | TextIOBase | abc.Iterable[str],
        read_method: SeqReader = read_fasta,
    ) -> ChainList[Chain]:
        ...

    @t.overload
    def from_seq(
        self, inp: str | tuple[str, str], read_method: SeqReader = read_fasta
    ) -> Chain:
        ...

    @classmethod
    def from_seq(
        cls,
        inp: str | tuple[str, str] | Path | TextIOBase | abc.Iterable[str],
        read_method: SeqReader = read_fasta,
    ) -> Chain | ChainList[Chain]:
        """

        :param inp: A string of with a sequence or a pair (header, seq).
            Otherwise, something that the `read_method` accepts.
        :param read_method: A callable accepting a path to a file or opened
            file or an iterable over the file lines and returning pairs
            (header, seq).
        :return: If a single sequence is provided as a string or pair,
            return an initialized chain. Otherwise, use `read_method` to parse
            the input and embed the resulting :class:`Chain`'s into
            a :class:`ChainList`.
        """
        match inp:
            case str():
                return cls(ChainSequence.from_string(inp))
            case [header, seq]:
                return cls(ChainSequence.from_string(seq, name=header))
            case _:
                return ChainList(
                    cls(ChainSequence.from_string(seq, name=name))
                    for name, seq in read_method(inp)
                )

    @classmethod
    def read(
        cls,
        path: Path,
        *,
        dump_names: DumpNames = DumpNames,
        search_children: bool = False,
    ) -> Chain:
        """
        :param path: A path to a directory with at least sequence and
            metadata files.
        :param dump_names: File names container.
        :param search_children: Recursively search for child segments and
            populate :attr:`children`.
        :return: An initialized chain.
        """
        seq = ChainSequence.read(path, dump_names=dump_names, search_children=False)

        structures = [
            ChainStructure.read(p, dump_names=dump_names)
            for p in (path / dump_names.structures_dir).glob("*")
        ]
        c = Chain(seq, structures)
        if search_children:
            for child_path in (path / dump_names.segments_dir).glob("*"):
                child = Chain.read(
                    child_path, dump_names=dump_names, search_children=True
                )
                c.children.append(child)
        return c

    def write(
        self,
        base_dir: Path,
        *,
        dump_names: DumpNames = DumpNames,
        str_fmt: str = "cif",
        write_children: bool = True,
    ) -> t.NoReturn:
        """
        Create a disk dump of this chain data.
        Created dumps can be reinitialized via :meth:`read`.

        :param base_dir: A writable dir to hold the data.
        :param dump_names: A file names container.
        :param str_fmt: A format to write :attr:`structures` in.
        :param write_children: Recursively write :attr:`children`.
        :return: Nothing.
        """
        base_dir.mkdir(parents=True, exist_ok=True)

        self.seq.write(base_dir, dump_names=dump_names, write_children=False)

        if self.structures:
            str_dir = base_dir / dump_names.structures_dir
            str_dir.mkdir(exist_ok=True)
            for s in self.structures:
                s.write(
                    str_dir / s.id, str_fmt, dump_names=dump_names, write_children=False
                )

        for c in self.children:
            c.write(
                base_dir / dump_names.segments_dir / c.id,
                dump_names=dump_names,
                str_fmt=str_fmt,
                write_children=write_children,
            )

    def add_structure(
        self,
        structure: ChainStructure,
        *,
        check_ids: bool = True,
        map_to_seq: bool = True,
        map_name: str = SeqNames.map_canonical,
        **kwargs,
    ) -> t.NoReturn:
        """
        Add a structure to :attr:`structures`.

        :param structure: A structure of a single chain corresponding to
            :attr:`seq`.
        :param check_ids: Check that existing :attr:`structures`
            don't encompass the structure with the same :meth:`id`.
        :param map_to_seq: Align the structure sequence to the :attr:`seq` and
            create a mapping within the former.
        :param map_name: If `map_to_seq` is ``True``, use this map name.
        :param kwargs: Passed to :meth:`ChainSequence.map_numbering`.
        :return: Mutates :attr:`structures` and returns nothing.
        :raise ValueError: If `check_ids` is ``True`` and the structure
            id clashes with the existing ones.
        """
        if check_ids:
            ids = [s.id for s in self.structures]
            if structure.id in ids:
                raise ValueError(
                    f"Protein already contains structure {structure.id}. "
                    f"Remove it first or disable `check_ids`"
                )
        if map_to_seq:
            structure.seq.map_numbering(self.seq, name=map_name, **kwargs)
        self.structures.append(structure)

    def transfer_seq_mapping(
        self,
        map_name: str,
        link_map: str = SeqNames.map_canonical,
        link_map_points_to: str = "i",
        **kwargs,
    ) -> t.NoReturn:
        """
        Transfer sequence mapping to each :attr:`ChainStructure.seq` within
        :attr:`structures`.

        This method simply utilizes :meth:`ChainSequence.relate` to transfer
        some map from the :attr:`seq` to each :attr:`ChainStructure.seq`.
        Check :meth:`ChainSequence.relate` for an explanation.

        :param map_name: The name of the map to transfer.
        :param link_map: A name of the map existing within
            :attr:`ChainStructure.seq` of each structure in :attr:`structures`.
        :param link_map_points_to: Which sequence values of the `link_map`
            point to.
        :param kwargs: Passed to :meth:`ChainSequence.relate`
        :return: Nothing.
        """
        for s in self.structures:
            self.seq.relate(s.seq, map_name, link_map, link_map_points_to, **kwargs)

    def spawn_child(
        self,
        start: int,
        end: int,
        name: None | str = None,
        *,
        subset_structures: bool = True,
        tolerate_failure: bool = False,
        keep: bool = True,
        seq_deep_copy: bool = False,
        seq_map_from: t.Optional[str] = None,
        seq_map_closest: bool = True,
        seq_keep_child: bool = False,
        str_deep_copy: bool = False,
        str_map_from: t.Optional[str] = None,
        str_map_closest: bool = True,
        str_keep_child: bool = False,
        str_seq_keep_child: bool = False,
    ) -> Chain:
        """
        Subset a :attr:`seq` and (optionally) each structure in
        :attr:`structures` using the provided :attr:`seq` boundaries
        (inclusive).

        :param start: Start coordinate.
        :param end: End coordinate.
        :param name: Name of a new chain.
        :param subset_structures: If ``True``, subset each structure in
            :attr:`structures`. If ``False``, structures are not inherited.
        :param tolerate_failure: If ``True``, a failure to subset a structure
            doesn't raise an error.
        :param keep: Save created child to :attr:`children`.
        :param seq_deep_copy: Deep copy potentially mutable sequences within
            :attr:`seq`.
        :param seq_map_from: Use this map to obtain coordinates within
            :attr:`seq`.
        :param seq_map_closest: Map to the closest matching coordinates of
            a :attr:`seq`. See :meth:`ChainSequence.map_boundaries`
            and :meth:`ChainSequence.find_closest`.
        :param seq_keep_child: Keep a spawned :class:`ChainSequence` as a child
            within :attr:`seq`. Should be ``False`` if `keep` is ``True`` to
            avoid data duplication.
        :param str_deep_copy: Deep copy each sub-structure.
        :param str_map_from: Use this map to obtain coordinates within
            :attr:`ChainStructure.seq` of each structure.
        :param str_map_closest: Map to the closest matching coordinates of
            a :attr:`seq`. See :meth:`ChainSequence.map_boundaries`
            and :meth:`ChainSequence.find_closest`.
        :param str_keep_child: Keep a spawned sub-structure as a child in
            :attr:`ChainStructure.children`. Should be ``False`` if `keep` is
            ``True`` to avoid data duplication.
        :param str_seq_keep_child: Keep a sub-sequence of a spawned structure
            within the :attr:`ChainSequence.children` of
            :attr:`ChainStructure.seq` of a spawned structure. Should be
            ``False`` if `keep` or `str_keep_child` is ``True`` to avoid
            data duplication.
        :return: A sub-chain with sub-sequence and (optionally) sub-structures.
        """

        def subset_structure(structure: ChainStructure) -> t.Optional[ChainStructure]:
            try:
                return structure.spawn_child(
                    start,
                    end,
                    name,
                    map_from=str_map_from,
                    map_closest=str_map_closest,
                    deep_copy=str_deep_copy,
                    keep=str_keep_child,
                    keep_seq_child=str_seq_keep_child,
                )
            except (AmbiguousMapping, MissingData, NoOverlap) as e:
                msg = (
                    "Failed to spawn substructure using boundaries "
                    f"{start, end} due to {e}"
                )
                if not tolerate_failure:
                    raise e
                LOGGER.warning(msg)
                return None

        name = name or self.seq.name

        seq = self.seq.spawn_child(
            start,
            end,
            name,
            map_from=seq_map_from,
            map_closest=seq_map_closest,
            deep_copy=seq_deep_copy,
            keep=seq_keep_child,
        )

        structures = None
        if subset_structures:
            structures = list(filter(bool, map(subset_structure, self.structures)))

        child = Chain(seq, structures, self)
        if keep:
            self.children.append(child)
        return child


CT = t.TypeVar("CT", bound=Chain | ChainSequence | ChainStructure)  # "Chain" type
SS = t.TypeVar("SS", bound=ChainSequence | ChainStructure)
ST = t.TypeVar("ST", bound=Segment)


def add_category(c: CT, cat: str):
    """
    :param c: A Chain*-type object.
    :param cat:
    :return:
    """
    meta = c.seq.meta if isinstance(c, ChainStructure) else c.meta
    field = MetaNames.category
    if field not in meta:
        meta[field] = cat
    else:
        existing = meta[field].split(",")
        if cat not in existing:
            meta[field] += f",{cat}"


class ChainList(abc.MutableSequence[CT]):
    # TODO: consider implementing pattern-based search over whole sequence
    # or a sequence region.

    # For the above, consider filtering to hits.
    # It may be beneficial to implement this functionality for ChainSequence.
    """
    A mutable single-type collection holding either :class:`Chain`'s,
    or :class:`ChainSequence`'s, or :class:`ChainStructure`'s.

    Object's funtionality relies on this type purity.
    Adding of / contatenating with objects of a different type shall
    raise an error.

    It behaves like a regular list with additional functionality.

    >>> s = ChainSequence.from_string('SEQUENCE', name='S')
    >>> x = ChainSequence.from_string('XXX', name='X')
    >>> x.meta['category'] = 'x'
    >>> cl = ChainList([s, s, x])
    >>> cl
    [S|1-8, S|1-8, X|1-3]
    >>> cl[0]
    S|1-8
    >>> cl['S']
    [S|1-8, S|1-8]
    >>> cl[:2]
    [S|1-8, S|1-8]
    >>> cl['1-3']
    [X|1-3]

    Adding/appending/removing objects of a similar type is easy and works
    similar to a regular list.

    >>> cl += [s]
    >>> assert len(cl) == 4
    >>> cl.remove(s)
    >>> assert len(cl) == 3

    Categories can be accessed as attributes or using ``[]`` syntax
    (similar to the `Pandas.DataFrame` columns).

    >>> cl.x
    [X|1-3]
    >>> cl['x']
    [X|1-3]

    While creating a chain list, using a `groups` parameter will assign
    categories to sequences.
    Note that such operations return a new :class:`ChainList` object.

    >>> cl = ChainList([s, x], categories=['S', ['X1', 'X2']])
    >>> cl.S
    [S|1-8]
    >>> cl.X2
    [X|1-3]
    >>> cl['X1']
    [X|1-3]

    """
    __slots__ = ("_chains", "_type")

    def __init__(
        self,
        chains: abc.Iterable[CT],
        categories: abc.Iterable[str | abc.Iterable[str]] | None = None,
    ):
        """
        :param chains: An iterable over ``Chain*``-type objects.
        :param categories: An optional list of categories.
            If provided, they will be assigned to inputs' `meta` attributes.
        """
        if not isinstance(chains, list):
            chains = list(chains)
        if categories is not None:
            for c, cat in zip(chains, categories, strict=True):
                if isinstance(cat, str):
                    add_category(c, cat)
                else:
                    for _cat in cat:
                        add_category(c, _cat)
        self._chains: list[CT] = chains
        self._type: str | None = None
        self._check_match_and_set_type(self._infer_type(self._chains))

    @staticmethod
    def _infer_type(objs: abc.Sequence[T]) -> t.Optional[str]:
        """
        Infer a type of objs.
        Populate :attr:`_type` with the inferred type if it's supported.
        Otherwise, raise an error.

        If objs do not belong to a single type, raise an error.

        :param objs: Arbitrary sequence or arbitrary objects.
        :return: One of ('chain', 'seq', and 'str') for
            (:class:`Chain`, :class:`ChainSequence`, and
            :class:`ChainStructure`).
        """
        types = set(map(type, objs))
        if len(types) > 1:
            raise TypeError(f"ChainList elements must have single type; got {types}")
        if objs:
            match objs[0]:
                case Chain():
                    return "chain"
                case ChainSequence():
                    return "seq"
                case ChainStructure():
                    return "str"
                case _:
                    raise TypeError(f"Unsupported type {objs[0]}")
        else:
            return None

    @property
    def type(self) -> str:
        """
        A type of the contained elements.

        :return: One of ('chain', 'seq', and 'str') for
            (:class:`Chain`, :class:`ChainSequence`,
            and :class:`ChainStructure`). None if the chain list is empty.
        """
        return self._type

    @property
    def categories(self) -> abc.Set[str]:
        """
        :return: A set of categories inferred from `meta` of encompassed
            objects.
        """
        return set(chain.from_iterable(map(lambda c: c.categories, self)))

    def _check_match_and_set_type(self, x: str) -> t.NoReturn:
        if self._type is not None:
            if x != self._type:
                raise TypeError(
                    f"Supplied type doesn't match existing type {self._type}"
                )
        else:
            self._type = x

    def __len__(self) -> int:
        return len(self._chains)

    @t.overload
    def __getitem__(self, index: int) -> CT:
        ...

    @t.overload
    def __getitem__(self, index: slice | str) -> ChainList[CT]:
        ...

    def __getitem__(self, index: int | slice | str) -> CT | ChainList[CT]:
        match index:
            case int():
                return self._chains[index]
            case slice():
                return ChainList(self._chains[index])
            case str():
                if index in self.categories:
                    return self.filter_category(index)
                return self.filter(lambda x: index in x.id)
            case _:
                raise TypeError(f"Incorrect index type {type(index)}")

    def __getattr__(self, name: str):
        """
        See the example in pandas:

        https://github.com/pandas-dev/pandas/blob/
        61e0db25f5982063ba7bab062074d55d5e549586/pandas/core/generic.py#L5811
        """

        if name == "__setstate__":
            raise AttributeError(name)

        if name.startswith("__"):
            object.__getattribute__(self, name)

        if name in self.categories:
            return self.filter(lambda c: any(cat == name for cat in c.categories))

        raise AttributeError

    def __setitem__(self, index: int, value: CT) -> t.NoReturn:
        _type = self._infer_type([value])
        if len(self) == 1 and index == 0:
            self._type = _type
        else:
            self._check_match_and_set_type(_type)
        self._chains.__setitem__(index, value)

    def __delitem__(self, index: int) -> t.NoReturn:
        self._chains.__delitem__(index)
        if len(self) == 0:
            self._type = None

    def __contains__(self, item: str | CT) -> bool:
        match item:
            case str():
                return first_true(
                    self._chains, default=False, pred=lambda c: c.id == item
                )
            case Chain() | ChainStructure() | ChainSequence():
                return item in self._chains
            case _:
                return False

    def __add__(self, other: ChainList | abc.Iterable):
        match other:
            case ChainList():
                return ChainList(self._chains + other._chains)
            case abc.Iterable():
                return ChainList(self._chains + list(other))
            case _:
                raise TypeError(f"Unsupported type {type(other)}")

    def __repr__(self) -> str:
        return self._chains.__repr__()

    def __iter__(self) -> abc.Iterator[CT]:
        return iter(self._chains)

    def index(self, value: CT, start: int = 0, stop: int | None = None) -> int:
        stop = stop or len(self)
        return self._chains.index(value, start, stop)

    def insert(self, index: int, value: CT) -> t.NoReturn:
        self._check_match_and_set_type(self._infer_type([value]))
        self._chains.insert(index, value)

    def iter_children(self) -> abc.Generator[ChainList[CT]]:
        """
        Simultaneously iterate over topological levels of children.

        >>> s = ChainSequence.from_string('ABCDE', name='A')
        >>> child1 = s.spawn_child(1, 4)
        >>> child2 = child1.spawn_child(2, 3)
        >>> x = ChainSequence.from_string('XXXX', name='X')
        >>> child3 = x.spawn_child(1, 3)
        >>> cl = ChainList([s, x])
        >>> list(cl.iter_children())
        [[A|1-4<-(A|1-5), X|1-3<-(X|1-4)], [A|2-3<-(A|1-4<-(A|1-5))]]

        :return: An iterator over chain lists of children levels.
        """
        yield from map(
            lambda xs: ChainList(chain.from_iterable(xs)),
            zip_longest(*map(lambda c: c.iter_children(), self._chains), fillvalue=[]),
        )

    def get_level(self, n: int) -> ChainList[CT]:
        """
        Get a specific level of a hierarchical tree starting from this list::

            l0: this list
            l1: children of each child of each object in l0
            l2: children of each child of each object in l1
            ...

        :param n: The level index (0 indicates this list).
            Other levels are obtained via :meth:`iter_children`.
        :return: A chain list of object corresponding to a specific topological
            level of a child tree.
        """
        if n == 0:
            return self
        return nth(self.iter_children(), n - 1, default=ChainList([]))

    def collapse_children(self) -> ChainList[CT]:
        """
        Collapse all children of each object in this list into a single
        chain list.

        >>> s = ChainSequence.from_string('ABCDE', name='A')
        >>> child1 = s.spawn_child(1, 4)
        >>> child2 = child1.spawn_child(2, 3)
        >>> cl = ChainList([s]).collapse_children()
        >>> assert isinstance(cl, ChainList)
        >>> cl
        [A|1-4<-(A|1-5), A|2-3<-(A|1-4<-(A|1-5))]

        :return: A chain list of all children.

        """
        return ChainList(chain.from_iterable(self.iter_children()))

    def iter_sequences(self) -> abc.Generator[ChainSequence]:
        """
        :return: An iterator over :class:`ChainSequence`'s.
        """
        match self._type:
            case "chain" | "str":
                yield from (c.seq for c in self._chains)
            case _:
                yield from iter(self._chains)

    def iter_structures(self) -> abc.Generator[ChainStructure]:
        """
        :return: An generator over :class:`ChainStructure`'s.
        """
        match self._type:
            case "chain":
                yield from chain.from_iterable(c.structures for c in self._chains)
            case "str":
                yield from iter(self)
            case _:
                yield from iter([])

    def iter_structure_sequences(self) -> abc.Generator[ChainSequence]:
        """
        :return: Iterate over :attr:`ChainStructure.seq` attributes.
        """
        yield from (s.seq for s in self.iter_structures())

    @staticmethod
    def _to_segment(s: abc.Sequence[int, int] | Segment) -> Segment:
        match s:
            case (start, end):
                return Segment(start, end)
            case Segment():
                return s
            case _:
                raise TypeError(f"Unsupported type {type(s)}")

    @staticmethod
    def _get_seg_matcher(
        s: str,
    ) -> abc.Callable[[ChainSequence, Segment, t.Optional[str]], bool]:
        def matcher(
            seq: ChainSequence, seg: Segment, map_name: t.Optional[str] = None
        ) -> bool:
            if map_name is not None:
                # Get elements in the seq whose mapped sequence matches
                # seg boundaries
                start = seq.get_closest(map_name, seg.start)._asdict()[map_name]
                end = seq.get_closest(map_name, seg.end, reverse=True)._asdict()[
                    map_name
                ]
                # If not such elements -> no match
                if start is None or end is None:
                    return False
                # Create a new temporary segment using the mapped boundaries
                seq = Segment(start, end)
            match s:
                case "overlap":
                    return seq.overlaps(seg)
                case "bounded":
                    return seq.bounded_by(seg)
                case "bounding":
                    return seq.bounds(seg)
                case _:
                    raise ValueError(f"Invalid matching mode {s}")

        return matcher

    @staticmethod
    def _get_pos_matcher(
        ps: abc.Iterable[Ord],
    ) -> abc.Callable[[ChainSequence, t.Optional[str]], bool]:
        def matcher(seq: ChainSequence, map_name: t.Optional[str] = None) -> bool:
            obj = seq
            if map_name:
                obj = seq[map_name]
            return all(p in obj for p in ps)

        return matcher

    def _filter_seqs(
        self,
        seqs: abc.Iterable[ChainSequence],
        match_type: str,
        s: Segment | abc.Collection[Ord],
        map_name: t.Optional[str],
    ) -> abc.Iterator[bool]:
        match s:
            case Segment():
                match_fn = partial(
                    self._get_seg_matcher(match_type),
                    seg=self._to_segment(s),
                    map_name=map_name,
                )
            case abc.Collection():
                match_fn = partial(self._get_pos_matcher(s), map_name=map_name)
            case _:
                raise TypeError(f"Unsupported type to match {type(s)}")
        return map(match_fn, seqs)

    def _filter_str(
        self,
        structures: abc.Iterable[ChainStructure],
        match_type: str,
        s: abc.Sequence[int, int] | Segment,
        map_name: t.Optional[str],
    ) -> abc.Iterator[bool]:
        return self._filter_seqs(
            map(lambda x: x.seq, structures), match_type, s, map_name
        )

    def filter_pos(
        self,
        s: Segment | abc.Collection[Ord],
        *,
        match_type: str = "overlap",
        map_name: t.Optional[str] = None,
    ) -> ChainList[SS]:
        """
        Filter to objects encompassing certain consecutive position regions
        or arbitrary positions' collections.

        For :class:`Chain` and :class:`ChainStructure`, the filtering is over
        `seq` attributes.

        :param s: What to search for:

            1) ``s=Segment(start, end)`` to find all objects encompassing
            certain region.
            2) ``[pos1, posX, posN]`` to find all objects encompassing the
            specified positions.

        :param match_type: If `s` is `Segment`, this value determines the
            acceptable relationships between `s` and each
            :class:`ChainSequence`:

            1) "overlap" -- it's enough to overlap with `s`.
            2) "bounding" -- object is accepted if it bounds `s`.
            3) "bounded" -- object is accepted if it's bounded by `s`.

        :param map_name: Use this map within to map positions of `s`.
            For instance, to each for all elements encompassing region 1-5 of
            a canonical sequence, one would use

                .. code-block:: python

                chain_list.filter_pos(
                    s=Segment(1, 5), match_type="bounding",
                    map_name="map_canonical"
                )

        :return: A list of hits of the same type.
        """

        if self.type == "seq":
            objs, fn = iter(self), self._filter_seqs
        elif self.type == "chain":
            objs, fn = self.iter_sequences(), self._filter_seqs
        else:
            objs, fn = iter(self), self._filter_str

        objs1, objs2 = tee(objs)
        mask = fn(objs1, match_type, s, map_name)

        return ChainList(
            map(op.itemgetter(1), filter(lambda x: x[0], zip(mask, objs2)))
        )

    def filter(self, pred: abc.Callable[[CT], bool]) -> ChainList[CT]:
        """
        >>> cl = ChainList(
        ...     [ChainSequence.from_string('AAAX', name='A'),
        ...      ChainSequence.from_string('XXX', name='X')]
        ... )
        >>> cl.filter(lambda c: c.seq1[0] == 'A')
        [A|1-4]

        :param pred: Predicate callable for filtering.
        :return: A filtered chain list (new object).
        """
        return ChainList(filter(pred, self))

    def filter_category(self, name: str) -> ChainList:
        """
        :param name: Category name.
        :return: Filtered objects having this category within their
            ``meta["category"]``.
        """
        return self.filter(lambda c: any(cat == name for cat in c.categories))

    def apply(self, fn: abc.Callable[[CT, ...], CT], *args, **kwargs) -> ChainList[CT]:
        """
        Apply a function to each object and return a new chain list of results.

        :param fn: A callable to apply.
        :param args: Passed to a `fn`.
        :param kwargs: Passed to a `fn`.
        :return: A new chain list with application results.
        """
        return ChainList([fn(c, *args, **kwargs) for c in self])


@curry
def _read_obj(
    path: Path, obj_type: t.Type[CT], tolerate_failures: bool, **kwargs
) -> CT | None:
    try:
        return obj_type.read(path, **kwargs)
    except Exception as e:
        LOGGER.warning(f"Failed to initialize {obj_type} from {path}")
        LOGGER.exception(e)
        if not tolerate_failures:
            raise e
    return None


@curry
def _write_obj(obj: CT, path: Path, tolerate_failures: bool, **kwargs) -> Path | None:
    try:
        obj.write(path, **kwargs)
        return path
    except Exception as e:
        LOGGER.warning(f"Failed to write {obj} to {path}")
        LOGGER.exception(e)
        if not tolerate_failures:
            raise e
    return None


class ChainIO:
    """
    A class handling reading/writing collections of `Chain*` objects.
    """

    # TODO: implement context manager
    def __init__(
        self,
        num_proc: None | int = None,
        verbose: bool = False,
        tolerate_failures: bool = False,
        dump_names: DumpNames = DumpNames,
    ):
        """
        :param num_proc: The number of parallel processes. Using more processes
            is especially beneficial for :class:`ChainStructure`'s and
            :class:`Chain`'s with structures. Otherwise, the increasing this
            number may not reduce or actually worsen the time needed to
            read/write objects.
        :param verbose: Output logging and progress bar.
        :param tolerate_failures: Errors when reading/writing do not raise
            an exception.
        :param dump_names: File names container.
        """
        #: The number of parallel processes
        self.num_proc = num_proc
        #: Output logging and progress bar.
        self.verbose = verbose
        #: Errors when reading/writing do not raise an exception.
        self.tolerate_failures = tolerate_failures
        #: File names container.
        self.dump_names = dump_names

    def _read(
        self,
        obj_type: t.Type[CT],
        path: Path | abc.Iterable[Path],
        non_blocking: bool = False,
        **kwargs,
    ) -> t.Optional[CT] | abc.Iterator[t.Optional[CT]]:

        if isinstance(path, Path):
            dirs = get_dirs(path)
        else:
            dirs = {p.name: p for p in path if p.is_dir()}

        _read = _read_obj(
            obj_type=obj_type, tolerate_failures=self.tolerate_failures, **kwargs
        )

        if DumpNames.segments_dir in dirs or not dirs and isinstance(path, Path):
            yield _read(path)
            return

        dirs = dirs.values()

        if self.num_proc is None:

            if self.verbose:
                dirs = tqdm(dirs, desc=f"Reading {obj_type}")

            yield from map(_read, dirs)

        else:

            with ProcessPoolExecutor(self.num_proc) as executor:

                futures = as_completed([executor.submit(_read, d) for d in dirs])

                if non_blocking:
                    yield from futures

                if self.verbose:
                    futures = tqdm(futures, desc=f"Reading {obj_type}")

                for future in futures:
                    yield future.result()

    def write(
        self,
        objs: CT | abc.Iterable[CT],
        base: Path,
        non_blocking: bool = False,
        **kwargs,
    ) -> abc.Generator[Future] | abc.Generator[Path] | t.NoReturn:
        """
        :param objs: A single or multiple objects to write.
            Each must have a `write` method accepting a directory.
        :param base: A writable dir. If `objs` are many, dump into `id`
            directories.
        :param non_blocking: If :attr:`num_proc` is >= 1, return `Future`
            objects instead of waiting for the result.
        :param kwargs: Passed to the `write` method of each object.
        :return: Whatever `write` method returns.
        """
        if isinstance(objs, (ChainSequence, ChainStructure, Chain)):
            objs.write(base)
        else:
            _write = _write_obj(tolerate_failures=self.tolerate_failures, **kwargs)

            if self.num_proc is None:
                if self.verbose:
                    objs = tqdm(objs, desc="Writing objects")
                for obj in objs:
                    yield _write(obj, base / obj.id)
            else:
                with ProcessPoolExecutor(self.num_proc) as executor:

                    futures = as_completed(
                        [executor.submit(_write, obj, base / obj.id) for obj in objs]
                    )

                    if non_blocking:
                        yield from futures

                    if self.verbose:
                        futures = tqdm(futures, desc="Writing objects")

                    for future in futures:
                        yield future.result()

    def read_chain(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> Chain | abc.Iterator[Chain | None] | None:
        """
        Read :class:`Chain`'s from the provided path.

        If `path` contains signature files and directories
        (such as `sequence.tsv` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`Chain` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`Chain.read`
        :return: An single chain or iterator over chain objects read
            sequentially or in parallel.
        """
        return self._read(Chain, path, **kwargs)

    def read_chain_seq(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> t.Optional[ChainSequence] | abc.Iterator[t.Optional[ChainSequence]]:
        """
        Read :class:`ChainSequence`'s from the provided path.

        If `path` contains signature files and directories
        (such as `sequence.tsv` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`ChainSequence` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`ChainSequence.read`
        :return: An single chain sequence or iterator over
            :class:`ChainSequence` objects read sequentially or in parallel.
        """
        return self._read(ChainSequence, path, **kwargs)

    def read_chain_struc(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> t.Optional[ChainStructure] | abc.Iterator[t.Optional[ChainStructure]]:
        """
        Read :class:`ChainStructure`'s from the provided path.

        If `path` contains signature files and directories
        (such as `structure.cif` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`ChainStructure` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`ChainSequence.read`
        :return: An single chain sequence or iterator over
            :class:`ChainStructure` objects read sequentially or in parallel.
        """
        return self._read(ChainStructure, path, **kwargs)


class InitializerCallback(t.Protocol):
    """
    A protocol defining signature for a callback used with
    :class:`ChainInitializer`.
    """

    @t.overload
    def __call__(self, inp: CT) -> CT | None:
        ...

    @t.overload
    def __call__(self, inp: list[ChainStructure]) -> list[ChainStructure] | None:
        ...

    @t.overload
    def __call__(self, inp: None) -> None:
        ...

    def __call__(
        self, inp: CT | list[ChainStructure] | None
    ) -> CT | list[ChainStructure] | None:
        ...


def _read_path(x, tolerate_failures, supported_seq_ext, supported_str_ext):
    if x.suffix in supported_seq_ext:
        return ChainSequence.from_file(x)
    if x.suffix in supported_str_ext:
        return [
            ChainStructure.from_structure(c)
            for c in GenericStructure.read(x).split_chains()
        ]
    if tolerate_failures:
        return None
    raise InitError(f"Suffix {x.suffix} of the path {x} is not supported")


def _init(x, tolerate_failures, supported_seq_ext, supported_str_ext, callbacks):
    match x:
        case ChainSequence() | ChainStructure():
            res = x
        case [str(), str()]:
            res = ChainSequence.from_string(x[1], name=x[0])
        case [Path(), xs]:
            structures = _read_path(
                x[0], tolerate_failures, supported_seq_ext, supported_str_ext
            )
            structures = [s for s in structures if s.pdb.chain in xs]
            res = structures or None
        case GenericStructure():
            res = ChainStructure.from_structure(x)
        case Path():
            res = _read_path(x, tolerate_failures, supported_seq_ext, supported_str_ext)
        case _:
            res = None
            if not tolerate_failures:
                raise InitError(f"Unsupported input type {type(x)}")
    if callbacks:
        for c in callbacks:
            res = c(res)
    return res


def _map_numbering(seq1, seq2):
    return seq1.map_numbering(seq2, save=False)


def map_numbering_12many(
    obj_to_map: str | tuple[str, str] | ChainSequence | Alignment,
    seqs: abc.Iterable[ChainSequence],
    num_proc: t.Optional[int] = None,
) -> abc.Iterator[list[int | None]]:
    """
    Map numbering of a single sequence to many other sequences.

    **This function does not save mapped numberings.**

    .. seealso::
        :meth:`ChainSequence.map_numbering`.

    :param obj_to_map: Object whose numbering should be mapped to `seqs`.
    :param seqs: Chain sequences to map the numbering to.
    :param num_proc: A number of parallel processes to use.
        If ``None``, run sequentially.
    :return: An iterator over the mapped numberings.
    """
    if num_proc:
        with ProcessPoolExecutor(num_proc) as executor:
            yield from executor.map(_map_numbering, seqs, repeat(obj_to_map))
    else:
        yield from (x.map_numbering(obj_to_map, save=False) for x in seqs)


def map_numbering_many2many(
    objs_to_map: abc.Sequence[str | tuple[str, str] | ChainSequence | Alignment],
    seq_groups: abc.Sequence[abc.Sequence[ChainSequence]],
    num_proc: t.Optional[int] = None,
    verbose: bool = False,
) -> abc.Iterator[list[list[int | None]]]:
    """
    Map numbering of each object `o` in `objs_to_map` to each sequence
    in each group of the `seq_groups` ::

        o1 -> s1_1 s1_1 s1_3 ...
        o2 -> s2_1 s2_1 s2_3 ...
                  ...

    **This function does not save mapped numberings.**

    For a single object-group pair, it's the same as
    :func:`map_numbering_12many`. The benefit comes from parallelization
    of this functionality.

    .. seealso::
        :meth:`ChainSequence.map_numbering`.
        :func:`map_numbering_12many`

    :param objs_to_map: An iterable over objects whose numbering to map.
    :param seq_groups: Group of objects to map numbering to.
    :param num_proc: A number of processes to use. If ``None``,
        run sequentially.
    :param verbose: Output a progress bar.
    :return: An iterator over lists of lists with numeric mappings

    ::

         [[s1_1 map, s1_2 map, ...]
          [s2_1 map, s2_2 map, ...]
                    ...
          ]

    """
    if len(objs_to_map) != len(seq_groups):
        raise LengthMismatch(
            f"The number of objects to map {len(objs_to_map)} != "
            f"the number of sequence groups {len(seq_groups)}"
        )
    staged = chain.from_iterable(
        ((obj, s) for s in g) for obj, g in zip(objs_to_map, seq_groups)
    )
    group_sizes = map(len, seq_groups)
    if num_proc:
        objs, seqs = unzip(staged)
        with ProcessPoolExecutor(num_proc) as executor:
            results = executor.map(_map_numbering, seqs, objs, chunksize=1)
            if verbose:
                results = tqdm(results, desc="Mapping numberings")
            yield from split_into(results, group_sizes)
    else:
        results = (s.map_numbering(o, save=False) for o, s in staged)
        if verbose:
            results = tqdm(results, desc="Mapping numberings")
        yield from split_into(results, group_sizes)


class ChainInitializer:
    """
    In contrast to :class:`ChainIO`, this object initializes new
    :class:`Chain`, :class:`ChainStructure`, or :class:`Chain` objects from
    various input types.

    To initialize :class:`Chain` objects, use :meth:`from_mapping`.

    To initialize :class:`ChainSequence` or :class:`ChainStructure` objects,
    use :meth:`from_iterable`.

    """

    def __init__(
        self,
        num_proc: int | None = None,
        tolerate_failures: bool = False,
        verbose: bool = False,
    ):
        """

        :param num_proc: The number of processes to use.
        :param tolerate_failures: Don't stop the execution if some object fails
            to initialize.
        :param verbose: Output progress bars.
        """
        self.num_proc = num_proc
        self.tolerate_failures = tolerate_failures
        self.verbose = verbose

    @property
    def supported_seq_ext(self) -> list[str]:
        """
        :return: Supported sequence file extensions.
        """
        return [".fasta"]

    @property
    def supported_str_ext(self) -> list[str]:
        """
        :return: Supported structure file extensions.
        """
        return [".cif", ".pdb", ".pdbx", ".mmtf", ".npz"]

    def from_iterable(
        self,
        it: abc.Iterable[
            ChainSequence
            | ChainStructure
            | Path
            | tuple[Path, abc.Sequence[str]]
            | tuple[str, str]
            | GenericStructure
        ],
        callbacks: list[InitializerCallback] | None = None,
    ) -> abc.Generator[ChainSequence | ChainStructure]:
        """
        Initialize :class:`ChainSequence`s or/and :class:`ChainStructure`'s
        from (possibly heterogeneous) iterable.

        :param it:
            Supported elements are:
                1) Initialized objects (passed without any actions).
                2) Path to a sequence or a structure file.
                3) (Path to a structure file, list of target chains).
                4) A pair (header, seq) to initialize a :class:`ChainSequence`.
                5) A :class:`GenericStructure` with a single chain.

        :param callbacks: A sequence of callables accepting and returning an
            initialized object.
        :return: A generator yielding initialized chain sequences and
            structures parsed from the inputs.
        """
        if self.num_proc is not None:
            with ProcessPoolExecutor(self.num_proc) as executor:
                futures = [
                    (
                        x,
                        executor.submit(
                            _init,
                            x,
                            self.tolerate_failures,
                            self.supported_seq_ext,
                            self.supported_str_ext,
                            callbacks,
                        ),
                    )
                    for x in it
                ]
                if self.verbose:
                    futures = tqdm(futures, desc="Initializing objects in parallel")
                for x, future in futures:
                    try:
                        yield future.result()
                    except Exception as e:
                        LOGGER.warning(f"Input {x} failed with an error {e}")
                        # LOGGER.exception(e)
                        if not self.tolerate_failures:
                            raise e
                        yield None
        else:
            if self.verbose:
                it = tqdm(it, desc="Initializing objects sequentially")
            yield from (
                _init(
                    x,
                    self.tolerate_failures,
                    self.supported_seq_ext,
                    self.supported_str_ext,
                    callbacks,
                )
                for x in it
            )

    def from_mapping(
        self,
        m: abc.Mapping[
            ChainSequence | tuple[str, str] | Path,
            abc.Sequence[
                ChainStructure
                | GenericStructure
                | bst.AtomArray
                | Path
                | tuple[Path, abc.Sequence[str]]
            ],
        ],
        key_callbacks: t.Optional[list[InitializerCallback]] = None,
        val_callbacks: t.Optional[list[InitializerCallback]] = None,
        num_proc_map_numbering: int | None = None,
        **kwargs,
    ) -> list[Chain]:
        """
        Initialize :class:`Chain`'s from mapping between sequences and
        structures.

        It will first initialize objects to which the elements of `m`
        refer (see below) and then create maps between each sequence and
        associated structures, saving these into structure
        :attr:`ChainStructure.seq`'s.

        :param m:
            A mapping of the form ``{seq => [structures]}``, where `seq`
            is one of:

                1) Initialized :class:`ChainSequence`.
                2) A pair (header, seq).
                3) A path to a **fasta** file containing a single sequence.

            While each structure is one of:

                1) Initialized :class:`ChainStructure`.
                2) :class:`GenericStructure` with a single chain.
                3) :class:`biotite.AtomArray` corresponding to a single chain.
                4) A path to a structure file.
                5) (A path to a structure file, list of target chains).

            In the latter two cases, the chains will be expanded
            and associated with the same sequence.

        :param key_callbacks: A sequence of callables accepting and returning
            a :class:`ChainSequence`.
        :param val_callbacks: A sequence of callables accepting and returning
            a :class:`ChainStructure`.
        :param num_proc_map_numbering: A number of processes to use for mapping
            between numbering of sequences and structures. Generally, this
            should be as high as possible for faster processing. In contrast
            to the other operations here, this one seems more CPU-bound and
            less resource hungry (although, keep in mind the size of the
            canonical sequence: if it's too high, the RAM usage will likely
            explode). If ``None``, will default to :attr:`num_proc`.
        :param kwargs: Passed to :meth:`Chain.add_structure`.
        :return: A list of initialized chains.
        """
        # Process keys and values
        keys = self.from_iterable(m, callbacks=key_callbacks)  # ChainSequences
        values_flattened = self.from_iterable(  # ChainStructures
            chain.from_iterable(m.values()), callbacks=val_callbacks
        )
        values = split_into(values_flattened, map(len, m.values()))

        m_new = valmap(
            lambda vs: collapse(filter(bool, vs)),
            keymap(
                Chain,  # create `Chain` objects from `ChainSequence`s
                keyfilter(bool, dict(zip(keys, values))),  # Filter possible failures
            ),
        )

        num_proc = num_proc_map_numbering or self.num_proc

        if num_proc is None or num_proc == 1:
            for c, ss in m_new.items():
                for s in ss:
                    c.add_structure(s, **kwargs)
        else:
            map_name = kwargs.get("map_name") or SeqNames.map_canonical

            # explicitly unpack value iterable into lists
            m_new = valmap(list, m_new)

            # create numbering groups -- lists of lists with numberings
            # for each structure in values
            numbering_groups = map_numbering_many2many(
                [x.seq for x in m_new],
                [[x.seq for x in val] for val in m_new.values()],
                num_proc=num_proc,
                verbose=self.verbose,
            )
            for (c, ss), num_group in zip(m_new.items(), numbering_groups, strict=True):
                if len(num_group) != len(ss):
                    raise LengthMismatch(
                        f"The number of mapped numberings {len(num_group)} must match "
                        f"the number of structures {len(ss)}."
                    )
                for s, n in zip(ss, num_group):
                    try:
                        s.seq.add_seq(map_name, n)
                        c.add_structure(s, map_to_seq=False, **kwargs)
                    except Exception as e:
                        LOGGER.warning(
                            f"Failed to add structure {s} to chain {c} due to {e}"
                        )
                        LOGGER.exception(e)
                        if not self.tolerate_failures:
                            raise e

        return list(m_new)


if __name__ == "__main__":
    pass
