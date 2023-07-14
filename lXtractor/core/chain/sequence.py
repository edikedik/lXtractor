from __future__ import annotations

import typing as t
import warnings
from collections import abc
from copy import copy
from io import TextIOBase
from itertools import filterfalse, starmap, repeat, chain
from pathlib import Path

import numpy as np
import pandas as pd
from more_itertools import first_true, always_reversible, split_into
from toolz import valmap, valfilter, keyfilter
from typing_extensions import Self

import lXtractor.core.segment as lxs
from lXtractor.core.alignment import Alignment
from lXtractor.core.base import (
    AminoAcidDict,
    AlignMethod,
    Ord,
    NamedTupleT,
    SeqReader,
    ApplyT,
    FilterT,
)
from lXtractor.core.chain.base import topo_iter
from lXtractor.core.chain.list import _wrap_children, add_category, ChainList
from lXtractor.core.config import (
    SeqNames,
    MetaNames,
    _SeqNames,
    _MetaNames,
    DumpNames,
    _DumpNames,
    UNK_NAME,
    ColNames,
)
from lXtractor.core.exceptions import (
    MissingData,
    InitError,
    AmbiguousMapping,
    LengthMismatch,
)
from lXtractor.util import biotite_align, apply
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.misc import is_empty
from lXtractor.util.seq import mafft_align, map_pairs_numbering, read_fasta

if t.TYPE_CHECKING:
    from lXtractor.core.chain import Chain, ChainStructure

# TODO: add "reset_numbering()" method for a segment
# It "reenumerates" the segment from the new start (1 by default)
# and may keep an existing numbering


__all__ = ("ChainSequence", "map_numbering_12many", "map_numbering_many2many")


class ChainSequence(lxs.Segment):
    """
    A class representing polymeric sequence of a single entity (chain).

    The sequences are stored internally as a dictionary `{seq_name => _seq}`
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

    @property
    def fields(self) -> tuple[str, ...]:
        """
        :return: Names of the currently stored sequences.
        """
        return tuple(self._seqs.keys())

    @classmethod
    def field_names(cls) -> _SeqNames:
        """
        :return: The default sequence's names.
        """
        return SeqNames

    @classmethod
    def meta_names(cls) -> _MetaNames:
        """
        :return: defaults names of the :attr:`meta` fields.
        """
        return MetaNames

    def _get_seq(self, name: str) -> abc.Sequence[str]:
        try:
            return self[name]
        except KeyError:
            raise MissingData(f"Missing sequence {name}")

    @property
    def numbering(self) -> abc.Sequence[int]:
        """
        :return: the primary sequence's (:meth:`seq1`) numbering.
        """
        if SeqNames.enum in self:
            return self[SeqNames.enum]
        return list(range(self.start, self.end + 1))

    @property
    def seq(self) -> Self:
        """
        This property exists for functionality relying on the `.seq` attribute.

        :return: This object.
        """
        return self

    @property
    def seq1(self) -> str:
        """
        :return: the primary sequence.
        """
        s = self._get_seq(SeqNames.seq1)
        return s if isinstance(s, str) else "".join(s)

    @property
    def seq3(self) -> abc.Sequence[str]:
        # TODO: remove and defer to subclasses representing concrete seqs.
        """
        :return: the three-letter codes of a primary sequence.
        """
        if SeqNames.seq3 not in self:
            try:
                seq1 = self[SeqNames.seq1]
            except KeyError as e:
                raise MissingData(
                    "Attempted to construct seq3 from seq1 but the latter is missing."
                ) from e
            mapping = AminoAcidDict()
            return [mapping[x] for x in seq1]
        return self[SeqNames.seq3]

    @property
    def categories(self) -> list[str]:
        """
        :return: A list of categories associated with this object.

        Categories are kept under "category" field in :attr:`meta`
        as a ","-separated list of strings. For instance, "domain,family_x".
        """
        cat: str = self.meta.get(MetaNames.category, "")
        return cat.split(",") if cat else []

    def _setup_and_validate(self) -> None:
        super()._setup_and_validate()

        if SeqNames.seq1 not in self:
            warnings.warn(f"Missing {SeqNames.seq1}")
        else:
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
        self.children: ChainList[ChainSequence] = _wrap_children(self.children)

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

        :param other: another chain _seq.
        :param align_method: a method to use for alignment.
        :param save: save the numbering as a sequence.
        :param name: a name to use if `save` is ``True``.
        :param kwargs: passed to `func:map_pairs_numbering`.
        :return: a list of integers with ``None`` indicating gaps.
        """

        def get_seq1(s: abc.Sequence[str] | str) -> str:
            if isinstance(s, str):
                return s
            return "".join(s)

        if isinstance(other, str):
            name = name or UNK_NAME
            other = ChainSequence.from_string(other)
        elif isinstance(other, tuple):
            name = other[0]
            other = ChainSequence.from_string(other[1], name=name)

        mapping: abc.Iterable[tuple]

        seq1 = get_seq1(self.seq1)

        if isinstance(other, ChainSequence):
            mapping = map_pairs_numbering(
                seq1,
                self.numbering,
                get_seq1(other.seq1),
                other.numbering,
                align=True,
                align_method=align_method,
                **kwargs,
            )
            if not name:
                name = f"map_{other.name}"
        elif isinstance(other, Alignment):
            self_name = self.name or UNK_NAME
            aligned_other = other.align((self_name, seq1))[self_name]
            aligned_other_num = [
                i for (i, c) in enumerate(aligned_other, start=1) if c != "-"
            ]
            mapping = map_pairs_numbering(
                seq1,
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

        mapped_numbering = [x[1] for x in mapping if x[0] is not None]
        if save:
            self[name] = mapped_numbering

        return mapped_numbering

    def map_boundaries(
            self, start: Ord, end: Ord, map_name: str, closest: bool = False
    ) -> tuple[NamedTupleT, NamedTupleT]:
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

            this.relate(
                other, map_name=aln_map,
                link_name=map_canonical, link_name_points_to="i"
            )

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
        if link_name == "i":
            other_seq = range(other.start, other.end + 1)
        else:
            other_seq = other[link_name]
        mapped = list(
            map(
                lambda x: x if x is None else x._asdict()[map_name],
                (mapping.get(x) for x in other_seq),
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

    def match(
            self,
            map_name1: str,
            map_name2: str,
            as_fraction: bool = True,
            save: bool = True,
    ) -> float:
        """
        :param map_name1: Mapping name 1.
        :param map_name2: Mapping name 2.
        :param as_fraction: Divide by the total length.
        :param save: Save to meta as 'Match_{map_name1}_{map_name2}'.
        :return: The total number or a fraction of matching characters between maps.
        """
        res: float = sum(1 for x, y in zip(self[map_name1], self[map_name2]) if x == y)
        div = len(self) if as_fraction else 1
        res = res / div
        if save:
            self.meta[f"Match_{map_name1}_{map_name2}"] = res
        return res

    def get_map(
            self, key: str, to: str | None = None, rm_empty: bool = False
    ) -> dict[t.Hashable, NamedTupleT]:
        """
        Obtain the mapping of the form "key->item(seq_name=*,...)".

        >>> s = ChainSequence.from_string('ABC', name='CS')
        >>> s.get_map('i')
        {1: Item(i=1, seq1='A'), 2: Item(i=2, seq1='B'), 3: Item(i=3, seq1='C')}
        >>> s.get_map('seq1')
        {'A': Item(i=1, seq1='A'), 'B': Item(i=2, seq1='B'), 'C': Item(i=3, seq1='C')}
        >>> s.add_seq('S', [1, 2, np.nan])
        >>> s.get_map('seq1', 'S', rm_empty=True)
        {'A': 1, 'B': 2}

        :param key: A _seq name to map from.
        :param to: A _seq name to map to.
        :param rm_empty: Remove empty keys and values. A numeric value is empty
            if it is of type NaN. A string value is empty if it is an empty
            string (``""``).
        :return: `dict` mapping key values to items.
        """
        keys = (x.i for x in self) if key == "i" else self[key]
        d = dict(zip(keys, iter(self)))
        if to is not None:
            d = valmap(lambda x: x._asdict()[to], d)
        if rm_empty:
            d = keyfilter(lambda x: not is_empty(x), d)
            d = valfilter(lambda x: not is_empty(x), d)
        return d

    def get_item(self, key: str, value: t.Any) -> NamedTupleT:
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
    ) -> t.Optional[NamedTupleT]:
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

        items = iter(self.get_map(key).items())
        if reverse:
            items = always_reversible(items)
        result = first_true(items, default=None, pred=pred)
        if result:
            return result[1]
        return None

    def as_df(self) -> pd.DataFrame:
        """
        :return: The pandas DataFrame representation of the sequence where
            each column correspond to a sequence or map.
        """
        return pd.DataFrame(list(iter(self)))

    # @lru_cache
    def as_np(self) -> np.ndarray:
        """
        :return: The numpy representation of a sequence as matrix.
            This is a shortcut to :meth:`as_df` and getting `df.values`.
        """
        return self.as_df().values

    def as_chain(
            self,
            transfer_children: bool = True,
            structures: abc.Sequence[ChainStructure] | None = None,
            **kwargs,
    ) -> Chain:
        """
        Convert this chain sequence to chain.

        .. note::
            Pass ``add_to_children=True`` to transfer `structure` to each child
            if ``transfer_children=True``.

        :param transfer_children: Transfer existing children.
        :param structures: Add structures to the created chain.
        :param kwargs: Passed to :meth:`Chain.add_structure
            <lXtractor.core.chain.chain.Chain.add_structure>`
        :return:
        """
        from lXtractor.core.chain import Chain

        c = Chain.from_seq(self)
        if transfer_children:
            c.children = ChainList(
                [x.as_chain(transfer_children=True) for x in self.children]
            )
        if structures:
            for s in structures:
                c.add_structure(s, **kwargs)
        return c

    # @lru_cache
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
            start, end = map(  # type: ignore  # incompatible assignment wrong
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
            # self.children: ChainList[ChainSequence]
            self.children.append(child)

        return child

    def iter_children(self) -> abc.Generator[ChainList[ChainSequence], None, None]:
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
        if self.children is not None:
            # self.children: ChainList[ChainSequence]
            yield from map(ChainList, topo_iter(self, lambda x: x.children))
        else:
            yield from iter(ChainList([]))

    def apply_children(self, fn: ApplyT[ChainSequence], inplace: bool = False) -> Self:
        """
        Apply some function to children.

        :param fn: A callable accepting and returning the chain sequence type
            instance.
        :param inplace: Apply to children in place. Otherwise, return a copy
            with only children transformed.
        :return: A chain sequence with transformed children.
        """
        children = self.children.apply(fn)
        if inplace:
            self.children = children
            return self
        return self.__class__(
            self.start,
            self.end,
            self.name,
            seqs=self._seqs,
            meta=self.meta,
            children=children,
            parent=self.parent,
            variables=self.variables,
        )

    def filter_children(
            self, pred: FilterT[ChainSequence], inplace: bool = False
    ) -> Self:
        """
        Filter children using some predicate.

        :param pred: Some callable accepting chain sequence and returning bool.
        :param inplace: Filter :attr:`children` in place. Otherwise, return
            a copy with only children transformed.
        :return: A chain sequence with filtered children.
        """
        children = self.children.filter(pred)
        if inplace:
            self.children = children
            return self
        return self.__class__(
            self.start,
            self.end,
            self.name,
            seqs=self._seqs,
            meta=self.meta,
            children=children,
            parent=self.parent,
            variables=self.variables,
        )

    def apply_to_map(
            self,
            map_name: str,
            fn: ApplyT[abc.Sequence],
            inplace: bool = False,
            preserve_children: bool = False,
            apply_to_children: bool = False,
    ) -> Self:
        """
        Apply some function to map/sequence in this chain sequence.

        :param map_name: Name of the internal sequence/map.
        :param fn: A function accepting and returning a sequence of the same
            length.
        :param inplace: Apply the operation to this object. Otherwise, create
            a copy with the transformed sequence.
        :param preserve_children: Preserve :attr:`children` of this instance in
            the transformed object. Passing ``True`` makes sense if the target
            sequence is mutable: the children's will be transformed naturally.
            In the target sequence is immutable, consider passing ``True`` with
            ``apply_to_children=True``.
        :param apply_to_children: Recursively apply the same `fn` to a child
            tree starting from this instance. If passed, sets
            ``preserve_children=True``: otherwise, one is at risk of removing
            all :attr:`children` in the child tree of the returned instance.
        :return:
        """

        def _apply_to_children() -> ChainList[ChainSequence]:
            return ChainList(
                [
                    c.apply_to_map(
                        map_name,
                        fn,
                        inplace=True,
                        preserve_children=True,
                        apply_to_children=True,
                    )
                    for c in self.children
                ]
            )

        s = self[map_name]
        size = len(s)
        s_x = fn(s)
        if len(s_x) != size:
            raise ValueError(f"Seq length changed from {size} to {len(s_x)}")

        children = self.children
        if apply_to_children:
            children = _apply_to_children()
        else:
            if not preserve_children:
                children = ChainList([])

        if inplace:
            self._seqs[map_name] = s_x
            self.children = children
            return self

        seqs = copy(self._seqs)
        seqs[map_name] = s_x
        return self.__class__(
            self.start,
            self.end,
            self.name,
            seqs=seqs,
            parent=self.parent,
            children=children,
            variables=self.variables,
        )

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
            (as used during initialization via `_seq` attribute).
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
            (as used during initialization via `_seq` attribute).
        :return: Initialized chain sequence.
        """
        if len(s) == 0:
            start = start or 0
            end = end or 0
        else:
            start = start or 1
            end = end or start + len(s) - 1

        return cls(start, end, name, meta=meta, seqs={SeqNames.seq1: s, **kwargs})

    @classmethod
    def make_empty(cls) -> ChainSequence:
        """
        :return: An empty chain sequence.
        """
        return cls.from_string("")

    @classmethod
    def from_df(
            cls,
            df: Path | pd.DataFrame,
            name: t.Optional[str] = None,
            meta: dict[str, t.Any] | None = None,
    ) -> Self:
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
        seqs: dict[str, t.Sequence[object]] = {
            col: list(df[col]) for col in map(str, df.columns) if col != "i"
        }
        return cls(start, end, name, meta=meta, seqs=seqs)

    @classmethod
    def read(
            cls,
            base_dir: Path,
            *,
            dump_names: _DumpNames = DumpNames,
            search_children: bool = False,
    ) -> Self:
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
            from lXtractor.variables import Variables

            seq.variables = Variables.read(files[dump_names.variables]).sequence

        if search_children and dump_names.segments_dir in dirs:
            for path in (base_dir / dump_names.segments_dir).iterdir():
                child = ChainSequence.read(
                    path, dump_names=dump_names, search_children=True
                )
                child.parent = seq
                seq.children.append(child)

        return seq

    def write_seq(self, path: Path, fields: list[str] | None = None, sep: str = "\t"):
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

    def write_meta(self, path: Path, sep="\t"):
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
            dump_names: _DumpNames = DumpNames,
            write_children: bool = False,
    ):
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
                child_dir = (
                        base_dir / dump_names.segments_dir / (child.name or UNK_NAME)
                )
                child.write(
                    child_dir, dump_names=dump_names, write_children=write_children
                )

    def summary(self, meta: bool = True, children: bool = False) -> pd.DataFrame:
        parent_id = self.parent.id if self.parent is not None else np.NaN
        vs = [parent_id, self.id, self.start, self.end]
        cols = [ColNames.parent_id, ColNames.id, ColNames.start, ColNames.end]

        if meta:
            vs += list(self.meta.values())
            cols += list(self.meta)

        rows = [pd.Series(vs, index=cols)]

        if self.children and children:
            rows += [c.summary(meta=meta, children=children) for c in self.children]

        return pd.DataFrame(rows)


def _map_numbering(
        pair: tuple[ChainSequence, str | tuple[str, str] | ChainSequence | Alignment]
) -> list[None | int]:
    seq, obj = pair
    return seq.map_numbering(obj, save=False, align_method=biotite_align)


def map_numbering_12many(
        obj_to_map: str | tuple[str, str] | ChainSequence | Alignment,
        seqs: abc.Iterable[ChainSequence],
        num_proc: int = 1,
        verbose: bool = False,
        **kwargs
) -> abc.Iterator[list[int | None]]:
    """
    Map numbering of a single sequence to many other sequences.

    **This function does not save mapped numberings.**

    .. seealso::
        :meth:`ChainSequence.map_numbering`.

    :param obj_to_map: Object whose numbering should be mapped to `seqs`.
    :param seqs: Chain sequences to map the numbering to.
    :param num_proc: A number of parallel processes to use.
    :param verbose: Output progress bar.
    :param kwargs: Passed to :func:`lXtractor.util.misc.apply`.
    :return: An iterator over the mapped numberings.
    """
    staged = zip(seqs, repeat(obj_to_map))
    total = len(seqs) if isinstance(seqs, abc.Sized) else None
    yield from apply(
        _map_numbering, staged, verbose, "Mapping numberings", num_proc, total, **kwargs
    )


def map_numbering_many2many(
        objs_to_map: abc.Sequence[str | tuple[str, str] | ChainSequence | Alignment],
        seq_groups: abc.Sequence[abc.Sequence[ChainSequence]],
        num_proc: int = 1,
        verbose: bool = False,
        **kwargs,
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
    :param num_proc: A number of processes to use.
    :param verbose: Output a progress bar.
    :param kwargs: Passed to :func:`lXtractor.util.misc.apply`.
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
        ((s, obj) for s in g) for obj, g in zip(objs_to_map, seq_groups)
    )
    group_sizes = map(len, seq_groups)
    results = apply(
        _map_numbering, staged, verbose, "Mapping numberings", num_proc, **kwargs
    )
    yield from split_into(results, group_sizes)


if __name__ == "__main__":
    raise RuntimeError
