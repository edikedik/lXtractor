from __future__ import annotations

import logging
import operator as op
import typing as t
from collections import namedtuple, abc
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from dataclasses import dataclass
from functools import lru_cache, partial
from io import TextIOBase
from itertools import starmap, chain, zip_longest, tee, filterfalse, repeat
from pathlib import Path

import biotite.structure as bst
import numpy as np
import pandas as pd
from more_itertools import unzip, first_true, split_into, zip_equal, collapse, nth
from toolz import curry, keyfilter, keymap, valmap
from tqdm.auto import tqdm

from lXtractor.core.alignment import Alignment
from lXtractor.core.base import AminoAcidDict, AbstractChain, Ord, AlignMethod, SeqReader
from lXtractor.core.config import Sep, DumpNames, SeqNames, MetaNames
from lXtractor.core.exceptions import MissingData, AmbiguousMapping, InitError, NoOverlap, LengthMismatch
from lXtractor.core.segment import Segment
from lXtractor.core.structure import GenericStructure, validate_chain, PDB_Chain
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.seq import mafft_align, map_pairs_numbering, read_fasta
from lXtractor.util.structure import filter_selection
from lXtractor.variables.base import Variables

T = t.TypeVar('T')

LOGGER = logging.getLogger(__name__)


def topo_iter(
        start_obj: T, iterator: abc.Callable[[T], abc.Iterator[T]]
) -> abc.Generator[list[T]]:
    def get_level(cs: abc.Iterable[T]) -> abc.Iterator[T]:
        return chain.from_iterable(map(iterator, cs))

    curr_level = list(iterator(start_obj))

    while True:
        yield curr_level
        curr_level = list(get_level(curr_level))
        if not curr_level:
            return


class ChainSequence(Segment):
    """
    A class representing polymeric sequence of a single entity (chain).

    The sequences are stored internally as a dictionary `{seq_name => seq}`
    and must all have the same length.
    A single gap-less primary sequence (:meth:`seq1`) is mandatory during the initialization.
    We refer to the sequences other than :meth:`seq1` as "maps."
    To view the standard sequence names supported by :class:`ChainSequence`,
    use the :meth:`flied_names` property.

    The sequence can be a part of a larger one. The child-parent relationships
    are indicated via :attr:`parent` and attr:`children`, where the latter
    entails any sub-sequence. A preferable way to create subsequences is
    the :meth:`spawn_child` method.

    """
    __slots__ = ()

    def iter_children(self) -> abc.Generator[list[ChainSequence]]:
        """
        Iterate over a child tree in topological order.

        :return: a generator over child tree levels, starting from the :attr:`children`
            and expandingsuch attributes over :class:`ChainSequence` instances
            within this attribute.
        """
        return topo_iter(self, lambda x: iter(x.children.values()))

    @property
    def fields(self) -> tuple[str, ...]:
        """
        :return: Names of the currently stored sequences.
        """
        return tuple(self._seqs.keys())

    @classmethod
    def field_names(cls) -> dataclass:
        """
        :return: default sequence names.
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
        return list(range(1, len(self) + 1))

    @property
    def seq1(self) -> str:
        """
        :return: the primary sequence.
        """
        return self[SeqNames.seq1]

    @property
    def seq3(self) -> t.Sequence[str]:
        """
        :return: the three-letter codes of a primary sequence.
        """
        if SeqNames.seq3 not in self:
            d = AminoAcidDict()
            return [d[x] for x in self.seq1]
        return self[SeqNames.seq3]

    def _setup_and_validate(self):
        super()._setup_and_validate()

        if SeqNames.seq1 not in self:
            raise MissingData(f'Requires {SeqNames.seq1} in `seqs`')
        # if SeqNames.seq3 not in self:
        #     d = AminoAcidDict()
        #     self[SeqNames.seq3] = [d[c] for c in self[SeqNames.seq1]]
        # if SeqNames.enum not in self._seqs:
        #     self[SeqNames.enum] = list(range(self.start, self.end + 1))

        if not isinstance(self.seq1, str):
            try:
                self[SeqNames.seq1] = ''.join(self.seq1)
            except Exception as e:
                raise InitError(f'Failed to convert {SeqNames.seq1} '
                                f'from type {type(self.seq1)} to str '
                                f'due to: {e}')

        self.meta[MetaNames.id] = self.id
        self.meta[MetaNames.name] = self.name

    def map_numbering(
            self, other: str | tuple[str, str] | ChainSequence | Alignment,
            align_method: AlignMethod = mafft_align,
            save: bool = True, name: t.Optional[str] = None, **kwargs
    ) -> list[None | int]:
        """
        Map the :meth:`numbering: of another sequence onto this one.
        For this, align primary sequences and relate their numbering.

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
                    self.seq1, self.numbering, other.seq1, other.numbering,
                    align=True, align_method=align_method, **kwargs
                )
            )
            if not name:
                name = f'map_{other.name}'
        elif isinstance(other, Alignment):
            aligned_aln = other.align((name, self.seq1))
            aligned_other = aligned_aln[name]
            aligned_other_num = [
                i for (i, c) in enumerate(aligned_other, start=1) if c != '-']
            mapping = map_pairs_numbering(
                self.seq1, self.numbering, aligned_other, aligned_other_num,
                align=True, align_method=align_method, **kwargs
            )
            if not name:
                name = SeqNames.map_aln
        else:
            raise TypeError(f'Unsupported type {type(other)}')

        mapped_numbering = [x[1] for x in mapping]
        if save:
            self[name] = mapped_numbering

        return mapped_numbering

    def map_boundaries(
            self, start: Ord, end: Ord, map_name: str, closest: bool = False
    ) -> tuple[namedtuple, namedtuple]:
        """
        Map the provided boundaries onto sequence.

        A convenient interface for common task where one wants to find sequence elements
        corresponding to arbitrary boundaries.

        :param start: some orderable object.
        :param end: some orderable object.
        :param map_name: use this sequence to search for boundaries. It is assumed that
            `map_name in self is True`.
        :param closest: If true, instead of exact mapping, search for the closest elements.
        :return: an tuple with two items corresponding to mapped `start` and `end`.
        """
        if closest:
            mapping = list(filterfalse(lambda x: x is None, self[map_name]))
            map_min, map_max = min(mapping), max(mapping)
            reverse_start = start > map_max
            reverse_end = end > map_min
            _start, _end = starmap(
                lambda c, r: self.get_closest(map_name, c, reverse=r),
                [(start, reverse_start), (end, reverse_end)]
            )
            if _start is None or _end is None:
                raise AmbiguousMapping(
                    f"Failed mapping {(start, end)}->{(_start, _end)} using map {map_name}")
        else:
            _start, _end = self.get_item(map_name, start), self.get_item(map_name, end)

        return _start, _end

    def relate(
            self, other: ChainSequence, map_name: str,
            link_name: str, link_name_points_to: str = 'i',
            save: bool = True, map_name_in_other: str | None = None,
    ) -> list[t.Any]:
        """
        Relate mapping from this sequence with `other` via some common "link" sequence.

        The "link" sequence is a part of `other` pointing to some sequence in this instance.

        To provide an example, consider the case of transferring the mapping to alignment
        positions `aln_map`. To do this, the `other` must be mapped to some sequence within
        this instance -- typically to the canonical numbering -- via some stored `map_canonical`
        sequence. Thus, one would use::
            this.transfer_to(
                other, map_name=aln_map,
                link_name=map_canonical, link_name_points_to="i")

        :param other: arbitrary chain sequence.
        :param map_name: the name of the sequence to transfer.
        :param link_name: the name of the "link" sequence that connects `self` and `other`.
        :param link_name_points_to: values within this instance the "link" sequence points to.
        :param save: store the obtained sequence within the `other`.
        :param map_name_in_other: the name of the mapped sequence to store within the `other`.
            By default, the `map_name` is used.
        :return: the mapped sequence.
        """
        mapping = self.get_map(link_name_points_to)
        mapped = list(map(
            lambda x: x if x is None else x._asdict()[map_name],
            (mapping.get(x) for x in other[link_name])))
        if save:
            other.add_seq(map_name_in_other or map_name, mapped)
        return mapped

    def coverage(
            self, map_names: abc.Sequence[str] | None = None,
            save: bool = True, prefix: str = 'cov',
    ) -> dict[str, float]:
        """
        Calculate maps' coverage, i.e., the number of non-empty elements.

        :param map_names: optionally, provide the sequence of map names
            to calculate coveage for.
        :param save: save the results to :attr:`meta`
        :param prefix: if `save` is ``True``, format keys f"{prefix}_{name}"
            for the :attr:`meta` dictionary.
        :return:
        """
        df = self.as_df()
        map_names = map_names or self.fields[3:]
        size = len(df)
        cov = {f'{prefix}_{n}': (~df[n].isna()).sum() / size for n in map_names}
        if save:
            self.meta.update(cov)
        return cov

    @lru_cache()
    def get_map(self, key: str) -> dict[t.Any, namedtuple]:
        """
        Obtain the mapping of the form "key->item(seq_name=*,...)".

        :param key: map name.
        :return: `dict` mapping key values to items.
        """
        return dict(zip(self[key], iter(self)))

    def get_item(self, key: str, value: t.Any) -> namedtuple:
        """
        Get a specific item. Same as :meth:`get_map`, but uses `value` to retrieve
        the needed item immediately.

        :param key: map name.
        :param value: sequence value of the sequence under the `key` name.
        :return: an item correpsonding to the desired sequence element.
        """
        return self.get_map(key)[value]

    def get_closest(
            self, key: str, value: Ord, *, reverse: bool = False
    ) -> t.Optional[namedtuple]:
        """
        Find the closest item for which item.key >=/<= value.
        By default, the search starts from the sequence's beggining, and expands towards the end
        until the first element for which the retrieved value >= the provided `value`.
        If the `reverse` is ``True``, the search direction is reversed.

        :param key: map name.
        :param value: map value. Must support comparison operators.
        :param reverse: reverse the sequence order and the comparison operator.
        :return: The first relevant item or `None` if no relevant items were found.
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

    @lru_cache
    def as_df(self) -> pd.DataFrame:
        """
        :return: The pandas DataFrame representation of the sequence where
            each column correspond to a sequence or map.
        """
        return pd.DataFrame(iter(self))

    @lru_cache
    def as_np(self) -> np.ndarray:
        """
        :return: The numpy representation of a sequence as matrix.
            This is a shortcut to :meth:`as_df`().values.
        """
        return self.as_df().values

    def spawn_child(
            self, start: int, end: int, name: t.Optional[str] = None, *,
            map_from: t.Optional[str] = None,
            map_closest: bool = False,
            deep_copy: bool = False, keep: bool = True
    ) -> ChainSequence:
        # TODO: spawning without deep copy shares meta -> copy meta by default!
        """
        Spawn the sub-sequence from the current instance.

        :param start: start of the sub-sequence.
        :param end: end of the sub-sequence.
        :param name: the name of the spawned child sequence.
        :param map_from: optionally, the map name the boundaries correspond to.
        :param map_closest: map to closest `start`, `end` boundaries (see :meth:`map_boundaries`).
        :param deep_copy: 
        :param keep:
        :return:
        """
        if map_from:
            start, end = map(
                lambda x: x._asdict()['i'],
                self.map_boundaries(start, end, map_from, map_closest))

        name = name or self.name

        child = self.sub(start, end, deep_copy=deep_copy, handle_mode='self')
        child.name = name

        if keep:
            self.children[child.id] = child

        return child

    @classmethod
    def from_file(
            cls, inp: Path | TextIOBase | abc.Iterable[str],
            reader: SeqReader = read_fasta,
            start: t.Optional[int] = None,
            end: t.Optional[int] = None,
            name: t.Optional[str] = None,
            meta: dict[str, t.Any] | None = None,
            **kwargs
    ):
        seqs = list(reader(inp))
        if not seqs:
            raise MissingData('No sequences in provided inp')
        if len(seqs) > 1:
            raise ValueError('Input contains more than one sequence')

        seq = seqs.pop()

        start = start or 1
        end = end or start + len(seq[1]) - 1

        if name is None:
            name = seq[0]

        return cls(start, end, name, meta=meta, seqs={SeqNames.seq1: seq[1], **kwargs})

    @classmethod
    def from_string(
            cls, s: str,
            start: t.Optional[int] = None,
            end: t.Optional[int] = None,
            name: t.Optional[str] = None,
            meta: dict[str, t.Any] | None = None,
            **kwargs
    ) -> ChainSequence:
        start = start or 1
        end = end or start + len(s) - 1

        return cls(start, end, name, meta=meta, seqs={SeqNames.seq1: s, **kwargs})

    @classmethod
    def from_df(
            cls, df: pd.DataFrame, name: t.Optional[str] = None,
            meta: dict[str, t.Any] | None = None
    ):
        if 'i' not in df.columns:
            raise InitError('Must contain the "i" column')
        assert len(df) >= 1
        start, end = df['i'].iloc[0], df['i'].iloc[-1]
        seqs = {col: list(df[col]) for col in df.columns if col != 'i'}
        return cls(start, end, name, meta=meta, seqs=seqs)

    @classmethod
    def read(
            cls, base_dir: Path,
            dump_names: DumpNames = DumpNames,
            *,
            search_children: bool = False
    ) -> ChainSequence:
        files = get_files(base_dir)
        dirs = get_dirs(base_dir)

        if dump_names.sequence not in files:
            raise InitError(f'{dump_names.sequence} must be present')

        if dump_names.meta in files:
            df = pd.read_csv(
                files[dump_names.meta], sep=r'\s+', names=['Title', 'Value'])
            meta = dict(zip(df['Title'], df['Value']))
            if MetaNames.name in meta:
                name = meta[MetaNames.name]
            else:
                name = 'UnnamedSequence'
        else:
            meta, name = {}, 'UnnamedSequence'

        df = pd.read_csv(files[dump_names.sequence], sep='\t')
        seq = cls.from_df(df, name)
        seq.meta = meta

        if dump_names.variables in files:
            seq.variables = Variables.read(files[dump_names.variables]).sequence

        if search_children and dump_names.segments_dir in dirs:
            for path in (base_dir / dump_names.segments_dir).iterdir():
                child = ChainSequence.read(path, dump_names, search_children=True)
                child.parent = seq
                seq.children[child.name] = child

        return seq

    def write_seq(
            self, path: Path, fields: t.Optional[t.Container[str]] = None, sep: str = '\t'
    ) -> None:
        self.as_df().drop_duplicates().to_csv(path, index=False, columns=fields, sep=sep)

    def write_meta(self, path: Path, sep='\t'):
        items = (f'{k}{sep}{v}' for k, v in self.meta.items()
                 if isinstance(v, (str, int, float)))
        path.write_text('\n'.join(items))

    def write(self, base_dir: Path, dump_names: DumpNames = DumpNames, *,
              write_children: bool = False) -> None:
        base_dir.mkdir(exist_ok=True, parents=True)
        self.write_seq(base_dir / dump_names.sequence)
        if self.meta:
            self.write_meta(base_dir / dump_names.meta)
            if self.variables:
                self.variables.write(base_dir / dump_names.variables)
        if write_children:
            for c in self.children.values():
                child_dir = base_dir / dump_names.segments_dir / c.name
                c.write(child_dir, dump_names, write_children=write_children)


class ChainStructure:
    __slots__ = ('pdb', 'seq', 'parent', 'variables', 'children')

    def __init__(
            self, pdb_id: str, pdb_chain: str,
            pdb_structure: GenericStructure,
            seq: t.Optional[ChainSequence] = None,
            parent: t.Optional[ChainStructure] = None,
            children: t.Optional[dict[str, ChainStructure]] = None,
            variables: t.Optional[Variables] = None,
    ):
        self.pdb = PDB_Chain(pdb_id, pdb_chain, pdb_structure)
        validate_chain(self.pdb)
        self.seq = seq
        self.parent = parent
        self.variables = variables or Variables()
        self.children = children or {}

        if self.seq is None:
            seq1, seq3, num = map(list, unzip(self.pdb.structure.get_sequence()))
            seqs = {SeqNames.seq3: seq3, SeqNames.enum: num}
            self.seq = ChainSequence.from_string(
                ''.join(seq1), name=f'{pdb_id}{Sep.chain}{pdb_chain}', **seqs)

        self.seq.meta[MetaNames.pdb_id] = pdb_id
        self.seq.meta[MetaNames.pdb_chain] = pdb_chain

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def iter_children(self) -> abc.Generator[list[ChainStructure]]:
        return topo_iter(self, lambda x: iter(x.children.values()))

    @property
    def id(self) -> str:
        return f'{self.__class__.__name__}({self.seq})'

    @property
    def array(self) -> bst.AtomArray:
        return self.pdb.structure.array

    @classmethod
    def from_structure(
            cls, structure: bst.AtomArray | GenericStructure,
            pdb_id: t.Optional[str] = None,
    ) -> ChainStructure:
        if isinstance(structure, bst.AtomArray):
            structure = GenericStructure(structure)

        chain_id = structure.array.chain_id[0]

        if pdb_id is None:
            pdb_id = structure.pdb_id

        return cls(pdb_id, chain_id, structure)

    def superpose(
            self, other: ChainStructure,
            res_id: abc.Sequence[int] | None = None,
            atom_names: abc.Sequence[abc.Sequence[str]] | abc.Sequence[str] | None = None,
            map_name_self: str | None = None, map_name_other: str | None = None,
            mask_self: np.ndarray | None = None, mask_other: np.ndarray | None = None,
            inplace: bool = False, rmsd_to_meta: bool = True,
    ):
        def _get_mask(c: ChainStructure, map_name: str) -> np.ndarray:
            if res_id is None:
                return np.ones_like(c.array, bool)
            _res_id = res_id if not map_name or res_id is None else [
                c.seq.get_item(map_name, x)._asdict()[SeqNames.enum] for x in res_id]
            return filter_selection(c.array, _res_id, atom_names)

        match atom_names:
            case [str(), *xs]:
                if res_id is not None:
                    atom_names = [atom_names] * len(res_id)
            case [[str(), *xs], *ys]:
                if len(res_id) != len(atom_names):
                    raise LengthMismatch(
                        'When specifying `atom_names` per residue, the number of '
                        f'residues must match the number of atom name groups; Got {len(res_id)} '
                        f'residues and {len(atom_names)} atom names groups.'
                    )

        if mask_self is None:
            mask_self = _get_mask(self, map_name_self)
        if mask_other is None:
            mask_other = _get_mask(other, map_name_other)

        superposed, rmsd, transformation = self.pdb.structure.superpose(
            other.pdb.structure, mask_self=mask_self, mask_other=mask_other)

        if inplace:
            other.pdb.structure = superposed
        else:
            other = ChainStructure(
                other.pdb.id, other.pdb.chain, superposed, other.seq,
                parent=other.parent, children=other.children,
                variables=other.variables
            )

        if rmsd_to_meta:
            map_name = map_name_other or other.seq.name
            other.seq.meta[f'rmsd_{map_name}'] = rmsd

        return other, rmsd, transformation

    def spawn_child(
            self, start: int, end: int, name: t.Optional[str] = None, *,
            map_from: t.Optional[str] = None,
            map_closest: bool = True,
            keep_seq_child: bool = False,
            keep: bool = True,
            deep_copy: bool = False
    ) -> ChainStructure:

        if start > end:
            raise ValueError(f'Invalid boundaries {start, end}')

        name = name or self.seq.name

        seq = self.seq.spawn_child(
            start, end, name, map_from=map_from, map_closest=map_closest,
            deep_copy=deep_copy, keep=keep_seq_child
        )
        structure = None
        if self.pdb.structure:
            enum_field = seq.field_names().enum
            start, end = seq[enum_field][0], seq[enum_field][-1]
            structure = self.pdb.structure.sub_structure(start, end)

        child = ChainStructure(self.pdb.id, self.pdb.chain, structure, seq, self)
        if keep:
            self.children[child.id] = child
        return child

    @classmethod
    def read(
            cls, base_dir: Path, dump_names: DumpNames = DumpNames, *,
            search_children: bool = False,
    ) -> ChainStructure:

        files = get_files(base_dir)
        dirs = get_dirs(base_dir)
        variables = None

        bname = dump_names.structure_base_name
        stems = {p.stem: p.name for p in files.values()}
        if bname not in stems:
            raise InitError(f'{base_dir} must contain {bname}.fmt '
                            f'where "fmt" is supported structure format')
        structure = GenericStructure.read(base_dir / stems[bname])

        seq = ChainSequence.read(base_dir, dump_names, search_children=False)
        pdb_id = seq.meta.get(MetaNames.pdb_id, 'UnkPDB')
        chain_id = seq.meta.get(MetaNames.pdb_chain, 'UnkChain')

        if dump_names.variables in files:
            variables = Variables.read(files[dump_names.variables]).structure

        cs = ChainStructure(pdb_id, chain_id, structure, seq, variables=variables)

        if search_children and dump_names.segments_dir in dirs:
            for path in (base_dir / dump_names.segments_dir).iterdir():
                child = ChainStructure.read(path, dump_names, search_children=True)
                child.parent = cs
                cs.children[path.name] = child

        return cs

    def write(
            self, base_dir: Path, fmt: str = 'cif', dump_names: DumpNames = DumpNames, *,
            write_children: bool = False,
    ) -> t.NoReturn:
        # if base_dir.name != dump_names.structures_dir:
        #     base_dir /= dump_names.structures_dir
        base_dir.mkdir(exist_ok=True, parents=True)

        self.seq.write(base_dir)
        self.pdb.structure.write(base_dir / f'{dump_names.structure_base_name}.{fmt}')
        if self.variables:
            self.variables.write(base_dir / dump_names.variables)

        if write_children:
            for child in self.children.values():
                child_dir = base_dir / DumpNames.segments_dir / child.seq.name
                child.write(child_dir, fmt, dump_names, write_children=True)


class Chain(AbstractChain):
    """
    A mutable container, holding data associated with a singe (full) protein chain.
    """

    __slots__ = ('seq', 'structures', 'parent', 'children')

    def __init__(
            self, seq: ChainSequence,
            structures: t.Optional[t.List[ChainStructure]] = None,
            parent: t.Optional[Chain] = None,
            children: t.Optional[t.Dict[str, Chain]] = None,
    ):
        self.seq = seq
        self.structures = structures or []
        self.parent = parent
        self.children = children or {}

    @property
    def id(self) -> str:
        return self.seq.id

    def __repr__(self) -> str:
        return self.id

    def __str__(self) -> str:
        return self.id

    def __getitem__(self, key: str | int) -> Chain:
        if isinstance(key, str):
            return self.children[key]
        if isinstance(key, int):
            return list(self.children.values())[key]
        else:
            raise TypeError('Wrong key type')

    def __contains__(self, item: Chain) -> bool:
        return item in self.children

    def iter_children(self) -> abc.Generator[list[CT]]:
        return topo_iter(self, lambda x: iter(x.children.values()))

    @classmethod
    def from_seq(
            cls, inp: str | tuple[str, str] | Path | TextIOBase | abc.Iterable[str],
            read_method: SeqReader = read_fasta
    ) -> Chain | list[Chain]:
        if isinstance(inp, str):
            return cls(ChainSequence.from_string(inp))
        elif isinstance(inp, tuple):
            name, s = inp
            return cls(ChainSequence.from_string(s, name=name))
        return [cls(ChainSequence.from_string(seq, name=name)) for name, seq in read_method(inp)]

    @classmethod
    def read(
            cls, path: Path, dump_names: DumpNames = DumpNames,
            *, search_children: bool = False
    ) -> Chain:
        seq = ChainSequence.read(path, dump_names, search_children=False)

        structures = [ChainStructure.read(p, dump_names)
                      for p in (path / dump_names.structures_dir).glob('*')]
        protein = Chain(seq, structures)
        if search_children:
            for child_path in (path / dump_names.segments_dir).glob('*'):
                protein.children[child_path.name] = Chain.read(
                    child_path, dump_names, search_children=True)
        return protein

    def write(
            self, base_dir: Path, dump_names: DumpNames = DumpNames,
            *, str_fmt: str = 'cif', write_children: bool = True,
    ) -> t.NoReturn:

        base_dir.mkdir(parents=True, exist_ok=True)

        self.seq.write(base_dir, dump_names, write_children=False)

        if self.structures:
            str_dir = base_dir / dump_names.structures_dir
            str_dir.mkdir(exist_ok=True)
            for s in self.structures:
                s.write(str_dir / s.id, str_fmt, dump_names, write_children=False)

        for name, prot in self.children.items():
            prot.write(base_dir / dump_names.segments_dir / prot.id)

    def add_structure(
            self, structure: ChainStructure, *, check_ids: bool = True,
            map_to_seq: bool = True, map_name: str = SeqNames.map_canonical,
            **kwargs
    ) -> t.NoReturn:
        if check_ids:
            ids = [s.id for s in self.structures]
            if structure.id in ids:
                raise ValueError(f'Protein already contains structure {structure.id}. '
                                 f'Remove it first or disable `check_ids`')
        if map_to_seq:
            structure.seq.map_numbering(self.seq, name=map_name, **kwargs)
        self.structures.append(structure)

    def transfer_seq_mapping(
            self, map_name: str, link_map: str = SeqNames.map_canonical,
            link_map_points_to: str = SeqNames.enum,
            map_name_in_other: str | None = None,
            **kwargs
    ) -> t.NoReturn:
        """Transfer sequence mapping to structure sequences"""
        for s in self.structures:
            self.seq.relate(
                s.seq, map_name, link_map, link_map_points_to,
                map_name_in_other=map_name_in_other, **kwargs)

    def spawn_child(
            self, start: int, end: int, name: None | str = None, *,
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

        def subset_structure(structure: ChainStructure) -> t.Optional[ChainStructure]:
            try:
                return structure.spawn_child(
                    start, end, name, map_from=str_map_from, map_closest=str_map_closest,
                    deep_copy=str_deep_copy, keep=str_keep_child, keep_seq_child=str_seq_keep_child)
            except (AmbiguousMapping, MissingData, NoOverlap) as e:
                msg = f'Failed to spawn substructure using boundaries {start, end} due to {e}'
                if not tolerate_failure:
                    raise e
                LOGGER.warning(msg)
                return None

        name = name or self.seq.name

        seq = self.seq.spawn_child(
            start, end, name, map_from=seq_map_from, map_closest=seq_map_closest,
            deep_copy=seq_deep_copy, keep=seq_keep_child)

        structures = None
        if subset_structures:
            structures = list(filter(bool, map(subset_structure, self.structures)))

        child = Chain(seq, structures, self)
        if keep:
            self.children[name] = child
        return child


CT = t.TypeVar('CT', bound=Chain | ChainSequence | ChainStructure)  # "Chain" type
SS = t.TypeVar('SS', bound=ChainSequence | ChainStructure)
ST = t.TypeVar('ST', bound=Segment)


class ChainList(abc.MutableSequence[CT]):
    __slots__ = ('_chains', '_type')

    def __init__(self, chains: abc.Iterable[CT]):
        self._chains: list[CT] = list(chains)
        self._type = None
        self._check_match_and_set_type(self._infer_type(self._chains))

    @staticmethod
    def _infer_type(objs: abc.Sequence[T]) -> t.Optional[str]:
        types = set(map(type, objs))
        if len(types) > 1:
            raise TypeError(f'ChainList elements must have single type; got {types}')
        if objs:
            match objs[0]:
                case Chain():
                    return 'chain'
                case ChainSequence():
                    return 'seq'
                case ChainStructure():
                    return 'str'
                case _:
                    raise TypeError(f'Unsupported type {objs[0]}')
        else:
            return None

    @property
    def type(self) -> str:
        return self._type

    def _check_match_and_set_type(self, x: str) -> t.NoReturn:
        if self._type is not None:
            if x != self._type:
                raise TypeError(
                    f"Supplied type doesn't match existing type {self._type}")
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
                return self.filter(lambda x: index in x.id)
            case _:
                raise TypeError(f'Incorrect index type {type(index)}')

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
                    self._chains, default=False, pred=lambda c: c.id == item)
            case Chain() | ChainStructure() | ChainSequence():
                return item in self._chains
            case _:
                return False

    def __add__(self, other: ChainList | abc.Iterable):
        match other:
            case abc.Iterable():
                return ChainList(self._chains + list(other))
            case ChainList():
                return ChainList(self._chains + other._chains)
            case _:
                raise TypeError(f'Unsupported type {type(other)}')

    def __repr__(self) -> str:
        return self._chains.__repr__()

    def __iter__(self) -> abc.Iterator[CT]:
        return iter(self._chains)

    def index(self, value: CT, start: int = ..., stop: int = ...) -> int:
        return self._chains.index(value, start, stop)

    def insert(self, index: int, value: CT) -> t.NoReturn:
        self._check_match_and_set_type(self._infer_type([value]))
        self._chains.insert(index, value)

    def iter_children(self) -> abc.Generator[ChainList[CT]]:
        levels = zip_longest(*map(lambda c: c.iter_children(), self._chains))
        for level in levels:
            yield ChainList(chain.from_iterable(level))

    def get_level(self, n: int) -> ChainList[CT]:
        if n == 0:
            return self
        return nth(self.iter_children(), n - 1, default=ChainList([]))

    def collapse_children(self) -> ChainList[CT] | list:
        children = list(chain.from_iterable(self.iter_children()))
        if children:
            return ChainList(children)
        return children

    def iter_sequences(self) -> abc.Generator[ChainSequence]:
        match self._type:
            case 'chain' | 'str':
                yield from (c.seq for c in self._chains)
            case _:
                yield from iter(self._chains)

    def iter_structures(self) -> abc.Generator[ChainStructure]:
        match self._type:
            case 'chain':
                yield from chain.from_iterable(c.structures for c in self._chains)
            case 'str':
                yield from iter(self._chains)
            case _:
                yield from iter([])

    def iter_structure_sequences(self) -> abc.Iterator[ChainSequence]:
        yield from (s.seq for s in self.iter_structures())

    @staticmethod
    def _to_segment(s: abc.Sequence[int, int] | Segment) -> Segment:
        match s:
            case (start, end):
                return Segment(start, end)
            case Segment():
                return s
            case _:
                raise TypeError(f'Unsupported type {type(s)}')

    @staticmethod
    def _get_seg_matcher(s: str) -> abc.Callable[[ChainSequence, Segment, t.Optional[str]], bool]:
        def matcher(seq: ChainSequence, seg: Segment, map_name: t.Optional[str] = None) -> bool:
            if map_name is not None:
                # Get elements in the seq whose mapped sequence matches seg boundaries
                start = seq.get_closest(map_name, seg.start)._asdict()[map_name]
                end = seq.get_closest(map_name, seg.end, reverse=True)._asdict()[map_name]
                # If not such elements -> no match
                if start is None or end is None:
                    return False
                # Create a new temporary segment using the mapped boundaries
                seq = Segment(start, end)
            match s:
                case 'overlap':
                    return seq.overlaps(seg)
                case 'bounded':
                    return seq.bounded_by(seg)
                case 'bounding':
                    return seq.bounds(seg)
                case _:
                    raise ValueError(f'Invalid matching mode {s}')

        return matcher

    @staticmethod
    def _get_pos_matcher(ps: abc.Iterable[Ord]) -> abc.Callable[[ChainSequence, t.Optional[str]], bool]:
        def matcher(seq: ChainSequence, map_name: t.Optional[str] = None) -> bool:
            obj = seq
            if map_name:
                obj = seq[map_name]
            return all(p in obj for p in ps)

        return matcher

    def _filter_seqs(
            self, seqs: abc.Iterable[ChainSequence], match_type: str,
            s: Segment | abc.Collection[Ord],
            map_name: t.Optional[str]
    ) -> abc.Iterator[bool]:
        match s:
            case Segment():
                match_fn = partial(
                    self._get_seg_matcher(match_type),
                    seg=self._to_segment(s), map_name=map_name)
            case abc.Collection():
                match_fn = partial(
                    self._get_pos_matcher(s), map_name=map_name)
            case _:
                raise TypeError(f'Unsupported type to match {type(s)}')
        return map(match_fn, seqs)

    def _filter_str(
            self, structures: abc.Iterable[ChainStructure], match_type: str,
            s: abc.Sequence[int, int] | Segment, map_name: t.Optional[str]
    ) -> abc.Iterator[bool]:
        return self._filter_seqs(
            map(lambda x: x.seq, structures), match_type, s, map_name)

    def filter_pos(
            self, s: Segment | abc.Collection[Ord], *,
            obj_type: str = 'seq', match_type: str = 'overlap',
            map_name: t.Optional[str] = None
    ) -> ChainList[SS]:
        # Use cases:
        # 1) search using Segment's start/end => good for matching canonical sequence
        # 2) use map_name => overlapping with ref/aln/etc-based boundaries; esp. for structures
        match obj_type[:3]:
            case 'seq':
                objs, fn, _type = self.iter_sequences(), self._filter_seqs, ChainSequence
            case 'str':
                objs, fn, _type = self.iter_structures(), self._filter_str, ChainStructure
            case _:
                raise ValueError(f'Unsupported object type {obj_type}')
        objs1, objs2 = tee(objs)
        mask = fn(objs1, match_type, s, map_name)
        res: ChainList[_type] = ChainList(
            map(op.itemgetter(1), filter(lambda x: x[0], zip(mask, objs2))))
        return res

    def filter(self, pred: abc.Callable[[CT], bool]) -> ChainList[CT]:
        return ChainList(filter(pred, self))

    def apply(self, fn: abc.Callable[[CT, ...], CT], *args, **kwargs) -> ChainList[CT]:
        return ChainList([fn(c, *args, **kwargs) for c in self])


@curry
def _read_obj(path: Path, obj_type: t.Type[CT], tolerate_failures: bool, **kwargs):
    try:
        return obj_type.read(path, **kwargs)
    except Exception as e:
        LOGGER.exception(e)
        if not tolerate_failures:
            raise e
    return


@curry
def _write_obj(obj: CT, path: Path, tolerate_failures: bool, **kwargs) -> Path:
    try:
        obj.write(path, **kwargs)
        return path
    except Exception as e:
        LOGGER.warning(f'Failed to initialize {obj} from {path}')
        LOGGER.exception(e)
        if not tolerate_failures:
            raise e


class ChainIO:
    # TODO: implement context manager
    def __init__(
            self, num_proc: None | int = None, verbose: bool = False,
            tolerate_failures: bool = False, dump_names: DumpNames = DumpNames,
    ):
        self.num_proc = num_proc
        self.verbose = verbose
        self.tolerate_failures = tolerate_failures
        self.dump_names = dump_names

    def _read(
            self, obj_type: t.Type[CT], path: Path | abc.Iterable[Path],
            non_blocking: bool = False, **kwargs
    ) -> t.Optional[CT] | abc.Iterator[t.Optional[CT]]:

        if isinstance(path, Path):
            dirs = get_dirs(path)
        else:
            dirs = {p.name: p for p in path if p.is_dir()}

        _read = _read_obj(obj_type=obj_type, tolerate_failures=self.tolerate_failures, **kwargs)

        if DumpNames.segments_dir in dirs or not dirs and isinstance(path, Path):
            return _read(path)

        dirs = dirs.values()

        _read = _read_obj(obj_type=obj_type, tolerate_failures=self.tolerate_failures, **kwargs)

        if self.num_proc is None:

            if self.verbose:
                dirs = tqdm(dirs, desc=f'Reading {obj_type}')

            yield from map(_read, dirs)

        else:

            with ProcessPoolExecutor(self.num_proc) as executor:

                futures = as_completed([executor.submit(_read, d) for d in dirs])

                if non_blocking:
                    return futures

                if self.verbose:
                    futures = tqdm(futures, desc=f'Reading {obj_type}')

                for future in futures:
                    yield future.result()

    def write(
            self, objs: CT | abc.Iterable[CT], base: Path, non_blocking: bool = False, **kwargs
    ) -> abc.Iterator[Future] | abc.Iterator[Path] | t.NoReturn:
        if isinstance(objs, (ChainSequence, ChainStructure, Chain)):
            objs.write(base)
        else:
            _write = _write_obj(tolerate_failures=self.tolerate_failures, **kwargs)

            if self.num_proc is None:
                if self.verbose:
                    objs = tqdm(objs, desc='Writing objects')
                for obj in objs:
                    yield _write(obj, base / obj.id)
            else:
                with ProcessPoolExecutor(self.num_proc) as executor:

                    futures = as_completed(
                        [executor.submit(_write, obj, base / obj.id) for obj in objs])

                    if non_blocking:
                        return futures

                    if self.verbose:
                        futures = tqdm(futures, desc='Writing objects')

                    for future in futures:
                        yield future.result()

    def read_chain(
            self, path: Path | abc.Iterable[Path], **kwargs
    ) -> t.Optional[Chain] | abc.Iterator[t.Optional[Chain]]:
        return self._read(Chain, path, **kwargs)

    def read_chain_seq(
            self, path: Path | abc.Iterable[Path], **kwargs
    ) -> t.Optional[ChainSequence] | abc.Iterator[t.Optional[ChainSequence]]:
        return self._read(ChainSequence, path, **kwargs)

    def read_chain_struc(
            self, path: Path | abc.Iterable[Path], **kwargs
    ) -> t.Optional[ChainStructure] | abc.Iterator[t.Optional[ChainStructure]]:
        return self._read(ChainStructure, path, **kwargs)


class InitializerCallback(t.Protocol):

    @t.overload
    def __call__(self, inp: CT) -> CT | None: ...

    @t.overload
    def __call__(self, inp: list[ChainStructure]) -> list[ChainStructure] | None: ...

    @t.overload
    def __call__(self, inp: None) -> None: ...

    def __call__(self, inp: CT | list[ChainStructure] | None) -> CT | list[ChainStructure] | None: ...


def _read_path(x, tolerate_failures, supported_seq_ext, supported_str_ext):
    if x.suffix in supported_seq_ext:
        return ChainSequence.from_file(x)
    elif x.suffix in supported_str_ext:
        return [ChainStructure.from_structure(c) for c in GenericStructure.read(x).split_chains()]
    else:
        if tolerate_failures:
            return None
        raise InitError(f'Suffix {x.suffix} of the path {x} is not supported')


def _init(x, tolerate_failures, supported_seq_ext, supported_str_ext, callbacks):
    match x:
        case ChainSequence() | ChainStructure():
            res = x
        case [str(), str()]:
            res = ChainSequence.from_string(x[1], name=x[0])
        case [Path(), xs]:
            structures = _read_path(x[0], tolerate_failures, supported_seq_ext, supported_str_ext)
            structures = [s for s in structures if s.pdb.chain in xs]
            res = structures or None
        case GenericStructure():
            res = ChainStructure.from_structure(x)
        case Path():
            res = _read_path(x, tolerate_failures, supported_seq_ext, supported_str_ext)
        case _:
            res = None
            if not tolerate_failures:
                raise InitError(f'Unsupported input type {type(x)}')
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
) -> abc.Iterator[list[int]]:
    if num_proc:
        with ProcessPoolExecutor(num_proc) as executor:
            yield from executor.map(_map_numbering, seqs, repeat(obj_to_map))
    else:
        yield from (x.map_numbering(obj_to_map, save=False) for x in seqs)


def map_numbering_many2many(
        objs_to_map: abc.Sequence[str | tuple[str, str] | ChainSequence | Alignment],
        seq_groups: abc.Sequence[abc.Sequence[ChainSequence]],
        num_proc: t.Optional[int] = None, verbose: bool = False,
):
    if len(objs_to_map) != len(seq_groups):
        raise LengthMismatch(
            f'The number of objects to map {len(objs_to_map)} != '
            f'the number of sequence groups {len(seq_groups)}')
    staged = chain.from_iterable(
        ((obj, s) for s in g) for obj, g in zip(objs_to_map, seq_groups))
    group_sizes = map(len, seq_groups)
    if num_proc:
        objs, seqs = unzip(staged)
        with ProcessPoolExecutor(num_proc) as executor:
            results = executor.map(_map_numbering, seqs, objs)
            if verbose:
                results = tqdm(results, desc='Mapping numberings')
            yield from split_into(results, group_sizes)
    else:
        results = (s.map_numbering(o, save=False) for o, s in staged)
        if verbose:
            results = tqdm(results, desc='Mapping numberings')
        yield from split_into(results, group_sizes)


class ChainInitializer:
    def __init__(
            self, num_proc: int | None = None, tolerate_failures: bool = False,
            verbose: bool = False):
        self.num_proc = num_proc
        self.tolerate_failures = tolerate_failures
        self.verbose = verbose

    @property
    def supported_seq_ext(self) -> list[str]:
        return ['.fasta']

    @property
    def supported_str_ext(self) -> list[str]:
        return ['.cif', '.pdb', '.pdbx', '.mmtf', '.npz']

    def from_iterable(
            self, it: abc.Iterable[
                SS | Path | tuple[Path, abc.Sequence[str]] | tuple[str, str] | GenericStructure],
            callbacks: list[InitializerCallback] | None = None
    ):
        """Initialize `ChainSequence` or `ChainStructure` objects from inputs."""
        if self.num_proc is not None:
            with ProcessPoolExecutor(self.num_proc) as executor:
                futures = [
                    (x, executor.submit(
                        _init, x, self.tolerate_failures, self.supported_seq_ext,
                        self.supported_str_ext, callbacks))
                    for x in it]
                if self.verbose:
                    futures = tqdm(futures, desc='Initializing objects in parallel')
                for x, future in futures:
                    try:
                        yield future.result()
                    except Exception as e:
                        LOGGER.warning(f'Input {x} failed with an error {e}')
                        # LOGGER.exception(e)
                        if not self.tolerate_failures:
                            raise e
                        yield None
        else:
            if self.verbose:
                it = tqdm(it, desc='Initializing objects sequentially')
            yield from (
                _init(x, self.tolerate_failures, self.supported_seq_ext,
                      self.supported_str_ext, callbacks)
                for x in it)

    def from_mapping(
            self, m: abc.Mapping[
                ChainSequence | tuple[str, str] | Path,
                abc.Sequence[
                    ChainStructure | GenericStructure | bst.AtomArray | Path |
                    tuple[Path, abc.Sequence[str]]]
            ], key_callbacks: t.Optional[list[InitializerCallback]] = None,
            val_callbacks: t.Optional[list[InitializerCallback]] = None,
            **kwargs
    ):
        """Initialize `Chain` objects from mapping between sequences and structures."""
        # Process keys and values
        keys = self.from_iterable(m, callbacks=key_callbacks)  # ChainSequences
        values_flattened = self.from_iterable(  # ChainStructures
            chain.from_iterable(m.values()), callbacks=val_callbacks)
        values = split_into(values_flattened, map(len, m.values()))

        m_new = valmap(
            lambda vs: collapse(filter(bool, vs)),
            keymap(
                Chain,  # create `Chain` objects from `ChainSequence`s
                keyfilter(bool, dict(zip(keys, values)))  # Filter possible failures
            )
        )

        if self.num_proc is None:
            for c, ss in m_new.items():
                for s in ss:
                    c.add_structure(s, **kwargs)
        else:
            map_name = kwargs.get('map_name') or SeqNames.map_canonical

            # explicitly unpack value iterable into lists
            m_new = valmap(list, m_new)

            # create numbering groups -- lists of lists with numberings for each structure in values
            numbering_groups = map_numbering_many2many(
                [x.seq for x in m_new], [[x.seq for x in val] for val in m_new.values()],
                num_proc=self.num_proc, verbose=self.verbose
            )
            for (c, ss), num_group in zip_equal(m_new.items(), numbering_groups):
                if len(num_group) != len(ss):
                    raise LengthMismatch(
                        f'The number of mapped numberings {len(num_group)} must match '
                        f'the number of structures {len(ss)}.')
                for s, n in zip(ss, num_group):
                    try:
                        s.seq.add_seq(map_name, n)
                        c.add_structure(s, map_to_seq=False, **kwargs)
                    except Exception as e:
                        LOGGER.warning(f'Failed to add structure {s} to chain {c} due to {e}')
                        LOGGER.exception(e)
                        if not self.tolerate_failures:
                            raise e

        return list(m_new)


if __name__ == '__main__':
    pass
