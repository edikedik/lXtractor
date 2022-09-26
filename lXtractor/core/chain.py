from __future__ import annotations

import logging
import typing as t
from collections import namedtuple, abc
from functools import lru_cache
from io import TextIOBase
from itertools import starmap
from pathlib import Path

import biotite.structure as bst
import pandas as pd
from more_itertools import unzip, first_true

from lXtractor.core.alignment import Alignment
from lXtractor.core.base import AminoAcidDict, AbstractChain, Ord, AlignMethod, SeqReader
from lXtractor.core.config import Sep, DumpNames, SeqNames
from lXtractor.core.exceptions import MissingData, AmbiguousMapping, InitError
from lXtractor.core.segment import Segment
from lXtractor.core.structure import GenericStructure, PDB_Chain, validate_chain
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.seq import mafft_align, map_pairs_numbering, read_fasta
from lXtractor.variables.base import Variables

LOGGER = logging.getLogger(__name__)


# TODO: variable calculation may depend on the original structure and should NOT require the extracted one
# -> find the closest parent with the structure having ID matching such of the assigned variable


class ChainSequence(Segment):

    __slots__ = ()

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self._seqs.keys())

    @classmethod
    def field_names(cls):
        return SeqNames

    @property
    def numbering(self) -> t.Sequence[int]:
        return self[SeqNames.enum]

    @property
    def seq1(self) -> str:
        return self[SeqNames.seq1]

    @property
    def seq3(self) -> t.Sequence[str]:
        return self[SeqNames.seq3]

    @property
    def variables(self) -> Variables:
        return self.meta[SeqNames.variables]

    def _setup_and_validate(self):
        super()._setup_and_validate()

        if SeqNames.seq1 not in self:
            raise MissingData(f'Requires {SeqNames.seq1} in `seqs`')
        if SeqNames.seq3 not in self:
            d = AminoAcidDict()
            self[SeqNames.seq3] = [d[c] for c in self[SeqNames.seq1]]
        if SeqNames.variables not in self.meta:
            self.meta[SeqNames.variables] = {}
        if SeqNames.enum not in self._seqs:
            self[SeqNames.enum] = list(range(self.start, self.end + 1))

    def map_numbering(
            self, other: str | tuple[str, str] | ChainSequence | Alignment,
            align_method: AlignMethod = mafft_align,
            save: bool = True, name: t.Optional[str] = None, **kwargs
    ) -> list[None | int]:

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

    @lru_cache()
    def get_map(self, key: str) -> dict[t.Any, namedtuple]:
        return {x: it for x, it in zip(self[key], iter(self))}

    def get_item(self, key: str, value: t.Any) -> namedtuple:
        return self.get_map(key)[value]

    def get_closest(
            self, key: str, value: Ord, *, reverse: bool = False
    ) -> t.Optional[namedtuple]:
        """
        Find the closest item for which item.key >=/<= value.

        :param key: Sequence name.
        :param value: Sequence value. Must support comparison operators.
        :param reverse: Reverse the sequence order and the comparison operator.
        :return: The first relevant item or `None` if no relevant items were found.
        """

        def pred(kv):
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

    # def as_record(self) -> SeqRec:
    #     return SeqRec(Seq(self.seq1), self.id, self.id, self.id)

    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame(iter(self))

    def spawn_child(
            self, start: int, end: int, *,
            deep_copy: bool = False, keep: bool = True
    ) -> ChainSequence:
        child = self.sub(start, end, deep_copy=deep_copy, handle_mode='self')
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

        return cls(start, end, name, seqs={SeqNames.seq1: seq[1], **kwargs})

    @classmethod
    def from_string(
            cls, s: str,
            start: t.Optional[int] = None,
            end: t.Optional[int] = None,
            name: t.Optional[str] = None,
            **kwargs
    ) -> ChainSequence:
        start = start or 1
        end = end or start + len(s) - 1

        return cls(start, end, name, seqs={SeqNames.seq1: s, **kwargs})

    @classmethod
    def from_df(cls, df: pd.DataFrame, name: t.Optional[str] = None):
        if 'i' not in df.columns:
            raise InitError('Must contain the "i" column')
        assert len(df) >= 1
        start, end = df['i'].iloc[0], df['i'].iloc[-1]
        seqs = {col: list(df[col]) for col in df.columns}
        return cls(start, end, name, seqs=seqs)

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

        df = pd.read_csv(base_dir / files[dump_names.sequence], sep=r'\s+')
        seq = cls.from_df(df, dump_names.sequence)

        if dump_names.meta in files:
            df = pd.read_csv(
                base_dir / dump_names.meta, sep=r'\s+', names=['Title', 'Value'])
            seq.meta = dict(zip(df['Title'], df['Value']))

        if dump_names.variables in files:
            seq.meta[dump_names.variables] = Variables.read(
                files[dump_names.variables]).sequence

        if search_children and dump_names.segments_dir in dirs:
            for path in (base_dir / dump_names.segments_dir).iterdir():
                child = ChainSequence.read(path, dump_names, search_children=True)
                child.parent = seq
                seq.children[path.name] = child

        return seq

    def write_seq(
            self, path: Path, fields: t.Optional[t.Container[str]] = None, sep: str = '\t'
    ) -> None:
        self.as_df().drop_duplicates().to_csv(path, columns=fields, sep=sep)

    def write_meta(self, path: Path, sep='\t'):
        items = (f'{k}{sep}{v}' for k, v in self.meta.items()
                 if isinstance(v, (str, int, float)))
        path.write_text('\n'.join(items))

    def write(self, base_dir: Path, dump_names: DumpNames = DumpNames) -> None:
        self.write_seq(base_dir / dump_names.sequence)
        if self.meta:
            self.write_meta(base_dir / dump_names.meta)
            if self.variables:
                self.variables.write(base_dir / dump_names.variables)

    def build_tree(self):
        raise NotImplementedError


class ChainStructure:

    __slots__ = ('pdb', 'seq', 'parent', 'variables')

    def __init__(
            self, pdb_id: str, pdb_chain: str,
            pdb_structure: t.Optional[GenericStructure] = None,
            seq: t.Optional[ChainSequence] = None,
            parent: t.Optional[ChainStructure] = None,
            variables: t.Optional[Variables] = None,
    ):
        self.pdb = PDB_Chain(pdb_id, pdb_chain, pdb_structure)
        validate_chain(self.pdb)
        if seq is None and self.pdb.structure is not None:
            seq1, seq3, num = map(list, unzip(self.pdb.structure.get_sequence()))
            seqs = {SeqNames.seq3: seq3, SeqNames.enum: num}
            self.seq = ChainSequence.from_string(
                ''.join(seq1), name=f'{pdb_id}{Sep.chain}{pdb_chain}', **seqs)
        else:
            self.seq = seq
        self.parent = parent
        self.variables = variables or Variables()

    @property
    def id(self):
        return self.seq.id

    @classmethod
    def from_structure(
            cls, structure: bst.AtomArray | GenericStructure, pdb_id: t.Optional[str] = None,
    ) -> ChainStructure:
        if isinstance(structure, bst.AtomArray):
            structure = GenericStructure(structure)

        chain_id = structure.array.chain_id[0]

        if pdb_id is None:
            pdb_id = structure.pdb_id

        return cls(pdb_id, chain_id, structure)

    def spawn_child(
            self, start: int, end: int, *,
            map_name: t.Optional[str] = None, deep_copy: bool = False
    ) -> ChainStructure:

        if start > end:
            raise ValueError(f'Invalid boundaries {start, end}')

        if map_name:
            if map_name not in self.seq:
                raise MissingData(f'Missing mapping {map_name} in {self.seq}')

            _start, _end = starmap(
                lambda c, r: self.seq.get_closest(map_name, c, reverse=r),
                [(start, False), (end, True)]
            )
            if _start is None or _end is None:
                raise AmbiguousMapping(
                    f"Couldn't map {(start, end)}->{(_start, _end)} from numbering {map_name} "
                    f"numbering to the current structure's numbering")
            _start, _end = map(lambda x: x._asdict()[SeqNames.enum], [_start, _end])
        else:
            _start, _end = start, end

        seq, structure = None, None
        if self.pdb.structure:
            structure = self.pdb.structure.sub_structure(_start, _end)
        if self.seq:
            seq = self.seq.sub(_start, _end, deep_copy=deep_copy)

        return ChainStructure(self.pdb.id, self.pdb.chain, structure, seq, self, None)

    @classmethod
    def read(
            cls, base_dir: Path,
            dump_names: DumpNames = DumpNames,
    ) -> ChainStructure:
        pdb_id, chain_id = base_dir.name.split(Sep.chain)
        files = get_files(base_dir)
        filenames = {p.name: p for p in files.values()}
        variables = None

        bname = dump_names.structure_base_name
        if bname not in filenames:
            raise InitError(f'{base_dir} must contain {bname}.fmt '
                            f'where "fmt" is supported structure format')
        structure = GenericStructure.read(base_dir / filenames[bname])

        seq = ChainSequence.read(base_dir, dump_names, search_children=False)

        if dump_names.variables in files:
            variables = Variables.read(files[dump_names.variables]).structure

        return ChainStructure(pdb_id, chain_id, structure, seq, variables=variables)

    def write(
            self, base_dir: Path, fmt: str = 'pdb',
            dump_names: DumpNames = DumpNames
    ) -> None:
        if base_dir.name != dump_names.structures_dir:
            base_dir /= dump_names.structures_dir
        base_dir /= f'{self.pdb.id}{Sep.chain}{self.pdb.chain}'
        base_dir.mkdir(exist_ok=True, parents=True)

        self.seq.write(base_dir / dump_names.sequence)
        self.pdb.structure.write(base_dir / f'{dump_names.structure_base_name}.{fmt}')
        if self.variables:
            self.variables.write(base_dir / dump_names.variables)


class Chain(AbstractChain):
    """
    A mutable container, holding data associated with a singe (full) protein chain.
    """

    __slots__ = ('seq', 'structures', 'children')

    def __init__(
            self, seq: ChainSequence,
            structures: t.Optional[t.List[ChainStructure]] = None,
            children: t.Optional[t.Dict[str, Chain]] = None,
    ):
        self.seq = seq
        self.structures = structures or []
        self.children = children or {}

    @property
    def id(self) -> str:
        return self.seq.id

    def __repr__(self) -> str:
        return self.id

    def __str__(self) -> str:
        return self.id

    def __getitem__(self, key) -> Chain:
        if isinstance(key, str):
            return self.children[key]
        if isinstance(key, int):
            return list(self.children.values())[key]
        else:
            raise TypeError('Wrong key type')

    def __contains__(self, item) -> bool:
        return item in self.children

    def __iter__(self) -> abc.Iterator[Chain]:
        yield from self.children.values()

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
        """
        Initialize a :class:`Protein` instance from the existing dump.

        :param path: Path to a dumped protein data.
        :param dump_names: Names of files (keep the default).
        :param search_children: Recursively search child segments and record
            them as :attr:`children`
        :return: Initialized `Protein` instance.
        """
        seq = ChainSequence.read(path, dump_names, search_children=False)

        structures = [ChainStructure.read(p, dump_names)
                      for p in (path / dump_names.structures_dir)]
        protein = Chain(seq, structures)
        if search_children:
            for child_path in (path / dump_names.segments_dir).glob('*'):
                protein.children[child_path.name] = Chain.read(
                    child_path, dump_names, search_children=True)
        return protein

    def write(
            self, base_dir: Path, dump_names: DumpNames = DumpNames,
            *, write_children: bool = True,
    ) -> None:
        """
        Dump protein into directory.

        :param base_dir: Base directory where the directory for the current protein
            shall be created.
        :param dump_names: File and directory names.
        :param write_children: Recursively write children.
        """
        path = base_dir / self.id
        path.mkdir(parents=True, exist_ok=True)

        for name, prot in self.children.items():
            prot.write(path / dump_names.segments_dir)

    def add_structure(
            self, structure: ChainStructure, *, check_ids: bool = True,
            map_to_seq: bool = True, map_name: str = SeqNames.map_canonical,
            **kwargs
    ):
        if check_ids:
            ids = [s.id for s in self.structures]
            if structure.id in ids:
                raise ValueError(f'Protein already contains structure {structure.id}. '
                                 f'Remove it first or disable `check_ids`')
        if map_to_seq:
            structure.seq.map_numbering(self.seq, name=map_name, **kwargs)
        self.structures.append(structure)

    def spawn_child(
            self, start: int, end: int, name: None | str = None, *,
            keep: bool = True, seq_deep_copy: bool = False,
            str_deep_copy: bool = False,
            subset_structures: bool = True,
            map_name: t.Optional[str] = None,
    ) -> Chain:

        def subset_structure(structure: ChainStructure) -> t.Optional[ChainStructure]:
            try:
                return structure.spawn_child(
                    start, end, map_name=map_name, deep_copy=str_deep_copy)
            except (AmbiguousMapping, MissingData) as e:
                LOGGER.warning(
                    f'Failed to spawn substructure using boundaries {start, end} due to {e}')
                return None

        seq = self.seq.spawn_child(start, end, deep_copy=seq_deep_copy)
        if name is None:
            name = seq.name

        structures = (list(filter(bool, map(subset_structure, self.structures)))
                      if subset_structures else None)
        child = Chain(seq, structures)
        if keep:
            self.children[name] = child
        return child


CT = t.TypeVar('CT', bound=Chain)


class ChainList(t.MutableSequence[CT]):

    @t.overload
    def __getitem__(self, index: int) -> CT: ...

    @t.overload
    def __getitem__(self, index: slice | str) -> ChainList[CT]: ...

    def __getitem__(self, index: int | slice | str) -> CT | ChainList[CT]:
        pass


if __name__ == '__main__':
    pass
