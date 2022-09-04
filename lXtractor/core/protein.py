import logging
import typing as t
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from pydoc import locate

import pandas as pd
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from more_itertools import partition, unique_everseen

from lXtractor.core.base import (
    SeqRec, ProteinDumpNames, Segment, Variables,
    Sep, MissingData, AmbiguousMapping, LengthMismatch)
from lXtractor.util.misc import parse_protein_path
from lXtractor.util.seq import cut
from lXtractor.util.structure import cut_structure, dump_pdb

LOGGER = logging.getLogger(__name__)


# TODO: check mapping types list vs dict (list when we allow None)
# TODO: change `metadata` type to dict


@dataclass
class Protein:
    """
    A flexible container to accumulate and save data related to a single protein chain
    """
    pdb: t.Optional[str] = None
    chain: t.Optional[str] = None
    uniprot_id: t.Optional[str] = None
    expected_domains: t.Optional[t.Sequence[str]] = None

    uniprot_seq: t.Optional[SeqRec] = None
    pdb_seq1: t.Optional[SeqRec] = None
    pdb_seq3: t.Optional[t.Tuple[str, ...]] = None

    structure: t.Optional[Structure] = None

    aln_mapping: t.Optional[t.Dict[int, int]] = None
    uni_pdb_map: t.Optional[t.Dict[int, t.Optional[int]]] = None
    uni_pdb_aln: t.Optional[t.Tuple[SeqRec, SeqRec]] = None

    metadata: t.List[t.Tuple[str, t.Any]] = field(default_factory=list)
    variables: t.Optional[Variables] = field(default_factory=Variables)

    # TODO: consider a different data structure?
    # For instance Domain->Domain mapping after making Domain hashable
    domains: t.Dict[str, 'Domain'] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f'{self.uniprot_id}{Sep.uni_pdb}{self.pdb}{Sep.chain}{self.chain}'

    def __repr__(self) -> str:
        return self.id

    def __str__(self) -> str:
        return self.id

    def __hash__(self):
        return hash(self.id)

    def __getitem__(self, key):
        if not self.domains:
            raise ValueError('Empty domains')
        if isinstance(key, str):
            return self.domains[key]
        if isinstance(key, int):
            return list(self.domains.values())[key]
        else:
            raise ValueError('Wrong key type')

    def __iter__(self):
        yield from self.domains.values()

    @classmethod
    def read(
            cls, path: Path,
            dump_names: ProteinDumpNames = ProteinDumpNames
    ) -> 'Protein':
        prot = ProteinIO(path, dump_names).read(Protein)
        prot.domains = {
            p.name: ProteinIO(p, dump_names).read(Domain)
            for p in path.glob('domains/*')}
        return prot

    def write(self, base_dir: Path, dump_names: ProteinDumpNames = ProteinDumpNames):
        path = base_dir / self.id
        path.mkdir(parents=True, exist_ok=True)
        io = ProteinIO(path, dump_names)
        io.write(self)
        LOGGER.debug(f'Written {self.id} to {path}')

    def extract_domains(self, pdb: bool = True, inplace: bool = True) -> t.List['Domain']:
        """
        For any :class:`Domain` the protein contains, extract its subsequence
        from :attr:`uniprot_seq` and save it to :attr:`Domain.uniprot_seq`.

        :param pdb: Also extract sub-structures and sub-sequences.
        :param inplace: :func:`deepcopy` domains before populating the data.
        :return: A list of domains with extracted data.
        """
        domains = []

        for domain in self:
            try:
                domain = domain.extract_uniprot(inplace)
            except (MissingData, AmbiguousMapping) as e:
                LOGGER.exception(
                    f'Failed to extract sequence {domain} from {self} due to {e}')
            if pdb:
                try:
                    domain = domain.extract_pdb(inplace)
                except (MissingData, AmbiguousMapping) as e:
                    LOGGER.exception(
                        f'Failed to extract structure {domain} from {self} due to {e}')
            domains.append(domain)
        return domains

    def spawn_domain(
            self, start: int, end: int, name: str,
            extract_seq: bool = True, extract_pdb: bool = True,
            save: bool = True
    ) -> 'Domain':
        """
        Use existing protein data and given domain boundaries to create new domain
        and (optionally) extract sequence/structure according to boundaries.

        :param start: Domain's start (seq numbering).
        :param end: Domain's end (seq numbering).
        :param name: Domain's name.
        :param extract_seq: Extract UniProt sequence according to boundaries.
            Requires :attr:`uniprot_seq`.
            See :meth:`Domain.extract_uniprot`.
        :param extract_pdb: Extract PDB structure and sequence.
            Requires :attr:`structure`.
            See :meth:`Domain.extract_pdb`.
        :param save: Save the result to :attr:`domains`
        :return: New :class:`Domain` instance.
        """
        # TODO: consider subsetting uni-pdb aln and aln mapping as well?
        dom = Domain(
            start, end, name,
            pdb=self.pdb, chain=self.chain, uniprot_id=self.uniprot_id,
            parent=self, parent_name=self.id, variables=Variables(),
        )
        if extract_seq:
            try:
                dom.extract_uniprot(inplace=True)
            except MissingData as e:
                LOGGER.exception(
                    f'Failed to extract UniProt domain from {dom} due to {e}')
        if extract_pdb:
            try:
                dom.extract_pdb(inplace=True)
            except MissingData as e:
                LOGGER.exception(
                    f'Failed to extract PDB domain from {dom} due to {e}')
        if save:
            self.domains[dom.name] = dom

        return dom


@dataclass
class Domain(Protein, Segment):
    """
    A mutable dataclass container, holding data associated with a protein domain:
    PDB and UniProt sequences, PDB structure (cut according to domain boundaries), etc.
    """
    pdb_start: t.Optional[int] = None
    pdb_end: t.Optional[int] = None
    parent: t.Optional[Protein] = None

    @property
    def id(self):
        # TODO: should I use PDB start/end for id?
        _id = f'{super().id}{Sep.dom}{self.name}'
        if self.start is not None and self.end is not None:
            _id += f'{Sep.start_end}{self.start}-{self.end}'
        return _id

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.id

    @classmethod
    def read(
            cls, path: Path,
            dump_names: ProteinDumpNames = ProteinDumpNames
    ) -> 'Domain':
        return ProteinIO(path, dump_names).read(Domain)

    def extract_uniprot(self, inplace: bool = True) -> 'Domain':
        """
        Extract UniProt sequence from parent's :attr:`Protein.uniprot_seq`.

        :param inplace: Perform the extraction on `self`. Otherwise, create a new object.
        :return: A domain with extracted sequence.
        :raises: :class:`MissingData` if required data is missing.
        """

        if self.parent is None:
            raise MissingData('Domain requires parent protein to extract data')
        if self.parent.uniprot_seq is None:
            raise MissingData('Domain requires UniProt sequence to extract sequence domain')
        start, end, rec = cut(self.parent.uniprot_seq, self)

        if (self.start, self.end) != (start, end):
            LOGGER.warning(
                f'Obtained domain boundaries {(start, end)} differ from '
                f'the existing ones {(self.start, self.end)}')
        obj = self if inplace else deepcopy(self)
        obj.start, obj.end = start, end
        obj.uniprot_seq = rec

        return obj

    def extract_pdb(self, inplace: bool = True) -> 'Domain':
        """
        Extract PDB structure given the domain boundaries.

        Requires:
            - :attr:`parent`
            - :attr:`Protein.structure`
            - :attr:`Protein.uni_pdb_map`

        Populates:
            - :attr:`structure`
            - :attr:`pdb_start`
            - :attr:`pdb_end`
            - :attr:`pdb_seq1`
            - :attr:`pdb_seq3`

        :param inplace: Perform the extraction on `self`. Otherwise, create a new object.
        :return: Domain with extracted data.
        :raises: :class:`MissingData` if required data is missing.
        """
        if self.parent is None:
            raise MissingData('Domain requires parent protein to extract data')
        if self.parent.structure is None:
            raise MissingData(
                f'Require structure to extract structure domain, '
                f'but parent {self.parent} is missing it.')
        if not self.parent.uni_pdb_map:
            raise MissingData(
                f'Mapping between UniProt and PDB numbering is needed '
                f'to extract structure domain, but parent {self.parent} lacks it.')

        cut_res = cut_structure(self.parent.structure, self, self.parent.uni_pdb_map)

        _id = f'{self.pdb}{Sep.chain}{self.chain}{Sep.start_end}{cut_res.start}-{cut_res.end}'

        obj = self if inplace else deepcopy(self)
        obj.structure = cut_res.structure
        obj.pdb_start, obj.pdb_end = cut_res.start, cut_res.end
        obj.pdb_seq1 = SeqRec(Seq(cut_res.seq1), _id, _id, _id)
        obj.pdb_seq3 = cut_res.seq3
        obj.uni_pdb_map = cut_res.mapping

        return obj


_ObjT = t.TypeVar('_ObjT', bound=t.Type[Protein])


class ProteinIO:
    def __init__(
            self, base_dir: Path,
            dump_names: ProteinDumpNames = ProteinDumpNames
    ):
        self.base_dir = base_dir
        self.dump_names = dump_names

    def write_pdb(self, structure: Structure) -> None:
        path = self.base_dir / self.dump_names.pdb_structure
        dump_pdb(structure, path)

    def write_rec(
            self, rec: t.Union[SeqRec, t.Iterable[SeqRec]], name: str
    ) -> None:
        if isinstance(rec, SeqRec):
            rec = [rec]
        path = self.base_dir / name
        SeqIO.write(rec, path, 'fasta')

    def write_meta(self, meta: t.Iterable[t.Tuple[str, t.Any]]) -> None:
        path = self.base_dir / self.dump_names.meta
        records = unique_everseen(meta)
        with path.open('w') as f:
            for name, value in records:
                print(name, value, sep='\t', file=f)

    def write_variables(
            self, variables: Variables,
            skip_if_contains: t.Collection[str] = ('ALL',)
    ) -> None:
        _path = self.base_dir / self.dump_names.variables
        items = (f'{v.id}\t{r}' for v, r in variables.items()
                 if all(x not in v.id for x in skip_if_contains))
        _path.write_text('\n'.join(items))
        LOGGER.debug(f'Saved {len(variables)} variables to {_path}')

    def write_pdist(self, distances: t.Iterable[t.Tuple[int, int, float]], path) -> None:
        # path = self.base_dir / self.dump_names.pdist_base_name
        with path.open('w') as f:
            for pos1, pos2, dist in distances:
                print(pos1, pos2, dist, sep='\t', file=f)

    def write_mapping(self, mapping: t.Dict[t.Any, t.Any], name: str) -> None:
        path = self.base_dir / name
        with path.open('w') as f:
            for k, v in mapping.items():
                print(k, v, sep='\t', file=f)

    def write_pdb_seq3(self, seq: t.Tuple[str, ...]) -> None:
        path = self.base_dir / self.dump_names.pdb_seq3
        with path.open('w') as f:
            print(*seq, sep='\n', file=f)

    def _write_obj(self, obj: t.Union[Protein, Domain]):
        if obj.uniprot_seq is not None:
            self.write_rec(obj.uniprot_seq, self.dump_names.uniprot_seq)
        if obj.pdb_seq1 is not None:
            self.write_rec(obj.pdb_seq1, self.dump_names.pdb_seq1)
        if obj.uni_pdb_aln:
            self.write_rec(obj.uni_pdb_aln, self.dump_names.uni_pdb_aln)
        if obj.pdb_seq3 is not None:
            self.write_pdb_seq3(obj.pdb_seq3)
        if obj.structure is not None:
            self.write_pdb(obj.structure)
        if obj.metadata is not None:
            meta = dict(obj.metadata)
            meta_base = [
                ('UniProt_ID', obj.uniprot_id),
                ('PDB', obj.pdb),
                ('Chain', obj.chain),
            ]
            if isinstance(obj, Domain):
                meta_base += [
                    ('Domain', obj.name),
                    ('UniProt_start', obj.start),
                    ('UniProt_end', obj.end),
                    ('PDB_start', obj.pdb_start),
                    ('PDB_end', obj.pdb_end)
                ]
            meta.update(dict(meta_base))
            self.write_meta(list(meta.items()))
        if obj.aln_mapping:
            self.write_mapping(obj.aln_mapping, self.dump_names.aln_mapping)
        if obj.uni_pdb_map:
            self.write_mapping(obj.uni_pdb_map, self.dump_names.uni_pdb_map)
        if obj.variables is not None:
            self.write_variables(obj.variables)
            pdist_maps = filter(lambda _v: 'ALL' in _v.id, obj.variables)
            for pdist in pdist_maps:
                agg_name = pdist.split()
                path = (self.base_dir /
                        self.dump_names.pdist_base_dir /
                        f'{self.dump_names.pdist_base_name}_{agg_name}.tsv')
                self.write_pdist(obj.variables[pdist], path)

    def write(self, obj: t.Union[Protein, Domain]):
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self._write_obj(obj)
        LOGGER.debug(f'Saved {obj} to {self.base_dir}')
        if isinstance(obj, Protein):
            for k, v in obj.domains.items():
                base_dir = self.base_dir / self.dump_names.domains_dir / v.name
                base_dir.mkdir(exist_ok=True, parents=True)
                io = self.__class__(base_dir)
                io._write_obj(v)
                LOGGER.debug(f'Saved {k} to {base_dir}')

    @staticmethod
    def _read_n_col_table(path: Path, n: int) -> t.Optional[pd.DataFrame]:
        df = pd.read_csv(path, sep='\t', header=None)
        if len(df.columns) != n:
            LOGGER.error(
                f'Expected two columns in the table {path}, '
                f'but found {len(df.columns)}')
            return None
        return df

    @staticmethod
    def _infer_boundaries(
            obj: t.Union[Protein, Domain], pdb: bool = False
    ) -> t.Tuple[int, int]:
        start, end = None, None
        bound_type = 'PDB' if pdb else 'UniProt'
        start_name, end_name = f'{bound_type}_start', f'{bound_type}_end'
        seq = obj.pdb_seq1 if pdb else obj.uniprot_seq
        if seq is not None:
            try:
                start, end = map(int, seq.id.split('/')[-1].split('-'))
            except (ValueError, IndexError):
                LOGGER.warning(
                    f'Failed to find sequence boundaries in the ID {seq.id}'
                    f' of the UniProt sequence for {obj}')
        if obj.metadata is not None and any([start is None, end is None]):
            try:
                meta = dict(obj.metadata)
                start = meta[start_name]
                end = meta[end_name]
            except (ValueError, KeyError):
                LOGGER.warning(
                    f'Failed to find sequence boundaries in the metadata of {obj}')
        if start is None or end is None:
            LOGGER.warning(f'Failed to find {bound_type} domain boundaries of {obj}')
            start, end = -1, -1
        return start, end

    @staticmethod
    def _infer_ids(
            obj: t.Union[Protein, Domain], path: t.Optional[Path], parent: t.Optional[Protein],
    ) -> t.Tuple[t.Optional[str], t.Optional[str], t.Optional[str]]:
        if obj.metadata is not None:
            try:
                meta = dict(obj.metadata)
                return meta['UniProt_ID'], meta['PDB'], meta['Chain']
            except KeyError:
                LOGGER.warning(f'Failed to infer IDs from existing metadata of {obj}')
        if path is not None:
            return parse_protein_path(path)
        if parent is not None:
            return parent.uniprot_id, parent.pdb, parent.chain
        LOGGER.warning(f'Failed to infer IDs for {obj}')
        return None, None, None

    def read_variables(self, path: Path) -> Variables:
        # TODO: read pdist
        assert path.exists()

        try:
            vs = self._read_n_col_table(path, 2)
        except pd.errors.EmptyDataError:
            vs = pd.DataFrame()
        variables = Variables()

        for v_id, v_val in vs.itertuples(index=False):
            v_name = v_id.split('(')[0]
            import_statement = f'from lXtractor.core.variables import {v_name}'
            try:
                exec(import_statement)
            except ImportError:
                LOGGER.exception(
                    f'Failed to exec {import_statement} for variable {v_name} '
                    f'causing variable\'s init to fail')
                continue
            try:
                v = eval(v_id)
            except Exception as e:
                LOGGER.exception(f'Failed to eval variable {v_id} due to {e}')
                continue
            try:
                v_val = eval(v_val)
            except Exception as e:
                LOGGER.debug(f'Failed to eval {v_val} for variable {v_name} due to {e}')
            variables[v] = v_val

        return variables

    def read_mapping(
            self, path: Path, as_dict: bool = True
    ) -> t.Union[
        t.Dict[int, t.Optional[int]],
        t.List[t.Tuple[t.Optional[int], t.Optional[int]]]
    ]:
        df = self._read_n_col_table(path, 2)
        mapping = [(x1, x2) for x1, x2 in df.itertuples(index=False)]
        if as_dict:
            mapping = dict(mapping)
            if None in mapping:
                LOGGER.warning(
                    f'Mapping {path} converted to dict type but has `None` within the keys, '
                    f'so maybe converting was a mistake')
        return mapping

    @staticmethod
    def read_seqs(
            path: Path, num_expected: int = 1, fmt='fasta',
    ) -> t.Union[SeqRec, t.List[SeqRec]]:
        seqs = list(SeqIO.parse(path, fmt))
        if len(seqs) != num_expected:
            raise LengthMismatch(
                f'Expected to find {num_expected} seqs in {path} '
                f'but found {len(seqs)}')
        if num_expected == 1:
            return seqs.pop()
        return seqs

    def read(
            self,
            obj_type: t.Type[_ObjT],
            obj: t.Optional[_ObjT] = None,
            parent: t.Optional[Protein] = None,
    ) -> _ObjT:

        if obj is None:
            if obj_type is Protein:
                obj = obj_type()
            else:
                obj = obj_type(start=-1, end=-1, name='')

        _, files = partition(lambda p: p.is_file(), self.base_dir.glob('*'))
        files = {x.name: x for x in files}

        if self.dump_names.meta in files:
            meta = self._read_n_col_table(files[self.dump_names.meta], 2)
            if meta is not None:
                obj.metadata = [(x1, x2) for x1, x2 in meta.itertuples(index=False)]
        if self.dump_names.variables in files:
            obj.variables = self.read_variables(files[self.dump_names.variables])
        if self.dump_names.uni_pdb_map in files:
            obj.uni_pdb_map = self.read_mapping(files[self.dump_names.uni_pdb_map])
        if self.dump_names.aln_mapping in files:
            obj.aln_mapping = self.read_mapping(files[self.dump_names.aln_mapping])
        if self.dump_names.uniprot_seq in files:
            obj.uniprot_seq = self.read_seqs(files[self.dump_names.uniprot_seq], 1)
        if self.dump_names.pdb_seq1 in files:
            obj.pdb_seq1 = self.read_seqs(files[self.dump_names.pdb_seq1], 1)
        if self.dump_names.uni_pdb_aln in files:
            obj.uni_pdb_aln = self.read_seqs(files[self.dump_names.uni_pdb_aln], 2)
        if self.dump_names.pdb_seq3 in files:
            obj.pdb_seq3 = self._read_n_col_table(
                files[self.dump_names.pdb_seq3], 1).iloc[:, 0].tolist()
        if self.dump_names.pdb_structure in files:
            path = files[self.dump_names.pdb_structure]
            obj.structure = PDBParser(QUIET=True).get_structure(path.stem, path)

        obj_path = self.base_dir
        if isinstance(obj, Domain):
            obj.start, obj.end = self._infer_boundaries(obj, pdb=False)
            obj.pdb_start, obj.pdb_end = self._infer_boundaries(obj, pdb=True)
            try:
                obj.name = dict(obj.metadata)['Domain']
            except KeyError:
                obj.name = self.base_dir
            obj_path = self.base_dir.parent.parent

        obj.uniprot_id, obj.pdb, obj.chain = self._infer_ids(obj, obj_path, parent)

        return obj


def unduplicate(
        proteins: t.Collection[Protein],
        domain_name: t.Optional[str] = None,
        always_keep: t.Optional[t.Container[str]] = None,
) -> t.Iterator[Protein]:
    seen = set()
    for _p in proteins:
        if always_keep and _p.id in always_keep:
            yield _p
        if domain_name:
            if domain_name not in _p.domains:
                yield _p
            seq = _p.domains[domain_name].uniprot_seq
        else:
            seq = _p.uniprot_seq
        if seq is None:
            yield _p
        else:
            seq = str(seq.seq)
            if seq not in seen:
                seen.add(seq)
                yield _p
            else:
                pass


if __name__ == '__main__':
    pass
