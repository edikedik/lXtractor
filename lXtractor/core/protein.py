import logging
import typing as t
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from more_itertools import partition

from lXtractor.core.base import SeqRec, ProteinDumpNames, LengthMismatch, Segment, Variables, Sep, MissingData, AmbiguousMapping
from lXtractor.util.io import Dumper
from lXtractor.util.seq import cut_record
from lXtractor.util.structure import cut_structure

LOGGER = logging.getLogger(__name__)


@dataclass
class Protein:
    """
    A flexible container to accumulate and save data
    related to a single protein chain

    :param pdb: PDB data
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

    def read(self):
        pass

    def write(self, base_dir: Path):
        path = base_dir / self.id
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ValueError(f'Cannot write to {base_dir}')
        dump(path, self.id, uniprot_seq=self.uniprot_seq, structure=self.structure, pdb_seq=self.pdb_seq,
             pdb_seq_raw=self.pdb_seq_raw, metadata=self.metadata, variables=self.variables,
             aln_mapping=self.aln_mapping, uni_pdb_map=self.uni_pdb_map, uni_pdb_aln=self.uni_pdb_aln)

        if self.domains:
            for domain in self.domains.values():

                start, end = domain.pdb_start, domain.pdb_end

                metadata = [
                    ('Name', domain.name),
                    ('UniProt_start', domain.start),
                    ('UniProt_end', domain.end),
                    ('PDB', self.pdb),
                    ('Chain', self.chain),
                    ('PDB_start', start),
                    ('PDB_end', end)
                ]

                log_pref = f'{domain.name}_{domain.start}-{domain.end}'

                dom_path = path / f'{domain.name}'
                dom_path.mkdir(exist_ok=True)
                dump(dom_path, log_pref, uniprot_seq=domain.uniprot_seq, structure=domain.structure,
                     pdb_seq=domain.pdb_seq, pdb_seq_raw=domain.pdb_seq_raw, metadata=domain.metadata + metadata,
                     variables=domain.variables, aln_mapping=domain.aln_mapping, uni_pdb_map=domain.uni_pdb_map,
                     uni_pdb_aln=domain.uni_pdb_aln)

    def extract_domains(self, pdb: bool = True, inplace: bool = True) -> t.List['Domain']:
        """
        For any :class:`lXtractor.base.Domain` the protein contains, extract its
        subsequence from :attr:`lXtractor.protein.Protein.uniprot_seq` and save it to
        :attr:`lXtractor.base.Domain.uniprot_seq`.

        :param pdb: Also extract sub-structures and sub-sequences.
        :param inplace: :func:`deepcopy` domains before populating the data.
        :return: A list of domains with extracted data.
        """
        domains = []

        for domain in self:
            try:
                domain = domain.extract_uniprot(inplace)
            except (MissingData, AmbiguousMapping) as e:
                LOGGER.exception(f'Failed to extract {domain} from {self} due to {e}')
            if pdb:
                try:
                    domain = domain.extract_pdb(inplace)
                except (MissingData, AmbiguousMapping) as e:
                    LOGGER.exception(f'Failed to extract {domain} from {self} due to {e}')
            domains.append(domain)
        return domains


# @dataclass
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
        return f'{super().id}{Sep.dom}{self.name}{Sep.start_end}{self.pdb_start}-{self.pdb_end}'

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.id

    def extract_uniprot(self, inplace: bool = True) -> 'Domain':
        """
        """

        if self.parent is None:
            raise MissingData('Domain requires parent protein to extract data')
        if self.parent.uniprot_seq is None:
            raise MissingData('Domain requires UniProt sequence to extract sequence domain')
        start, end, rec = cut_record(self.parent.uniprot_seq, self)

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


def dump(
        base_path: Path,
        log_prefix: str,
        uniprot_seq: t.Optional[SeqRec] = None,
        structure: t.Optional[Structure] = None,
        pdb_seq: t.Optional[SeqRec] = None,
        pdb_seq_raw: t.Optional[t.Tuple[str, ...]] = None,
        metadata: t.Optional[t.Iterable[t.Tuple[str, t.Any]]] = None,
        variables: t.Optional[t.Dict] = None,
        aln_mapping: t.Optional[t.Dict[int, int]] = None,
        uni_pdb_map: t.Optional[t.Dict[int, int]] = None,
        uni_pdb_aln: t.Optional[t.Tuple[SeqRec, SeqRec]] = None
) -> None:
    # TODO: move into method?
    dumper = Dumper(base_path)
    if uniprot_seq:
        LOGGER.debug(f'{log_prefix} -- dumping UniProt sequence')
        dumper.dump_seq_rec(
            uniprot_seq, ProteinDumpNames.uniprot_seq)
    if structure:
        LOGGER.debug(f'{log_prefix} -- dumping full PDB structure')
        dumper.dump_pdb(
            structure, ProteinDumpNames.pdb_structure)
    if pdb_seq:
        LOGGER.debug(f'{log_prefix} -- dumping full PDB sequence')
        dumper.dump_seq_rec(pdb_seq, ProteinDumpNames.pdb_seq)
    if pdb_seq_raw:
        LOGGER.debug(f'{log_prefix} -- dumping raw PDB sequence')
        dumper.dump_pdb_raw(pdb_seq_raw, ProteinDumpNames.pdb_seq_raw)
    if metadata:
        LOGGER.debug(f'{log_prefix} -- dumping metadata')
        dumper.dump_meta(metadata, ProteinDumpNames.pdb_meta)
    if variables:
        LOGGER.debug(f'{log_prefix} -- dumping variables')
        dumper.dump_variables(variables, ProteinDumpNames.variables)
        maps = filter(
            lambda _item: 'ALL' in _item[0] and _item[1][1] is not None,
            variables.items())
        for item in maps:
            k, (v, val) = item
            agg_name = k.split()[0]
            name = f'{ProteinDumpNames.distance_map_base}_{agg_name}.tsv'
            dumper.dump_distance_map(val, name)
    if aln_mapping:
        LOGGER.debug(f'{log_prefix} -- dumping aln mapping')
        dumper.dump_mapping(aln_mapping, ProteinDumpNames.aln_mapping)
    if uni_pdb_map:
        LOGGER.debug(f'{log_prefix} -- dumping UniProt-PDB SIFTS mapping')
        dumper.dump_mapping(uni_pdb_map, ProteinDumpNames.uni_pdb_map)
    if uni_pdb_aln:
        LOGGER.debug(f'{log_prefix} -- dumping uni-pdb aln')
        dumper.dump_seq_rec(uni_pdb_aln, ProteinDumpNames.uni_pdb_aln)


def _check_dir(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist')
    if not path.is_dir():
        raise NotADirectoryError


def _read_mapping(path: Path) -> t.Dict[int, t.Optional[int]]:
    with path.open() as f:
        lines = filter(
            lambda x: len(x) == 2,
            (x.rstrip().split('\t') for x in f if x != '\n'))
        return {
            int(p1): (int(p2) if p2 != 'None' else None)
            for p1, p2 in lines}


def _read_seqs(
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


def read_protein(base_dir: Path):
    _check_dir(base_dir)
    domain_dirs, _ = partition(
        lambda p: p.is_file(), base_dir.glob('*'))
    domains = {}
    for dom_dir in domain_dirs:
        try:
            domains[dom_dir.name] = read_domain(dom_dir)
            domains[dom_dir.name].parent_name = base_dir.name.split('_')[0]
        except Exception as e:
            LOGGER.error(
                f'Failed to init domain form {dom_dir} due to error {e}')
    parsed_files = _read_files(base_dir)

    uniprot_id, pdb_id, chain_id = None, None, None
    try:
        uniprot_id, pdb = base_dir.name.split('_')
        pdb_id, chain_id = pdb.split(':')
    except ValueError:
        LOGGER.warning(
            f'Failed to parse dir {base_dir}. '
            f'Perhaps it does not follow the standard UniProtID_PDBID:Chain.')

    uniprot_seq = parsed_files[ProteinDumpNames.uniprot_seq]
    if uniprot_id is None and uniprot_seq is not None:
        try:
            uniprot_id = uniprot_seq.id.split('|')[1]
        except IndexError:
            pass

    return Protein(
        _id=base_dir.name,
        pdb=pdb_id, chain=chain_id, uniprot_id=uniprot_id,
        metadata=parsed_files[ProteinDumpNames.pdb_meta],
        uniprot_seq=parsed_files[ProteinDumpNames.uniprot_seq],
        pdb_seq=parsed_files[ProteinDumpNames.pdb_seq],
        pdb_seq_raw=parsed_files[ProteinDumpNames.pdb_seq_raw],
        structure=parsed_files[ProteinDumpNames.pdb_structure],
        variables=parsed_files[ProteinDumpNames.variables],
        aln_mapping=parsed_files[ProteinDumpNames.aln_mapping],
        uni_pdb_map=parsed_files[ProteinDumpNames.uni_pdb_map],
        uni_pdb_aln=parsed_files[ProteinDumpNames.uni_pdb_aln],
        domains=domains)


def read_domain(base_dir: Path) -> Domain:
    parsed_files = _read_files(base_dir)
    uniprot_seq = parsed_files[ProteinDumpNames.uniprot_seq]
    meta = parsed_files[ProteinDumpNames.pdb_meta]
    start, end = None, None
    if uniprot_seq:
        try:
            start, end = map(int, uniprot_seq.id.split('/')[-1].split('-'))
        except (ValueError, IndexError):
            LOGGER.warning(
                f'Failed to find sequence boundaries in the ID'
                f' of the UniProt sequence for domain {base_dir}')
    if meta and any([start is None, end is None]):
        try:
            _meta = dict(meta)
            start = _meta['UniProt_start']
            end = _meta['UniProt_end']
        except (ValueError, KeyError):
            LOGGER.warning(
                f'Failed to find sequence boundaries in the metadata '
                f'for domain {base_dir}')
    if start is None or end is None:
        LOGGER.warning(
            f'Failed to find domain boundaries of domain in {base_dir}')
        start, end = -1, -1
    return Domain(
        name=base_dir.name,
        start=start,
        end=end,
        metadata=meta,
        uniprot_seq=uniprot_seq,
        pdb_seq=parsed_files[ProteinDumpNames.pdb_seq],
        pdb_seq_raw=parsed_files[ProteinDumpNames.pdb_seq_raw],
        structure=parsed_files[ProteinDumpNames.pdb_structure],
        variables=parsed_files[ProteinDumpNames.variables],
        aln_mapping=parsed_files[ProteinDumpNames.aln_mapping],
        uni_pdb_map=parsed_files[ProteinDumpNames.uni_pdb_map],
        uni_pdb_aln=parsed_files[ProteinDumpNames.uni_pdb_aln]
    )


def _read_files(base_dir: Path):
    _check_dir(base_dir)
    _, files = partition(
        lambda p: p.is_file(), base_dir.glob('*'))
    files = {x.name: x for x in files}

    uni_pdb_map, aln_mapping, uniprot_seq, pdb_seq, \
    pdb_seq_raw, structure, uni_pdb_aln = (
        None, None, None, None, None, None, None)
    meta, vs = [], {}

    if ProteinDumpNames.pdb_meta in files:
        path = files[ProteinDumpNames.pdb_meta]
        meta = pd.read_csv(path, sep='\t', header=None)
        if len(meta.columns) != 2:
            LOGGER.error(
                f'Expected two columns in the metadata table {path}, '
                f'but found {len(meta.columns)}')
            meta = None
        else:
            meta = [tuple(x[1:]) for x in meta.itertuples()]

    # TODO: maybe find an easy way to properly init objects from their IDs?
    if ProteinDumpNames.variables in files:
        path = files[ProteinDumpNames.variables]
        vs = pd.read_csv(path, sep='\t')
        if len(vs.columns) != 2:
            LOGGER.error(
                f'Expected two columns in the variables table {path}, '
                f'but found {len(vs.columns)}')
        vs = {
            name: (None, value) for _, name, value in vs.itertuples()}

    if ProteinDumpNames.uni_pdb_map in files:
        path = files[ProteinDumpNames.uni_pdb_map]
        try:
            uni_pdb_map = _read_mapping(path)
        except ValueError as e:
            LOGGER.error(
                f'Failed to read SIFTS mapping {path} '
                f'due to: {e}')

    if ProteinDumpNames.aln_mapping in files:
        path = files[ProteinDumpNames.aln_mapping]
        try:
            aln_mapping = _read_mapping(path)
        except ValueError as e:
            LOGGER.error(
                f'Failed to read alignment mapping {path} '
                f'due to: {e}')

    if ProteinDumpNames.uniprot_seq in files:
        path = files[ProteinDumpNames.uniprot_seq]
        try:
            uniprot_seq = _read_seqs(path)
        except (LengthMismatch, ValueError) as e:
            LOGGER.error(
                f'Failed to read uniprot seq from {path} '
                f'due to: {e}')

    if ProteinDumpNames.pdb_seq in files:
        path = files[ProteinDumpNames.pdb_seq]
        try:
            pdb_seq = _read_seqs(path)
        except (LengthMismatch, ValueError) as e:
            LOGGER.error(
                f'Failed to read pdb seq from {path} '
                f'due to: {e}')
    if ProteinDumpNames.pdb_seq_raw in files:
        path = files[ProteinDumpNames.pdb_seq_raw]
        try:
            with path.open() as f:
                pdb_seq_raw = [
                    line.rstrip() for line in f
                    if line and line != '\n']
        except Exception as e:
            LOGGER.error(
                f'Failed to read pdb raw sequence from {path} '
                f'due to {e}')

    if ProteinDumpNames.pdb_structure in files:
        path = files[ProteinDumpNames.pdb_structure]
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(path.stem, path)
        except Exception as e:
            LOGGER.error(
                f'Failed to read a structure {path} due to {e}')

    if ProteinDumpNames.uni_pdb_aln in files:
        path = files[ProteinDumpNames.uni_pdb_aln]
        try:
            uni_pdb_aln = _read_seqs(path, 2)
        except Exception as e:
            LOGGER.error(
                f'Failed to read UniProt-PDB alignment at {path} '
                f'due to {e}')

    return {
        ProteinDumpNames.pdb_structure: structure,
        ProteinDumpNames.pdb_seq: pdb_seq,
        ProteinDumpNames.uniprot_seq: uniprot_seq,
        ProteinDumpNames.variables: vs,
        ProteinDumpNames.pdb_meta: meta,
        ProteinDumpNames.uni_pdb_map: uni_pdb_map,
        ProteinDumpNames.aln_mapping: aln_mapping,
        ProteinDumpNames.uni_pdb_aln: uni_pdb_aln,
        ProteinDumpNames.pdb_seq_raw: pdb_seq_raw,
    }


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
