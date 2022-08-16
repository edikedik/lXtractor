import logging
import typing as t
from copy import deepcopy, copy
from itertools import starmap, product, chain, tee, filterfalse
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from Bio.PDB.Structure import Structure
from more_itertools import flatten, peekable, ilen
from toolz import curry, pipe
from toolz.curried import filter, map, groupby
from tqdm.auto import tqdm

from lXtractor.alignment import Alignment, mafft_align, map_pairs_numbering, _Align_method
from lXtractor.base import (
    SeqRec, FormatError, MissingData, AbstractVariable,
    FailedCalculation, Seq, Domain, AminoAcidDict, Sep)
from lXtractor.cutters import extract_pdb_domains
from lXtractor.input_parser import init
from lXtractor.pdb import PDB, get_sequence, wrap_raw_pdb
from lXtractor.protein import Protein
from lXtractor.sifts import SIFTS
from lXtractor.uniprot import UniProt
from lXtractor.utils import split_validate, run_handles
from lXtractor.variables import _ParsedVariables, parse_variables

_DomainList = t.Sequence[t.Tuple[str, t.Union[str, t.Sequence[str]]]]
_VariableSetup = t.Tuple[AbstractVariable, t.Optional[str], t.Optional[str]]
_ChainLevel = 'Chain'
_KeyT = t.Union[int, str, slice, t.Sequence[bool], np.ndarray]
_AlnT = t.Union[t.List[SeqRec], t.List[str], Alignment]
LOGGER = logging.getLogger(__name__)

# TODO: extend docs
# TODO: optimize memory consumption for large datasets
# TODO: -> use common (sequence/UniProt), including sequences
# TODO: range variable
# TODO: test init with various possible input types


T = t.TypeVar('T')


class lXtractor:
    def __init__(
            self, inputs: t.Optional[t.Sequence[str]] = None,
            proteins: t.Optional[t.List[Protein]] = None,
            expected_domains: t.Optional[t.Union[t.Sequence[str]]] = None,
            uniprot: t.Optional[UniProt] = None,
            pdb: t.Optional[PDB] = None,
            sifts: t.Optional[SIFTS] = None,
            sifts_id_mapping: bool = True,
            sifts_segment_mapping: bool = False,
            alignment: t.Optional[Alignment] = None,
            num_threads: t.Optional[int] = None
    ):

        if inputs is None and proteins is None:
            raise ValueError('Must provide either ``inputs`` or ``proteins``')

        self.alignment = alignment
        self.pdb = pdb or PDB()
        self.uniprot = uniprot or UniProt()

        if isinstance(expected_domains, str):
            expected_domains = [expected_domains]
        self.expected_domains = expected_domains

        self.sifts = sifts
        if sifts_id_mapping and self.sifts is None:
            self.sifts = SIFTS(load_segments=sifts_segment_mapping,
                               load_id_mapping=sifts_id_mapping)
            if sifts_segment_mapping and self.sifts.df is None:
                self.sifts.parse()

        if proteins is None:

            if expected_domains:
                expected_domains = ','.join(expected_domains)
                inputs = [
                    f'{x}{Sep.dom}{expected_domains}' if Sep.dom not in x else x
                    for x in inputs]

            processed_inputs = []
            for inp in tqdm(inputs, desc='Processing inputs'):
                try:
                    processed_inputs.append(list(init(inp, self.sifts)))
                except Exception as e:
                    LOGGER.exception(f'Failed to init input {inp} due to {e}')

            self.proteins = list(flatten(processed_inputs))
        else:
            self.proteins = proteins
        LOGGER.info(
            f'Processed input into {len(self.proteins)} proteins')

        self.num_threads = num_threads
        if num_threads is not None:
            if not ray.is_initialized():
                LOGGER.info(f'Initializing ray with {num_threads} threads')
                ray.init(num_cpus=num_threads)
            else:
                LOGGER.info('Omitting ray init as it is already running')

    def __len__(self) -> int:
        return len(self.proteins)

    def __repr__(self) -> str:
        return f'lXtractor(proteins={len(self)})'

    def __getitem__(self, key: _KeyT) -> t.Optional[t.Union[Protein, t.List[Protein]]]:
        def get_one_or_more(xs):
            if len(xs) > 1:
                return xs
            if  len(xs) == 1:
                return xs.pop()
            return None

        # INT or SLICE
        if isinstance(key, int) or isinstance(key, slice):
            return self.proteins[key]

        # Sequence or boolean
        if not isinstance(key, str) and (isinstance(key, t.Sequence) or isinstance(key, np.ndarray)):
            if len(key) != len(self.proteins):
                raise IndexError(
                    f'For boolean indexing, length of the idx ({len(key)}) '
                    f'must match the number of proteins ({len(self.proteins)}')
            return [x for x, b in zip(self.proteins, key) if b]

        # Full ID
        if '_' in key:
            ps = list(filter(lambda p: p.id == key, self.proteins))

        # PDB_ID:CHAIN
        elif ':' in key:
            ps = list(filter(
                lambda p: f'{p.pdb}:{p.chain}' == key,
                self.proteins))

        # PDB_ID
        elif len(key) == 4:
            ps = list(
                filter(lambda p: p.pdb == key, self.proteins))

        # UniProt ID
        else:
            ps = list(
                filter(lambda p: p.uniprot_id == key, self.proteins))

        return get_one_or_more(ps)

    def __copy__(self, **kwargs) -> "lXtractor":
        """
        """
        ps = [copy(p) for p in self.proteins]
        return self.__class__(
            proteins=ps, expected_domains=self.expected_domains, uniprot=self.uniprot, pdb=self.pdb,
            sifts=self.sifts, alignment=self.alignment)

    def __deepcopy__(self, *args, **kwargs) -> "lXtractor":
        ps = [deepcopy(p, *args, **kwargs) for p in self.proteins]
        return self.__class__(
            proteins=ps, expected_domains=self.expected_domains, uniprot=self.uniprot, pdb=self.pdb,
            sifts=self.sifts, alignment=self.alignment)

    def __iter__(self):
        return iter(self.proteins)

    @property
    def domains(self) -> t.List[Domain]:
        return list(chain.from_iterable(p.domains.values() for p in self))

    def subset(self, key: _KeyT, deep: bool = False) -> "lXtractor":
        ps = self.__getitem__(key)
        if isinstance(ps, Protein):
            ps = [ps]
        tmp = self.proteins
        self.proteins = ps
        new_lxt = deepcopy(self) if deep else copy(self)
        self.proteins = tmp
        return new_lxt

    def fetch_uniprot(
            self, missing: bool = True, keep_expected: bool = True,
            uniprot: t.Optional[UniProt] = None, **kwargs
    ) -> t.Optional[pd.DataFrame]:
        expected_domains = self.expected_domains
        if keep_expected and not expected_domains:
            LOGGER.warning('`keep_expected` is true but `domains` attribute is empty')
        if uniprot is None:
            if self.uniprot is None:
                raise ValueError('No UniProt instance initialized')
            uniprot = self.uniprot
        proteins = list(filter(
            lambda p: p.uniprot_id is not None and (
                    not missing or p.uniprot_seq is None or not p.domains),
            self.proteins))

        LOGGER.info(f'Found {len(proteins)} to fetch UniProt data')
        df = None
        if proteins:
            _, df = uniprot.fetch_proteins_data(
                proteins, keep_expected_if_any=keep_expected, **kwargs)
        return df

    def fetch_pdb(self, missing: bool = True) -> None:
        if self.pdb is None:
            raise ValueError('No PDB instance initialized')
        proteins = list(filter(
            lambda p: p.pdb is not None and (not missing or p.structure is None),
            self.proteins))
        LOGGER.info(f'Found {len(proteins)} to fetch PDB data')
        if proteins:
            self.pdb.fetch(proteins)

    def map_uni_pdb(
            self, missing: bool = True,
            alignment_based: bool = False,
            length_cap: t.Optional[int] = None):

        if alignment_based:
            proteins = [
                p for p in self.proteins if
                (not p.uni_pdb_map or missing) and
                p.structure and
                (p.uniprot_seq or p.uni_pdb_aln) and
                (length_cap is None or len(p.uniprot_seq) < length_cap)
            ]
        else:
            proteins = [
                p for p in self.proteins if
                (not p.uni_pdb_map or missing) and
                p.pdb and p.chain and p.uniprot_id
            ]

        LOGGER.info(f'Found {len(proteins)} proteins to map numbering')

        if proteins:

            if alignment_based:
                method = 'MSA'
                self.map_numbering_aln(proteins)
            else:
                method = 'SIFTS'
                self.map_numbering_sifts(proteins)

            LOGGER.info(f'Used {method}-based mapping on {len(proteins)} proteins')

    def map_numbering_sifts(self, proteins: t.Optional[t.Sequence[Protein]] = None):
        proteins = proteins or self.proteins

        for p in tqdm(
                proteins, desc='Mapping SIFTS numbering', position=0, leave=True):
            try:
                p.uni_pdb_map = map_whole_structure_numbering(p, self.sifts)
            except ValueError as e:
                LOGGER.warning(
                    f"Failed to map whole chain sequence for protein "
                    f"{p.id} due to error {e}")

    def map_numbering_aln(self, proteins: t.Optional[t.Sequence[Protein]] = None):
        proteins = proteins or self.proteins
        inputs = [(p.id, p.structure, p.uni_pdb_aln, p.uniprot_seq, p.pdb_seq)
                  for p in proteins]
        handles = [map_by_alignment_remote.remote(*inp) for inp in inputs]
        bar = tqdm(desc='Mapping Uni-PDB by alignment', total=len(inputs),
                   position=0, leave=True)
        results = run_handles(handles, bar)
        proteins = {p.id: p for p in proteins}
        for p_id, uni_pdb_aln, mapping in results:
            proteins[p_id].uni_pdb_aln = uni_pdb_aln
            if isinstance(mapping, Exception):
                LOGGER.error(f'Failed to acquire mapping for {p_id} '
                             f'due to error {mapping}')
                continue
            proteins[p_id].uni_pdb_map = mapping

    def align(self, domain: t.Optional[t.Union[str, t.Container[str]]] = None,
              pdb: bool = True, method: _Align_method = mafft_align) -> Alignment:

        def accept(obj: t.Union[Protein, Domain]):
            return ((domain and isinstance(obj, Domain) and obj.name in domain)
                    or not domain)

        if domain is not None:
            if isinstance(domain, str):
                domain = [domain]
            obj_type = 'domain'
            objs = self.domains
        else:
            obj_type = 'protein'
            objs = self.proteins

        seqs = pipe(
            objs,
            filter(accept),
            map(lambda x: x.pdb_seq if pdb else x.uniprot_seq),
            filter(lambda x: x is not None),
            list)

        LOGGER.debug(f'Found {len(seqs)} {obj_type} sequences to align')

        if not seqs:
            msg = 'No suitable sequences to align'
            LOGGER.error(msg)
            raise MissingData(msg)

        seqs = method(seqs)

        LOGGER.info(f'Aligned {len(seqs)} {obj_type} sequences')

        return Alignment(seqs=seqs)

    def map_numbering_to_msa(
            self, alignment: _AlnT, domains: t.Optional[t.Union[t.Sequence[str], str]] = None,
            pdb_seq: bool = True, parallel: bool = True, missing: bool = True) -> None:

        def get_seq(obj: t.Union[Protein, Domain]) -> str:
            if pdb_seq:
                if obj.pdb_seq is None:
                    return get_sequence(obj.structure)
                return str(obj.pdb_seq.seq)
            return str(obj.uniprot_seq.seq)

        def get_num(obj: t.Union[Protein, Domain]) -> t.Tuple[int, ...]:
            if pdb_seq:
                return get_sequence(obj.structure, numbering=True)
            start = obj.start if isinstance(obj, Domain) else 1
            return tuple(i for i, x in enumerate(obj.uniprot_seq, start=start))

        def accept(obj: t.Union[Protein, Domain]):

            acc_exist = (missing and obj.aln_mapping is None) or not missing
            acc_dom = (domains is not None and obj.name in domains) or domains is None
            acc_seq = ((pdb_seq and obj.structure is not None) or
                       (not pdb_seq and obj.uniprot_seq is not None))

            return all([acc_exist, acc_dom, acc_seq])

        def prep_aln(aln: _AlnT) -> Alignment:
            if isinstance(aln, Alignment):
                return aln
            elif isinstance(aln, t.Sequence):
                if not aln:
                    raise MissingData('No sequences in the provided aln')
                if isinstance(aln[0], str):
                    seqs = [SeqRec(Seq(s), i, i, i) for i, s in enumerate(aln)]
                elif isinstance(aln[0], SeqRec):
                    seqs = aln
                else:
                    raise ValueError('Unsupported sequence type')
                return Alignment(seqs=seqs)
            else:
                raise ValueError('Unsupported aln type')

        alignment = prep_aln(alignment)
        domains = [domains] if isinstance(domains, str) else domains

        # Group objects by actual sequence to optimize mapping
        groups = pipe(
            self.domains if domains else self.proteins,
            filter(accept),
            map(lambda x: (x, get_seq(x), get_num(x))),
            groupby(lambda x: (x[1], x[2]))
        )
        desc, total, total_objs = (
            f'Mapping to an MSA {alignment}', len(groups), sum(map(len, groups.values())))
        LOGGER.info(f'Found {total} unique sequence (from {total_objs} objects) for MSA mapping')

        if parallel:
            aln_remote = ray.put(alignment)
            fn = ray.remote(map_to_alignment)
            bar = tqdm(position=0, leave=True, desc=desc, total=total)
            handles = [fn.remote(seq, num, aln_remote) for seq, num in groups]
            results = run_handles(handles, bar)
        else:
            bar = tqdm(groups, position=0, leave=True, desc=desc, total=total)
            results = [map_to_alignment(seq, num, alignment) for seq, num in bar]

        for seq, numbering, mapping in results:
            objs = [x for x, _, _ in groups[(seq, numbering)]]
            if not isinstance(mapping, t.Dict):
                ids = [x.id for x in objs]
                LOGGER.exception(f'Failed to map sequence for {ids} due to error {mapping}')
                continue
            for x in objs:
                x.aln_mapping = mapping
                LOGGER.debug(f'Assigned new aln mapping to {x.id}')

    def extract_structure_domains(self, parallel: bool = False) -> None:

        proteins = {
            p.id: p for p in self.proteins if
            (p.structure and p.domains and p.uni_pdb_map)}

        LOGGER.info(f'Found {len(proteins)} proteins for '
                    f'structure domains extraction.')
        if parallel:
            extract_structure_domains_remote = ray.remote(extract_structure_domains)
            bar = tqdm(
                desc='Extracting structure domains',
                total=len(proteins), position=0, leave=True)
            handles = [extract_structure_domains_remote.remote(p)
                       for p in proteins.values()]
            results = run_handles(handles, bar)
        else:
            results = list(map(
                extract_structure_domains,
                tqdm(proteins.values(), desc='Extracting structure domains',
                     total=len(proteins), position=0, leave=True)))
        for r in results:
            if not isinstance(r, Protein):
                p_id, exc = r
                LOGGER.warning(f'Failed to extract domains from {p_id} '
                               f'due to error {exc}')
            else:
                proteins[r.id].domains = r.domains

    def compute_seq_meta(self, exclude_mod: t.Tuple[str, ...] = ('HOH', )):
        proteins = [
            p for p in self.proteins if (p.structure and p.pdb_seq and p.uniprot_seq)]
        if proteins:
            LOGGER.info(f'Found {len(proteins)} proteins to calculate seq metadata')
            inputs = [
                (p.id, p.uniprot_seq, p.pdb_seq, p.structure, p.uni_pdb_aln, exclude_mod)
                for p in proteins]
            handles = [seq_stats_remote.remote(*inp) for inp in inputs]
            bar = tqdm(
                desc='Calculating sequence metadata (protein level)',
                total=len(inputs), position=0, leave=True)
            results = run_handles(handles, bar)
            proteins_map = {p.id: p for p in proteins}
            for obj_id, aln, meta in results:
                proteins_map[obj_id].metadata += meta
                proteins_map[obj_id].uni_pdb_aln = aln

        domains = dict(chain.from_iterable(
            ((f'{p.id}_{name}', dom) for name, dom in p.domains.items()
             if (dom.structure and dom.pdb_seq and dom.uniprot_seq))
            for p in self.proteins))
        if domains:
            LOGGER.info(
                f'Found {len(domains)} domains to calculate seq metadata')
            inputs = [
                (dom_id, dom.uniprot_seq, dom.pdb_seq,
                 dom.structure, dom.uni_pdb_aln, exclude_mod)
                for dom_id, dom in domains.items()]
            handles = [seq_stats_remote.remote(*inp) for inp in inputs]
            bar = tqdm(
                desc='Calculating sequence metadata (domain level)',
                total=len(inputs), position=0, leave=True)
            results = run_handles(handles, bar)
            for obj_id, aln, meta in results:
                domains[obj_id].metadata += meta
                domains[obj_id].uni_pdb_aln = aln

    def assign_variables(self, variables: t.Iterable[str]):
        def try_parse(v: str) -> t.Optional[_ParsedVariables]:
            try:
                parsed_var = parse_variables(v)
                LOGGER.debug(f'Successfully parsed variable {v}')
                return parsed_var
            except (ValueError, FormatError) as e:
                LOGGER.error(
                    f'Failed to parse variable {v} due to error {e}')
                return None

        variables = list(filter(bool, map(
            try_parse, tqdm(variables, 'Processing variables'))))

        assign_variables(variables, self.proteins)
        LOGGER.debug(f'Successfully assigned variables')

    def calculate_variables(self) -> None:

        def get_variables(obj: t.Union[Protein, Domain]):
            return [x[0] for x in obj.variables.values()]

        proteins = [p for p in self.proteins if all(
            [p.variables, p.aln_mapping, p.structure])]

        LOGGER.info(f'Found {len(proteins)} full protein chains with '
                    f'variables staged for calculation')
        if proteins:
            inputs = [
                (p.id, p.structure, get_variables(p), p.aln_mapping) for p in proteins]
            bar = tqdm(
                desc='Calculating variables on protein level',
                total=len(inputs), position=0, leave=True)
            handles = [calculate_variables_remote.remote(*inp) for inp in inputs]
            results = run_handles(handles, bar)
            for obj_id, success, var, value in flatten(results):
                if success:
                    self.__getitem__(obj_id).variables[var.id] = (var, value)
                    LOGGER.debug(f'Calculated {var.id} for {obj_id}')
                else:
                    LOGGER.warning(
                        f'Failed to calculate {var.id} for {obj_id} '
                        f'due to error {value}')

        domains = flatten(
            ((p.id, k, d) for k, d in p.domains.items()) for p in self.proteins)
        domains = list(filter(
            lambda x: all([x[-1].variables, x[-1].structure, x[-1].aln_mapping]),
            domains))
        LOGGER.info(f'Found {len(domains)} domains with variables staged for calculation')
        if domains:
            inputs = [(f'{p_id}---{k}', d.structure, get_variables(d), d.aln_mapping)
                      for p_id, k, d in domains]
            bar = tqdm(
                desc='Calculating variables on a domain level',
                total=len(inputs), position=0, leave=True)
            handles = [calculate_variables_remote.remote(*inp) for inp in inputs]
            results = run_handles(handles, bar)
            for obj_id, success, var, value in flatten(results):
                p_id, d_id = obj_id.split('---')
                if success:
                    self.__getitem__(p_id).domains[d_id].variables[var.id] = (var, value)
                    LOGGER.debug(f'Calculated {var.id}={value} for domain {d_id} of {p_id}')
                else:
                    LOGGER.warning(
                        f'Failed to calculate {var.id} for domain {d_id} of {obj_id} '
                        f'due to error {value}')

    def dump(self):
        raise NotImplementedError


@ray.remote
def parse_structure_remote(
        path: Path
) -> t.Union[t.Tuple[Structure, t.List[t.Tuple[str, t.Any]]], t.Tuple[str, Exception]]:
    try:
        _, structure, meta = wrap_raw_pdb(path.read_text())
        return structure, meta
    except Exception as e:
        return path.stem, e


@ray.remote
def seq_stats_remote(
        obj_id: str, uni_seq: SeqRec,
        pdb_seq: SeqRec, structure: Structure,
        uni_pdb_aln: t.Optional[t.Tuple[SeqRec, SeqRec]],
        exclude: t.Tuple[str, ...] = ('HOH',),
) -> t.Tuple[str, t.Tuple[SeqRec, SeqRec], t.List[t.Tuple[str, t.Union[str, float]]]]:
    def coverage(seq1: SeqRec, seq2: SeqRec) -> float:
        try:
            sub = tee((c1, c2) for c1, c2 in zip(seq1, seq2) if c1 != '-')
            return ilen(c for _, c in sub[0] if c != '-') / ilen(sub[1])
        except Exception as e:
            LOGGER.warning(
                f'Failed to calculate sequence {seq1.id}({obj_id}) '
                f'coverage due to {e}')
            return 0

    aa_dict = AminoAcidDict()
    mods = [
        f'{r.resname}_{r.id[1]}' for r in structure.get_residues()
        if r.resname not in aa_dict and r.resname not in exclude]

    if uni_pdb_aln:
        uni_aln, pdb_aln = uni_pdb_aln
    else:
        uni_aln, pdb_aln = mafft_align(
            [uni_seq, pdb_seq], thread=1)
    num_gaps = sum(
        1 for x, y in zip(uni_aln, pdb_aln)
        if x == '-' or y == '-')
    num_miss = sum(
        1 for x, y in zip(uni_aln, pdb_aln)
        if x != '-' and y != '-' and x != y)
    return obj_id, (uni_aln, pdb_aln), [
        ('PDB non-canonical', ';'.join(mods)),
        ('PDB num non-canonical', len(mods)),
        ('Uni seq coverage', round(coverage(uni_aln, pdb_aln), 2)),
        ('PDB seq coverage', round(coverage(pdb_aln, uni_aln), 2)),
        ('Uni-PDB num gaps', num_gaps),
        ('Uni-PDB num mismatches', num_miss),
    ]


# @ray.remote
def map_to_alignment(
        seq: str,
        numbering: t.Sequence[int],
        alignment: Alignment
) -> t.Tuple[str, t.Sequence[int], t.Union[t.Dict[int, int], Exception]]:
    try:
        return seq, numbering, alignment.map_seq_numbering(
            SeqRec(Seq(seq)), numbering)
    except Exception as e:
        return seq, numbering, e


@ray.remote
def map_by_alignment_remote(
        obj_id: str,
        structure: Structure,
        uni_pdb_aln: t.Optional[t.Tuple[SeqRec, SeqRec]] = None,
        uni_seq: t.Optional[SeqRec] = None,
        pdb_seq: t.Optional[SeqRec] = None,
) -> t.Tuple[str, t.Tuple[SeqRec, SeqRec], t.Union[t.Dict[int, t.Optional[int]], Exception]]:
    if uni_seq is None and uni_pdb_aln is None:
        raise ValueError(
            f'Wrong input for {obj_id}. Must provide a UniProt sequence: '
            f'either in the `uni_pdb_aln` or via `uni_seq`')
    if uni_pdb_aln is None:
        if pdb_seq is None:
            pdb_seq = SeqRec(
                # Do not filter out anything so that the numbering's length
                # always matches the sequence's length
                Seq(get_sequence(structure, convert=True, filter_out=())),
                id=structure.id, name=structure.id,
                description=structure.id)
        uni_pdb_aln = mafft_align([uni_seq, pdb_seq])

    uni_aln, pdb_aln = uni_pdb_aln

    uni_num = list(range(1, len(uni_seq) + 1))
    pdb_num = [r.id[1] for r in structure.get_residues()]
    try:
        raw_mapping = map_pairs_numbering(
            uni_aln, uni_num, pdb_aln, pdb_num, align=False)
    except Exception as e:
        return obj_id, uni_pdb_aln, e

    mapping = dict(filterfalse(
        lambda x: x[0] is None,
        raw_mapping))

    return obj_id, uni_pdb_aln, mapping


@ray.remote
def calculate_variables_remote(
        obj_id: str,
        structure: Structure,
        variables: t.Iterable[AbstractVariable],
        aln_mapping: t.Dict[int, int]
) -> t.List[t.Tuple[str, bool, AbstractVariable, t.Union[str, float]]]:
    results = []
    for v in variables:
        try:
            results.append((obj_id, True, v, v.calculate(structure, aln_mapping)))
        except (MissingData, FailedCalculation) as e:
            results.append((obj_id, False, v, str(e)))
    return results


def extract_structure_domains(
        protein: Protein
) -> t.Union[Protein, t.Tuple[str, Exception]]:
    try:
        return extract_pdb_domains(protein)
    except Exception as e:
        return protein.id, e


def map_whole_structure_numbering(
        protein: Protein, sifts: SIFTS
) -> t.Dict[int, t.Optional[int]]:
    obj_id = f'{protein.pdb}:{protein.chain}'
    mappings = sifts.map_numbering(obj_id)

    def _raise():
        raise MissingData(
            f"No mapping between {protein.uniprot_id} "
            f"and {obj_id} in SIFTS")

    if mappings is None:
        _raise()
    else:
        mapping = peekable(
            filter(
                lambda m: m.id_from == protein.uniprot_id,
                mappings)
        ).peek(None)
        if mapping is None:
            _raise()
        else:
            return mapping


def assign_variables(
        variables: t.Sequence[_ParsedVariables],
        proteins: t.Sequence[Protein]
) -> None:
    def filter_objects(
            _id: t.Optional[str], domain: t.Optional[str] = None
    ) -> t.Union[t.List[Protein], t.List[Domain]]:

        @curry
        def get_matching_domains(protein: Protein, domain_name: str):
            return (d for k, d in protein.domains.items() if domain_name in k)

        def narrow_to_domain(
                _proteins: t.List[Protein]
        ) -> t.Union[t.List[Protein], t.List[Domain]]:
            if domain is None:
                LOGGER.debug(f'No domain to narrow down to for {_id}')
                return _proteins

            domains = list(chain.from_iterable(map(
                get_matching_domains(domain_name=domain), _proteins)))
            # domains = list(filter(
            #     bool, (p.domains.get(domain) for p in _proteins)))
            LOGGER.debug(f'Narrowed down to {len(domains)} {_id}::{domain}')
            return domains

        if _id is None:
            return narrow_to_domain(list(proteins))

        # First check whether _id matches UniProt ID of any protein
        # UniProt IDs are the broadest category, so we start with them
        match = [p for p in proteins if p.uniprot_id == _id]
        LOGGER.debug(f'Got {len(match)} proteins with UniProt ID '
                     f'matching {_id}')
        if match:
            return narrow_to_domain(match)

        # Check whether `_id` contains `chain_id` separated by `:`
        if ':' in _id:
            pdb_id, chain_id = split_validate(_id, ':', 2)
        else:
            pdb_id, chain_id = _id, None

        # Try matching by PDB ID
        match = [p for p in proteins if p.pdb == pdb_id.upper()]
        LOGGER.debug(f'Got {len(match)} proteins with PDB ID '
                     f'matching {pdb_id}')

        # If chain ID is present, filter by it
        if match and chain_id:
            match = [p for p in match if p.chain == chain_id]
            LOGGER.debug(f'Got {len(match)} proteins with PDB ID '
                         f'matching {pdb_id} and chain {chain_id}')
        return narrow_to_domain(match)

    def assign(
            variable: AbstractVariable,
            _id: t.Optional[str],
            domain: t.Optional[str]
    ) -> None:
        objects = filter_objects(_id, domain)
        if not objects:
            LOGGER.warning(
                f"No proteins with ID:domain {_id}:{domain} "
                f"for variable {variable}")
        for x in objects:
            x.variables[variable.id] = (variable, None)
            parent = f':{x.parent_name}' if isinstance(x, Domain) else ""
            LOGGER.debug(f'Created a placeholder for {variable} '
                         f'in object {x.id}{parent} ({type(x)})')

    # Flatten variables into unique setups
    # i.e., one variable - one protein ID - one domain
    flat_variables = flatten(starmap(product, variables))

    # Iterate over flattened variables and assign them to proteins
    for var, _id, _domain in flat_variables:
        LOGGER.debug(
            f'Assigning {var} to protein {_id} domain {_domain}')
        assign(var, _id, _domain)


if __name__ == '__main__':
    raise RuntimeError
