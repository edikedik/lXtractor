import logging
import typing as t
from itertools import starmap, product, chain, tee, filterfalse
from pathlib import Path

import ray
from Bio.PDB.Structure import Structure
from more_itertools import flatten, peekable, ilen
from toolz import groupby, curry
from tqdm.auto import tqdm

from lXtractor.alignment import Alignment, mafft_align, map_pairs_numbering
from lXtractor.base import (
    SeqRec, FormatError, MissingData, AbstractVariable,
    FailedCalculation, Seq, Domain, AminoAcidDict)
from lXtractor.domains import extract_uniprot_domains, extract_pdb_domains
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
LOGGER = logging.getLogger(__name__)

# TODO: extend docs
# TODO: optimize memory consumption for large datasets
# TODO: -> use common (sequence/UniProt), including sequences
# TODO: simplify init

T = t.TypeVar('T')


class lXtractor:
    def __init__(
            self, inputs: t.Optional[t.Sequence[str]] = None,
            proteins: t.Optional[t.List[Protein]] = None,
            domains: t.Optional[t.Sequence[str]] = None,
            uniprot: t.Optional[UniProt] = None,
            pdb: t.Optional[PDB] = None,
            sifts: t.Optional[SIFTS] = None,
            sifts_id_mapping: bool = True,
            sifts_segment_mapping: bool = False,
            alignment: t.Optional[Alignment] = None,
            pdb_dir: t.Optional[Path] = None,
    ):

        if inputs is None and proteins is None:
            raise ValueError('Must provide either ``inputs`` or ``proteins``')

        self.alignment = alignment
        if pdb_dir is not None and not pdb_dir.exists():
            pdb_dir.mkdir(parents=True)
        self.pdb_dir = pdb_dir
        self.pdb = pdb or PDB(pdb_dir=pdb_dir)
        self.uniprot = uniprot or UniProt()
        self.domains = domains

        self.sifts = sifts
        self.sifts_id_mapping = sifts_id_mapping
        if sifts_id_mapping and self.sifts is None:
            self.sifts = SIFTS()
            if sifts_segment_mapping and sifts.df is None:
                self.sifts.parse()

        if proteins is None:

            if domains:
                domains = ','.join(domains)
                inputs = [f'{x}::{domains}' for x in inputs]

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

    def __len__(self) -> int:
        return len(self.proteins)

    def __repr__(self) -> str:
        return f'lXtractor(id={id(self)},proteins={len(self)})'

    def __getitem__(
            self, id_: t.Union[int, str]
    ) -> t.Optional[t.Union[Protein, t.List[Protein]]]:
        if isinstance(id_, int):
            return self.proteins[id_]

        prots = list(filter(lambda p: p.id == id_, self.proteins))
        if len(prots) == 1:
            return prots.pop()
        if len(prots) > 1:
            return prots
        if ':' in id_:
            return list(filter(
                lambda p: f'{p.pdb}:{p.chain}' == id_,
                self.proteins))
        prots = list(filter(lambda p: p.uniprot_id == id_, self.proteins))
        if prots:
            return prots
        prots = list(filter(lambda p: p.pdb == id_, self.proteins))
        if prots:
            return prots
        return None

    def __iter__(self):
        return iter(self.proteins)

    def fetch_missing(self, uniprot: bool = True, pdb: bool = True) -> None:

        if uniprot:
            missing_uniprot = [p for p in self.proteins
                               if p.uniprot_seq is None and p.uniprot_id is not None]
            if missing_uniprot:
                LOGGER.info(
                    f'Found {len(missing_uniprot)} proteins with no UniProt sequence')
                self.uniprot.fetch_fasta(missing_uniprot)
        if pdb:
            missing_pdb = [p for p in self.proteins if p.structure is None and p.pdb is not None]
            if missing_pdb:
                LOGGER.info(
                    f'Found {len(missing_pdb)} proteins with no PDB structure')
                self.pdb.fetch(missing_pdb)

    def map_uni_pdb(
            self, ignore_existing: bool = False,
            alignment_based: bool = False,
            length_cap: t.Optional[int] = None):

        if alignment_based:
            proteins = [
                p for p in self.proteins if
                (not p.uni_pdb_map or ignore_existing) and
                p.structure and
                (p.uniprot_seq or p.uni_pdb_aln) and
                (length_cap is None or len(p.uniprot_seq) < length_cap)
            ]
        else:
            proteins = [
                p for p in self.proteins if
                (not p.uni_pdb_map or ignore_existing) and
                p.pdb and p.chain and p.uniprot_id
            ]
        LOGGER.info(
            f'Found {len(proteins)} proteins to map numbering')
        if proteins:
            if alignment_based:
                self.map_with_alignment(proteins)
            else:
                self.map_with_sifts(proteins)

    def map_with_sifts(self, proteins: t.Optional[t.Sequence[Protein]] = None):
        proteins = proteins or self.proteins

        for p in tqdm(
                proteins, desc='Mapping SIFTS numbering', position=0, leave=True):
            try:
                p.uni_pdb_map = map_whole_structure_numbering(p, self.sifts)
            except ValueError as e:
                LOGGER.warning(
                    f"Failed to map whole chain sequence for protein "
                    f"{p.id} due to error {e}")

    def map_with_alignment(self, proteins: t.Optional[t.Sequence[Protein]] = None):
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

    def fetch_domain_boundaries(self) -> None:
        def partial_match(s1: str, s2: str) -> bool:
            return s1 in s2 or s2 in s1

        proteins = self.uniprot.fetch_domains(self.proteins)
        for p in proteins:
            if not p.domains:
                LOGGER.warning(f'No domains extracted for proteins {p.id}')
                continue
            if p.expected_domains is not None:
                missing = set()
                for exp in p.expected_domains:
                    if not any(partial_match(exp, d) for d in p.domains):
                        missing.add(exp)
                if missing:
                    LOGGER.warning(
                        f"Among the expected domains {p.expected_domains} "
                        f"for protein {p.id} "
                        f"failed to extract the following: {missing}")
            LOGGER.debug(
                f'Assigned sequence domains {list(p.domains)} to protein {p.id}')

    def fetch_meta(self, fields=('entry_name', 'families')) -> None:

        proteins = [p for p in self.proteins if p.uniprot_id is not None]
        LOGGER.info(f'Fetching metadata for {len(proteins)} '
                    f'out of {len(self.proteins)} proteins')
        self.uniprot.fetch_meta(proteins, fields=fields)

    def extract_sequence_domains(self, keep_parent: bool = True) -> None:
        # TODO: rm

        proteins = [p for p in self.proteins if not any(
            [p.uniprot_seq is None, p.domains is None])]

        LOGGER.info(f'Found {len(proteins)} proteins for '
                    f'sequence domains extraction.')

        for p in tqdm(
                proteins, desc='Extracting sequence domains',
                position=0, leave=True):
            try:
                extract_uniprot_domains(p, keep_parent)
            except MissingData as e:
                LOGGER.warning(
                    f'Failed to extract UniProt domains due to error {e}')

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

    def compute_seq_meta(self, exclude_mod: t.Tuple[str, ...] = ('HOH',)):
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
             if (dom.pdb_sub_structure and dom.pdb_seq and dom.uniprot_seq))
            for p in self.proteins))
        if domains:
            LOGGER.info(
                f'Found {len(domains)} domains to calculate seq metadata')
            inputs = [
                (dom_id, dom.uniprot_seq, dom.pdb_seq,
                 dom.pdb_sub_structure, dom.uni_pdb_aln, exclude_mod)
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

    def map_to_alignment(
            self, alignment: Alignment,
            skip_mapped: bool = True
    ) -> None:
        def get_structure(
                obj: t.Union[Domain, Protein]
        ) -> Structure:
            return (
                obj.structure if isinstance(obj, Protein)
                else obj.pdb_sub_structure)

        def group_unique_seqs(
                objs: t.Iterable[t.Union[Domain, Protein]],
        ) -> t.Dict[t.Tuple[str, t.Tuple[int, ...]], t.Dict[int, int]]:
            staged = (
                (obj, get_structure(obj))
                for obj in objs)
            staged = (
                (obj,
                 get_sequence(structure),
                 tuple(r.get_id()[1] for r in structure.get_residues())
                 ) for obj, structure in staged
            )
            return groupby(lambda x: (x[1], x[2]), staged)

        proteins = [
            p for p in self.proteins if (p.variables and p.structure and (
                    not skip_mapped or not p.aln_mapping))]
        LOGGER.info(f'Found {len(proteins)} full protein chains '
                    f'for mapping to the alignment')

        domains = [
            d for d in flatten(p.domains.values() for p in self.proteins)
            if d.variables and d.pdb_sub_structure and (
                    not skip_mapped or not d.aln_mapping)]
        LOGGER.info(f'Found {len(domains)} domains for mapping to the alignment')

        groups = group_unique_seqs(chain(proteins, domains))
        LOGGER.info(f'Found {len(groups)} unique sequences to map')

        aln_remote = ray.put(alignment)
        bar = tqdm(
            desc='Mapping to an MSA', total=len(groups), position=0, leave=True)
        handles = [
            map_to_alignment_remote.remote(seq, numbering, aln_remote)
            for seq, numbering in groups]
        results = run_handles(handles, bar)

        for seq, numbering, mapping in results:
            objects = [_obj for _obj, _, _ in groups[(seq, numbering)]]
            if not isinstance(mapping, t.Dict):
                ids = [_obj.id if isinstance(_obj, Protein) else f'{_obj.id}({_obj.parent_name})'
                       for _obj in objects]
                LOGGER.warning(
                    f'Failed to map sequence of the group {ids} '
                    f'due to error {mapping}')
                continue
            for _obj in objects:
                _obj.aln_mapping = mapping
                LOGGER.debug(f'Assigned new aln mapping to {_obj.id}')

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
            lambda x: all([x[-1].variables, x[-1].pdb_sub_structure, x[-1].aln_mapping]),
            domains))
        LOGGER.info(f'Found {len(domains)} domains with '
                    f'variables staged for calculation')
        if domains:
            inputs = [(f'{p_id}---{k}', d.pdb_sub_structure, get_variables(d), d.aln_mapping)
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


@ray.remote
def map_to_alignment_remote(
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
