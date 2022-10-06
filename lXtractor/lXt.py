# """
# Main module for the package containing an ``lXtractor`` interface.
# """
# import logging
# import typing as t
# from copy import deepcopy, copy
# from itertools import chain, tee, filterfalse, starmap
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# # import ray
# from Bio.PDB.Structure import Structure
# from Bio.SeqRecord import SeqRecord as SeqRec
# from more_itertools import flatten, peekable, ilen
# from toolz import pipe, curry
# from toolz.curried import filter, map, groupby
# from tqdm.auto import tqdm
#
# from lXtractor.core.alignment import Alignment
# from lXtractor.core.base import AminoAcidDict
# from lXtractor.core.exceptions import MissingData, FormatError, FailedCalculation
# from lXtractor.core.config import Sep
# from lXtractor.ext.pdb import PDB
# from lXtractor.core.chain import Chain
# from lXtractor.ext.sifts import SIFTS
# from lXtractor.ext.uniprot import UniProt
# from lXtractor.input_parser import init
# from lXtractor.util.io import run_handles
# from lXtractor.util.seq import map_pairs_numbering, mafft_align
# from lXtractor.variables.parser import parse_var
#
# _KeyT = t.Union[int, str, slice, t.Sequence[bool], np.ndarray]
# _AlnT = t.Union[t.List[SeqRec], t.List[str], Alignment]
# LOGGER = logging.getLogger(__name__)
#
# # TODO: extend docs
# # TODO: optimize memory consumption for large datasets
# # TODO: -> use common (sequence/UniProt), including sequences
# # TODO: range variable
# # TODO: test init with various possible input types
#
#
# T = t.TypeVar('T')
#
#
# class lXtractor:
#     """
#     Main interface encompassing core functionalities.
#     """
#
#     def __init__(
#             self, inputs: t.Optional[t.Sequence[str]] = None,
#             proteins: t.Optional[t.List[Chain]] = None,
#             expected_domains: t.Optional[t.Union[t.Sequence[str]]] = None,
#             uniprot: t.Optional[UniProt] = None,
#             pdb: t.Optional[PDB] = None,
#             sifts: t.Optional[SIFTS] = None,
#             sifts_id_mapping: bool = True,
#             sifts_segment_mapping: bool = False,
#             alignment: t.Optional[Alignment] = None,
#             num_threads: t.Optional[int] = None
#     ):
#         """
#
#         :param inputs: A sequence of inputs to be parsed into ``Protein`` objects.
#         :param proteins: A list of protein objects. If provided, ignores ``inputs`` argument.
#         :param expected_domains:
#         :param uniprot:
#         :param pdb:
#         :param sifts:
#         :param sifts_id_mapping:
#         :param sifts_segment_mapping:
#         :param alignment:
#         :param num_threads:
#         """
#
#         if inputs is None and proteins is None:
#             raise ValueError('Must provide either ``inputs`` or ``proteins``')
#
#         self.alignment = alignment
#         self.pdb = pdb or PDB()
#         self.uniprot = uniprot or UniProt()
#
#         if isinstance(expected_domains, str):
#             expected_domains = [expected_domains]
#         self.expected_domains = expected_domains
#
#         self.sifts = sifts
#         if sifts_id_mapping and self.sifts is None:
#             self.sifts = SIFTS(load_segments=sifts_segment_mapping,
#                                load_id_mapping=sifts_id_mapping)
#             if sifts_segment_mapping and self.sifts.df is None:
#                 self.sifts.parse()
#
#         if proteins is None:
#
#             if expected_domains:
#                 expected_domains = ','.join(expected_domains)
#                 inputs = [
#                     f'{x}{Sep.dom}{expected_domains}' if Sep.dom not in x else x
#                     for x in inputs]
#
#             processed_inputs = []
#             for inp in tqdm(inputs, desc='Processing inputs'):
#                 try:
#                     processed_inputs.append(list(init(inp, self.sifts)))
#                 except Exception as e:
#                     LOGGER.exception(f'Failed to init input {inp} due to {e}')
#
#             self.proteins = list(flatten(processed_inputs))
#         else:
#             self.proteins = proteins
#         LOGGER.info(
#             f'Processed input into {len(self.proteins)} proteins')
#
#         self.num_threads = num_threads
#         if num_threads is not None:
#             if not ray.is_initialized():
#                 LOGGER.info(f'Initializing ray with {num_threads} threads')
#                 ray.init(num_cpus=num_threads)
#             else:
#                 LOGGER.info('Omitting ray init as it is already running')
#
#     def __len__(self) -> int:
#         return len(self.proteins)
#
#     def __repr__(self) -> str:
#         return f'lXtractor(proteins={len(self)})'
#
#     def __getitem__(self, key: _KeyT) -> t.Optional[t.Union[Chain, t.List[Chain]]]:
#         def get_one_or_more(xs):
#             if len(xs) > 1:
#                 return xs
#             if len(xs) == 1:
#                 return xs.pop()
#             return None
#
#         # INT or SLICE
#         if isinstance(key, int) or isinstance(key, slice):
#             return self.proteins[key]
#
#         # Sequence or boolean
#         if not isinstance(key, str) and (isinstance(key, t.Sequence) or isinstance(key, np.ndarray)):
#             if len(key) != len(self.proteins):
#                 raise IndexError(
#                     f'For boolean indexing, length of the idx ({len(key)}) '
#                     f'must match the number of proteins ({len(self.proteins)}')
#             return [x for x, b in zip(self.proteins, key) if b]
#
#         # Full ID
#         if '_' in key:
#             ps = list(filter(lambda p: p.id == key, self.proteins))
#
#         # PDB_ID:CHAIN
#         elif ':' in key:
#             ps = list(filter(
#                 lambda p: f'{p.pdb}:{p.chain}' == key,
#                 self.proteins))
#
#         # PDB_ID
#         elif len(key) == 4:
#             ps = list(
#                 filter(lambda p: p.pdb == key, self.proteins))
#
#         # UniProt ID
#         else:
#             ps = list(
#                 filter(lambda p: p.uniprot_id == key, self.proteins))
#
#         return get_one_or_more(ps)
#
#     def __copy__(self, **kwargs) -> "lXtractor":
#         """
#         """
#         ps = [copy(p) for p in self.proteins]
#         return self.__class__(
#             proteins=ps, expected_domains=self.expected_domains, uniprot=self.uniprot, pdb=self.pdb,
#             sifts=self.sifts, alignment=self.alignment)
#
#     def __deepcopy__(self, *args, **kwargs) -> "lXtractor":
#         ps = [deepcopy(p, *args, **kwargs) for p in self.proteins]
#         return self.__class__(
#             proteins=ps, expected_domains=self.expected_domains, uniprot=self.uniprot, pdb=self.pdb,
#             sifts=self.sifts, alignment=self.alignment)
#
#     def __iter__(self):
#         return iter(self.proteins)
#
#     @property
#     def domains(self) -> t.List:
#         return list(chain.from_iterable(p.children.values() for p in self))
#
#     @property
#     def variables(self) -> pd.DataFrame:
#         xs = []
#         for protein in self:
#             for v, r in protein.variables.items():
#                 xs.append((protein, np.nan, v, r))
#             for domain in protein:
#                 for v, r in domain.variables.items():
#                     xs.append((protein, domain, v, r))
#         return pd.DataFrame(xs, columns=['Protein', 'Domain', 'Variable', 'Result'])
#
#     def subset(self, key: _KeyT, deep: bool = False) -> "lXtractor":
#         ps = self.__getitem__(key)
#         if isinstance(ps, Chain):
#             ps = [ps]
#         tmp = self.proteins
#         self.proteins = ps
#         new_lxt = deepcopy(self) if deep else copy(self)
#         self.proteins = tmp
#         return new_lxt
#
#     def query(
#             self,
#             protein_specs: t.Optional[t.Sequence[str]],
#             domain_specs: t.Optional[t.Sequence[str]]
#     ) -> t.Union[t.List[Chain], t.List]:
#         objs = self.proteins
#         if protein_specs:
#             objs = pipe(
#                 (self[x] for x in protein_specs),
#                 map(lambda x: [x] if isinstance(x, Chain) else x),
#                 flatten,
#                 list)
#         if domain_specs:
#             objs = pipe(
#                 objs,
#                 map(lambda p: (p.children.get_item(x, None) for x in domain_specs)),
#                 chain.from_iterable,
#                 filter(bool),
#                 list
#             )
#         return objs
#
#     def fetch_uniprot(
#             self, missing: bool = True, keep_expected: bool = True,
#             uniprot: t.Optional[UniProt] = None, **kwargs
#     ) -> t.Optional[pd.DataFrame]:
#         expected_domains = self.expected_domains
#         if keep_expected and not expected_domains:
#             LOGGER.warning('`keep_expected` is true but `domains` attribute is empty')
#         if uniprot is None:
#             if self.uniprot is None:
#                 raise ValueError('No UniProt instance initialized')
#             uniprot = self.uniprot
#         proteins = list(filter(
#             lambda p: p.uniprot_id is not None and (
#                     not missing or p.uniprot_seq is None or not p.children),
#             self.proteins))
#
#         LOGGER.info(f'Found {len(proteins)} to fetch UniProt data')
#         df = None
#         if proteins:
#             _, df = uniprot.fetch_proteins_data(
#                 proteins, keep_expected_if_any=keep_expected, **kwargs)
#         return df
#
#     def fetch_pdb(self, missing: bool = True) -> None:
#         if self.pdb is None:
#             raise ValueError('No PDB instance initialized')
#         proteins = list(filter(
#             lambda p: p.pdb is not None and (not missing or p.structure is None),
#             self.proteins))
#         LOGGER.info(f'Found {len(proteins)} to fetch PDB data')
#         if proteins:
#             self.pdb.fetch(proteins)
#
#     def map_uni_pdb(
#             self, missing: bool = True,
#             alignment_based: bool = False,
#             length_cap: t.Optional[int] = None):
#
#         if alignment_based:
#             proteins = [
#                 p for p in self.proteins if
#                 (not p.uni_pdb_map or missing) and
#                 p.structure and
#                 (p.uniprot_seq or p.uni_pdb_aln) and
#                 (length_cap is None or len(p.uniprot_seq) < length_cap)
#             ]
#         else:
#             proteins = [
#                 p for p in self.proteins if
#                 (not p.uni_pdb_map or missing) and
#                 p.pdb and p.chain and p.uniprot_id
#             ]
#
#         LOGGER.info(f'Found {len(proteins)} proteins to map numbering')
#
#         if proteins:
#
#             if alignment_based:
#                 method = 'MSA'
#                 self.map_numbering_aln(proteins)
#             else:
#                 method = 'SIFTS'
#                 self.map_numbering_sifts(proteins)
#
#             LOGGER.info(f'Used {method}-based mapping on {len(proteins)} proteins')
#
#     def map_numbering_sifts(self, proteins: t.Optional[t.Sequence[Chain]] = None):
#         proteins = proteins or self.proteins
#
#         for p in tqdm(
#                 proteins, desc='Mapping SIFTS numbering', position=0, leave=True):
#             try:
#                 p.uni_pdb_map = map_whole_structure_numbering(p, self.sifts)
#             except ValueError as e:
#                 LOGGER.warning(
#                     f"Failed to map whole chain sequence for protein "
#                     f"{p.id} due to error {e}")
#
#     def map_numbering_aln(self, proteins: t.Optional[t.Sequence[Chain]] = None):
#         proteins = proteins or self.proteins
#         inputs = [(p.id, p.structure, p.uni_pdb_aln, p.uniprot_seq, p.pdb_seq)
#                   for p in proteins]
#         handles = [map_by_alignment_remote.remote(*inp) for inp in inputs]
#         bar = tqdm(desc='Mapping Uni-PDB by alignment', total=len(inputs),
#                    position=0, leave=True)
#         results = run_handles(handles, bar)
#         proteins = {p.id: p for p in proteins}
#         for p_id, uni_pdb_aln, mapping in results:
#             proteins[p_id].uni_pdb_aln = uni_pdb_aln
#             if isinstance(mapping, Exception):
#                 LOGGER.error(f'Failed to acquire mapping for {p_id} '
#                              f'due to error {mapping}')
#                 continue
#             proteins[p_id].uni_pdb_map = mapping
#
#     def align(self, domain: t.Optional[t.Union[str, t.Container[str]]] = None,
#               pdb: bool = True, method=mafft_align) -> Alignment:
#
#         def accept(obj: t.Union[Chain]):
#             return ((domain and isinstance(obj, Domain) and obj.name in domain)
#                     or not domain)
#
#         if domain is not None:
#             if isinstance(domain, str):
#                 domain = [domain]
#             obj_type = 'domain'
#             objs = self.domains
#         else:
#             obj_type = 'protein'
#             objs = self.proteins
#
#         seqs = pipe(
#             objs,
#             filter(accept),
#             map(lambda x: x.pdb_seq1 if pdb else x.uniprot_seq),
#             filter(lambda x: x is not None),
#             list)
#
#         LOGGER.debug(f'Found {len(seqs)} {obj_type} sequences to align')
#
#         if not seqs:
#             msg = 'No suitable sequences to align'
#             LOGGER.error(msg)
#             raise MissingData(msg)
#
#         seqs = method(seqs)
#
#         LOGGER.info(f'Aligned {len(seqs)} {obj_type} sequences')
#
#         return Alignment(seqs=seqs)
#
#     def map_numbering_to_msa(
#             self, alignment: _AlnT, domains: t.Optional[t.Union[t.Sequence[str], str]] = None,
#             pdb_seq: bool = True, parallel: bool = False, missing: bool = True) -> None:
#
#         def map_to_alignment(
#                 seq: str, num: t.Sequence[int], aln: Alignment
#         ) -> t.Tuple[str, t.Sequence[int], t.Union[t.Dict[int, int], Exception]]:
#             try:
#                 return seq, num, aln.map_seq_numbering(SeqRec(Seq(seq)), num)
#             except Exception as e:
#                 return seq, num, e
#
#         def get_seq(obj: t.Union[Chain]) -> str:
#             if pdb_seq:
#                 if obj.pdb_seq1 is None:
#                     return get_sequence(obj.structure)
#                 return str(obj.pdb_seq1.seq)
#             return str(obj.uniprot_seq.seq)
#
#         def get_num(obj: t.Union[Chain]) -> t.Tuple[int, ...]:
#             if pdb_seq:
#                 return get_sequence(obj.structure)
#             start = obj.start if isinstance(obj, Domain) else 1
#             return tuple(i for i, x in enumerate(obj.uniprot_seq, start=start))
#
#         def accept(obj: t.Union[Chain]):
#
#             acc_exist = (missing and obj.aln_mapping is None) or not missing
#             acc_dom = (domains is not None and obj.name in domains) or domains is None
#             acc_seq = ((pdb_seq and obj.structure is not None) or
#                        (not pdb_seq and obj.uniprot_seq is not None))
#
#             return all([acc_exist, acc_dom, acc_seq])
#
#         def prep_aln(aln: _AlnT) -> Alignment:
#             if isinstance(aln, Alignment):
#                 return aln
#             elif isinstance(aln, t.Sequence):
#                 if not aln:
#                     raise MissingData('No sequences in the provided aln')
#                 if isinstance(aln[0], str):
#                     seqs = [SeqRec(Seq(s), i, i, i) for i, s in enumerate(aln)]
#                 elif isinstance(aln[0], SeqRec):
#                     seqs = aln
#                 else:
#                     raise ValueError('Unsupported sequence type')
#                 return Alignment(seqs=seqs)
#             else:
#                 raise ValueError('Unsupported aln type')
#
#         alignment = prep_aln(alignment)
#         domains = [domains] if isinstance(domains, str) else domains
#
#         # Group objects by actual sequence to optimize mapping
#         groups = pipe(
#             self.domains if domains else self.proteins,
#             filter(accept),
#             map(lambda x: (x, get_seq(x), get_num(x))),
#             groupby(lambda x: (x[1], x[2]))
#         )
#         desc, total, total_objs = (
#             f'Mapping to an MSA {alignment}', len(groups), sum(map(len, groups.values())))
#         LOGGER.info(f'Found {total} unique sequence (from {total_objs} objects) for MSA mapping')
#
#         if parallel:
#             aln_remote = ray.put(alignment)
#             fn = ray.remote(map_to_alignment)
#             bar = tqdm(position=0, leave=True, desc=desc, total=total)
#             handles = [fn.remote(seq, num, aln_remote) for seq, num in groups]
#             results = run_handles(handles, bar)
#         else:
#             bar = tqdm(groups, position=0, leave=True, desc=desc, total=total)
#             results = [map_to_alignment(seq, num, alignment) for seq, num in bar]
#
#         for seq, numbering, mapping in results:
#             objs = [x for x, _, _ in groups[(seq, numbering)]]
#             if not isinstance(mapping, t.Dict):
#                 ids = [x.id for x in objs]
#                 LOGGER.exception(f'Failed to map sequence for {ids} due to error {mapping}')
#                 continue
#             for x in objs:
#                 x.aln_mapping = mapping
#                 LOGGER.debug(f'Assigned new aln mapping to {x.id}')
#
#     def extract_structure_domains(self, parallel: bool = False) -> None:
#
#         def extract_structure_domains(
#                 protein: Chain
#         ) -> t.Union[Chain, t.Tuple[str, Exception]]:
#             try:
#                 _ = protein.extract_domains(pdb=True, inplace=True)
#                 return protein
#             except Exception as e:
#                 return protein.id, e
#
#         proteins = {
#             p.id: p for p in self.proteins if
#             (p.structure and p.children and p.uni_pdb_map)}
#
#         LOGGER.info(f'Found {len(proteins)} proteins for '
#                     f'structure domains extraction.')
#         if parallel:
#             fn = ray.remote(extract_structure_domains)
#             bar = tqdm(
#                 desc='Extracting structure domains',
#                 total=len(proteins), position=0, leave=True)
#             handles = [fn.remote(p) for p in proteins.values()]
#             results = run_handles(handles, bar)
#         else:
#             results = list(map(
#                 extract_structure_domains,
#                 tqdm(proteins.values(), desc='Extracting structure domains',
#                      total=len(proteins), position=0, leave=True)))
#         for r in results:
#             if not isinstance(r, Chain):
#                 p_id, exc = r
#                 LOGGER.exception(f'Failed to extract domains from {p_id} due to {exc}')
#             else:
#                 proteins[r.id].children = r.children
#
#     def compute_seq_meta(self, exclude_mod: t.Tuple[str, ...] = ('HOH',)):
#         # TODO: rm when range variable implemented
#         proteins = [
#             p for p in self.proteins if (p.structure and p.pdb_seq and p.uniprot_seq)]
#         if proteins:
#             LOGGER.info(f'Found {len(proteins)} proteins to calculate seq metadata')
#             inputs = [
#                 (p.id, p.uniprot_seq, p.pdb_seq, p.structure, p.uni_pdb_aln, exclude_mod)
#                 for p in proteins]
#             handles = [seq_stats_remote.remote(*inp) for inp in inputs]
#             bar = tqdm(
#                 desc='Calculating sequence metadata (protein level)',
#                 total=len(inputs), position=0, leave=True)
#             results = run_handles(handles, bar)
#             proteins_map = {p.id: p for p in proteins}
#             for obj_id, aln, meta in results:
#                 proteins_map[obj_id].metadata += meta
#                 proteins_map[obj_id].uni_pdb_aln = aln
#
#         domains = dict(chain.from_iterable(
#             ((f'{p.id}_{name}', dom) for name, dom in p.children.items()
#              if (dom.structure and dom.pdb_seq and dom.uniprot_seq))
#             for p in self.proteins))
#         if domains:
#             LOGGER.info(
#                 f'Found {len(domains)} domains to calculate seq metadata')
#             inputs = [
#                 (dom_id, dom.uniprot_seq, dom.pdb_seq,
#                  dom.structure, dom.uni_pdb_aln, exclude_mod)
#                 for dom_id, dom in domains.items()]
#             handles = [seq_stats_remote.remote(*inp) for inp in inputs]
#             bar = tqdm(
#                 desc='Calculating sequence metadata (domain level)',
#                 total=len(inputs), position=0, leave=True)
#             results = run_handles(handles, bar)
#             for obj_id, aln, meta in results:
#                 domains[obj_id].metadata += meta
#                 domains[obj_id].uni_pdb_aln = aln
#
#     def assign_variables(
#             self, variables: t.Union[str, t.Sequence[str]],
#             overwrite: bool = False
#     ) -> None:
#         def try_parse(v: str):
#             try:
#                 parsed_var = parse_var(v)
#                 LOGGER.debug(f'Successfully parsed input {v}')
#                 return parsed_var
#             except (ValueError, FormatError) as e:
#                 LOGGER.exception(
#                     f'Failed to parse variable {v} due to error {e}')
#                 return None
#
#         if isinstance(variables, str):
#             variables = [variables]
#
#         parsed_variables = list(filter(bool, map(try_parse, variables)))
#         LOGGER.info(f'Parsed {len(variables)} inputs into {len(parsed_variables)} variables')
#         if not parsed_variables:
#             LOGGER.warning('No variables to assign')
#             return None
#
#         for vs, p_specs, d_specs in parsed_variables:
#             _objs = self.query(p_specs, d_specs)
#             print(vs, p_specs, d_specs, _objs)
#             if not _objs:
#                 LOGGER.warning(
#                     f'No objects falling under specifications '
#                     f'protein={p_specs} domain={d_specs} for variables {vs}')
#             for obj in _objs:
#                 for variable in vs:
#                     print(obj, variable)
#                     if variable in obj.variables:
#                         if overwrite:
#                             obj.variables[variable] = None
#                             LOGGER.debug(f'Overwritten existing {obj}-s variable {variable}')
#                         else:
#                             LOGGER.debug(f'Skipping existing {obj}-s variable {variable}')
#                     else:
#                         obj.variables[variable] = None
#                         LOGGER.debug(f'Assigned new variable {variable} to {obj}')
#
#         LOGGER.debug(f'Successfully assigned variables')
#
#     def calculate_variables(
#             self, parallel: bool = True, missing: bool = True
#     ) -> None:
#
#         def calculate(
#                 protein_id: str, domain_id: t.Optional[str],
#                 target: t.Union[Structure, SeqRec],
#                 variables,
#                 mapping: t.Dict[int, int]
#         ) -> t.List[t.Tuple[str, str, T, t.Union[str, float, Exception]]]:
#             rs = []
#             for v in variables:
#                 try:
#                     rs.append((protein_id, domain_id, v, v.calculate(target)))
#                 except (MissingData, FailedCalculation) as e:
#                     LOGGER.exception(
#                         f'Failed calculating {v} on {protein_id}::{domain_id} due to: {e}')
#                     rs.append((protein_id, domain_id, v, e))
#             return rs
#
#         def accept_obj(obj: t.Union[Chain]):
#             if obj.aln_mapping is None:
#                 return False
#             if str:
#                 return obj.structure is not None
#             return obj.uniprot_seq is not None
#
#         def get_ids(obj: t.Union[Chain]):
#             if isinstance(obj, Domain):
#                 return obj.parent_name, obj.name
#             else:
#                 return obj.id, None
#
#         @curry
#         def get_vars(
#                 obj: t.Union[Chain], str_var: bool = True
#         ) -> t.Optional[t.Tuple[str, t.Optional[str], Structure, t.List[T], t.Dict[int, int]]]:
#             vs = list(filter(
#                 lambda v: missing and obj.variables[v] is None or not missing,
#                 obj.variables.structure if str_var else obj.variables.sequence
#             ))
#             if vs:
#                 needed = 'Both structure and MSA mapping are' if str_var else 'UniProt sequence is'
#                 if not accept_obj(obj):
#                     LOGGER.warning(f'{needed} needed to calculate variables {vs} for {obj}')
#                     return None
#             else:
#                 LOGGER.debug(f'No variables for {obj} (structural={str_var})')
#                 return None
#
#             target = obj.structure if str_var else obj.uniprot_seq
#             p_id, d_id = get_ids(obj)
#
#             return p_id, d_id, target, vs, obj.aln_mapping
#
#         staged = filter(bool, chain(
#             map(get_vars, self.proteins),  # Structural variables
#             map(get_vars, self.domains),
#             map(get_vars(str_var=False), self.proteins),  # Sequence variables
#             map(get_vars(str_var=False), self.domains)
#         ))
#
#         if parallel:
#             fn = ray.remote(calculate)
#             handles = [fn.remote(*inp) for inp in staged]
#             bar = tqdm(desc='Calculating variables', total=len(handles))
#             results = run_handles(handles, bar)
#         else:
#             results = starmap(calculate, staged)
#
#         for p_id, domain_name, v, res in chain.from_iterable(results):
#             obj_id = p_id
#             if domain_name:
#                 obj_id += f'{Sep.dom}{domain_name}'
#             if isinstance(res, Exception):
#                 LOGGER.error(f'Failed on {obj_id} due to: {res}')
#             else:
#                 LOGGER.debug(f'Calculated variable {v}={res} for {obj_id}')
#                 self[p_id][domain_name].variables[v] = res
#
#     def dump(self):
#         raise NotImplementedError
#
#
# @ray.remote
# def parse_structure_remote(
#         path: Path
# ) -> t.Union[t.Tuple[Structure, t.List[t.Tuple[str, t.Any]]], t.Tuple[str, Exception]]:
#     try:
#         _, structure, meta = _wrap_raw_pdb(path.read_text())
#         return structure, meta
#     except Exception as e:
#         return path.stem, e
#
#
# @ray.remote
# def seq_stats_remote(
#         obj_id: str, uni_seq: SeqRec,
#         pdb_seq: SeqRec, structure: Structure,
#         uni_pdb_aln: t.Optional[t.Tuple[SeqRec, SeqRec]],
#         exclude: t.Tuple[str, ...] = ('HOH',),
# ) -> t.Tuple[str, t.Tuple[SeqRec, SeqRec], t.List[t.Tuple[str, t.Union[str, float]]]]:
#     def coverage(seq1: SeqRec, seq2: SeqRec) -> float:
#         try:
#             sub = tee((c1, c2) for c1, c2 in zip(seq1, seq2) if c1 != '-')
#             return ilen(c for _, c in sub[0] if c != '-') / ilen(sub[1])
#         except Exception as e:
#             LOGGER.warning(
#                 f'Failed to calculate sequence {seq1.id}({obj_id}) '
#                 f'coverage due to {e}')
#             return 0
#
#     aa_dict = AminoAcidDict()
#     mods = [
#         f'{r.resname}_{r.id[1]}' for r in structure.get_residues()
#         if r.resname not in aa_dict and r.resname not in exclude]
#
#     if uni_pdb_aln:
#         uni_aln, pdb_aln = uni_pdb_aln
#     else:
#         uni_aln, pdb_aln = mafft_align(
#             [uni_seq, pdb_seq], thread=1)
#     num_gaps = sum(
#         1 for x, y in zip(uni_aln, pdb_aln)
#         if x == '-' or y == '-')
#     num_miss = sum(
#         1 for x, y in zip(uni_aln, pdb_aln)
#         if x != '-' and y != '-' and x != y)
#     return obj_id, (uni_aln, pdb_aln), [
#         ('PDB non-canonical', ';'.join(mods)),
#         ('PDB num non-canonical', len(mods)),
#         ('Uni seq coverage', round(coverage(uni_aln, pdb_aln), 2)),
#         ('PDB seq coverage', round(coverage(pdb_aln, uni_aln), 2)),
#         ('Uni-PDB num gaps', num_gaps),
#         ('Uni-PDB num mismatches', num_miss),
#     ]
#
#
# @ray.remote
# def map_by_alignment_remote(
#         obj_id: str,
#         structure: Structure,
#         uni_pdb_aln: t.Optional[t.Tuple[SeqRec, SeqRec]] = None,
#         uni_seq: t.Optional[SeqRec] = None,
#         pdb_seq: t.Optional[SeqRec] = None,
# ) -> t.Tuple[str, t.Tuple[SeqRec, SeqRec], t.Union[t.Dict[int, t.Optional[int]], Exception]]:
#     if uni_seq is None and uni_pdb_aln is None:
#         raise ValueError(
#             f'Wrong input for {obj_id}. Must provide a UniProt sequence: '
#             f'either in the `uni_pdb_aln` or via `uni_seq`')
#     if uni_pdb_aln is None:
#         if pdb_seq is None:
#             pdb_seq = SeqRec(
#                 # Do not filter out anything so that the numbering's length
#                 # always matches the sequence's length
#                 Seq(get_sequence(structure, exclude=())),
#                 id=structure.id, name=structure.id,
#                 description=structure.id)
#         uni_pdb_aln = mafft_align([uni_seq, pdb_seq])
#
#     uni_aln, pdb_aln = uni_pdb_aln
#
#     uni_num = list(range(1, len(uni_seq) + 1))
#     pdb_num = [r.id[1] for r in structure.get_residues()]
#     try:
#         raw_mapping = map_pairs_numbering(
#             uni_aln, uni_num, pdb_aln, pdb_num, align=False)
#     except Exception as e:
#         return obj_id, uni_pdb_aln, e
#
#     mapping = dict(filterfalse(
#         lambda x: x[0] is None,
#         raw_mapping))
#
#     return obj_id, uni_pdb_aln, mapping
#
#
# def map_whole_structure_numbering(
#         protein: Chain, sifts: SIFTS
# ) -> t.Dict[int, t.Optional[int]]:
#     # TODO - rm?
#     obj_id = f'{protein.pdb}:{protein.chain}'
#     mappings = sifts.map_numbering(obj_id)
#
#     def _raise():
#         raise MissingData(
#             f"No mapping between {protein.uniprot_id} "
#             f"and {obj_id} in SIFTS")
#
#     if mappings is None:
#         _raise()
#     else:
#         mapping = peekable(
#             filter(
#                 lambda m: m.id_from == protein.uniprot_id,
#                 mappings)
#         ).peek(None)
#         if mapping is None:
#             _raise()
#         else:
#             return mapping
#
#
# if __name__ == '__main__':
#     raise RuntimeError
