# """
# Module with misc experimental procedures.
# """
# import logging
# import typing as t
# from random import sample
#
# import pandas as pd
# import ray
# from toolz import groupby, curry
# from tqdm.auto import tqdm
#
# from lXtractor import Protein, UniProt
# from lXtractor.util.seq import seq_identity, seq_coverage, mafft_align
# from lXtractor.core.chain import unduplicate
# from lXtractor.util.io import run_handles
#
# LOGGER = logging.getLogger(__name__)
#
#
# @ray.remote
# def calc_identity_and_coverage(seq1, seq2):
#     seq1_aligned, seq2_aligned = mafft_align([seq1, seq2])
#     return (
#         seq1.id.split('|')[1],
#         seq2.id.split('|')[1],
#         seq_identity(seq1_aligned, seq2_aligned, False),
#         seq_coverage(seq2_aligned, seq1_aligned, False)
#     )
#
#
# def identity_and_coverage(
#         proteins: t.Sequence[Protein],
#         seed_map: t.Dict[str, str],
#         domain_name: str,
# ):
#     id2p = {p.id: p for p in proteins}
#
#     inputs = [
#         (p.children[domain_name].uniprot_seq,
#          id2p[seed_map[p.id]].children[domain_name].uniprot_seq)
#         for p in proteins if
#         p.id not in seed_map.values() and
#         p.id in seed_map and
#         seed_map[p.id] in id2p
#     ]
#
#     handles = [calc_identity_and_coverage.remote(s1, s2) for s1, s2 in inputs]
#     bar = tqdm(desc='Aligning pairs', total=len(inputs), position=0, leave=True)
#     results = run_handles(handles, bar)
#
#     for id_ortholog, id_seed, ident, coverage in results:
#         id2p[id_ortholog].children[domain_name].meta['Seed'] = id_seed
#         id2p[id_ortholog].children[domain_name].meta['Ident'] = ident
#         id2p[id_ortholog].children[domain_name].meta['Coverage'] = coverage
#
#
# def find_domain_orthologs(
#         seeds: t.List[str],
#         domain_name: str,
#         num: t.Optional[int] = None,
#         subsample: bool = True,
#         exclude_by_existence: t.Collection[str] = ('Predicted',),
#         exclude_from_results: t.Optional[t.Collection[str]] = None,
#         identity_bounds: t.Tuple[float, float] = (0.8, 0.98),
#         min_coverage: float = 0.9,
#
#         uniprot: t.Optional[UniProt] = None
# ) -> (t.Dict[str, Protein], pd.DataFrame):
#     """
#     Find orhtologous sequences in UniProt, on a domain level. Namely,
#
#     #. Take in a list of seed sequences
#     #. Find relevant UniProt IDs corresponding to the same gene name as seeds
#     #. Filter the obtained list by 'existence' level
#     #. Extract the same domain for each of the filtered sequences
#     #. Align each domain to a domain of a corresponding seed sequence
#         to calculate sequence identity
#     #. Filter out sequences by the provided identity bounds
#     #. Group sequences by seed id, and do one of the
#         - Pick ``num`` random sequences (``subsample`` is true)
#         - Pick ``num`` sequences with the highest identity
#
#     WARNING: mind potential isoforms in the results
#
#     :param seeds:
#     :param domain_name:
#     :param num:
#     :param subsample:
#     :param exclude_by_existence:
#     :param exclude_from_results:
#     :param identity_bounds:
#     :param min_coverage:
#     :param uniprot:
#     :return:
#     """
#     if uniprot is None:
#         uniprot = UniProt()
#
#     ps = [Protein() for x in seeds]
#     df = uniprot.fetch_orthologs(
#         ps, output_columns=['id', 'existence', 'reviewed', 'organism'])
#     LOGGER.debug(f'Fetched {len(df)} candidates')
#     if exclude_by_existence:
#         df = df[~df['existence'].isin(exclude_by_existence)]
#         LOGGER.debug(f'Filtered by existense to {len(df)} records')
#     if exclude_from_results is not None:
#         df = df[~df['id'].isin(exclude_from_results)]
#         LOGGER.debug(f'Excluded by {len(exclude_from_results)} '
#                      f'provided ids to {len(df)} records')
#     ps += [Protein() for x in df['id']]
#     uniprot.fetch_fasta(ps)
#     uniprot.fetch_domains(ps)
#     LOGGER.debug(f'Obtained {len(ps)} total proteins')
#
#     ps = list(filter(
#         lambda _p: domain_name in _p.children,
#         ps))
#     LOGGER.debug(f'Filtered to {len(ps)} proteins with {domain_name} domain')
#     for p in ps:
#         p.children = {domain_name: p.children[domain_name]}
#     for p in ps:
#         extract_uniprot_domains(p)
#     ps = list(unduplicate(ps, domain_name, seeds))
#     LOGGER.debug(f'Unduplicated to {len(ps)} proteins')
#
#     id2seed = {row.id: row.UniProt_ID for _, row in df.iterrows()}
#     identity_and_coverage(ps, id2seed, domain_name)
#     LOGGER.debug(f'Added identity and coverage to `data` field')
#
#     ps = filter(
#         lambda _p: {'Seed', 'Ident', 'Coverage'}.issubset(
#             _p.children[domain_name].meta),
#         ps
#     )
#
#     @curry
#     def get_data_field(prot, field_name):
#         return prot.children[domain_name].meta[field_name]
#
#     lower, upper = identity_bounds
#     ps = filter(
#         lambda _p: lower <= get_data_field(_p, 'Ident') <= upper,
#         ps)
#     ps = filter(
#         lambda _p: get_data_field(_p, 'Coverage') >= min_coverage,
#         ps)
#
#     ps = list(ps)
#     LOGGER.debug(f'Filtered to {len(ps)} records by identity and coverage')
#     groups = groupby(get_data_field(field_name='Seed'), ps)
#     LOGGER.debug(f'Obtained {len(groups)} seed-groups')
#
#     num = num or len(df)
#     for k in groups:
#         if subsample and len(groups[k]) > num:
#             groups[k] = sample(groups[k], num)
#         groups[k] = sorted(groups[k], key=get_data_field(field_name='Ident'))[:num]
#     return groups, df
#
#
# @curry
# def domain_scan(
#         proteins: t.Iterable[Protein],
#         uniprot: t.Optional[UniProt] = None
# ) -> t.Iterator[Protein]:
#     if uniprot is None:
#         uniprot = UniProt()
#     proteins = uniprot.fetch_domains(list(proteins))
#     return (p for p in proteins if p.children)
#
#
# if __name__ == '__main__':
#     raise RuntimeError
