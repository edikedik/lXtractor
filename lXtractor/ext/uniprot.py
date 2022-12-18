import logging
import operator as op
import typing as t
from urllib.parse import urlencode

from lXtractor.util.io import fetch_text, fetch_chunks

T = t.TypeVar('T')
LOGGER = logging.getLogger(__name__)


# class UniProt:
#     """
#     Class to retrieve and parse UniProt data.
#     Methods accept ``Protein`` objects, fetch data,
#     and fill in corresponding fields in the ``Protein`` objects.
#     Thus, each method is "dirty" from the functional perspective,
#     changing the internal state of a received mutable argument.
#
#     Each method uses ``uniprot_id`` attribute to get the database accessions.
#     It's assumed that each ``Protein`` has this attribute filled in.
#     """
#
#     def __init__(
#             self, max_retries: int = 10,
#             num_threads: t.Optional[int] = None,
#             chunk_size: int = 1000,
#             verbose: bool = False,
#     ):
#         """
#         :param max_retries: a maximum number of fetching attempts.
#         :param num_threads: a number of threads for the ``ThreadPoolExecutor``.
#         :param chunk_size: a number of entries processed by a single thread.
#         :param verbose: progress bar.
#         """
#         self.max_retries = max_retries
#         self.num_threads = num_threads
#         self.chunk_size = chunk_size
#         self.verbose = verbose
#
#     @staticmethod
#     def _parse_domains(s: T) -> t.Union[T, t.List[t.Tuple[int, int, str]]]:
#         def parse_one(l: str):
#             try:
#                 boundaries = re.findall('\d+\.\.\d+', l)
#                 start, stop = map(int, boundaries[0].split('..'))
#                 name = re.findall(r'/note=\"([\w\s]+)\"', l)[0]
#                 return start, stop, name
#             except IndexError:
#                 return np.nan
#
#         if not isinstance(s, str):
#             return s
#
#         return list(map(parse_one, filter(bool, s.split('DOMAIN'))))
#
#     @staticmethod
#     def _check_input(proteins: t.Iterable[Chain]) -> None:
#         for p in proteins:
#             if p.uniprot_id is None:
#                 raise MissingData(f'Protein {p.id} misses UniProt ID')
#
#     def fetch_proteins_data(
#             self, proteins: t.Sequence[Chain],
#             overwrite: bool = False,
#             complement: bool = True,
#             keep_expected_if_any: bool = True,
#             base_fields: str = 'accession,id,sequence,ft_domain',
#             meta_fields: t.Optional[str] = None,
#             **kwargs
#     ) -> t.Tuple[t.List[Chain], pd.DataFrame]:
#         """
#         Fetches data from UniProt and populates `Protein` instances.
#
#         By default, fetches domain boundaries and populates :attr:`Protein.domains`
#             with relevant subsequences.
#
#         :param proteins: A collection of ``Protein`` objects.
#         :param overwrite: Overwrite existing data.
#         :param complement: Complement existing data.
#         :param base_fields: Do not change this value.
#         :param meta_fields: Additional ","-separated field names
#             (see ``base.MetaColumns`` for available column names).
#         :param expected_domains: Expected domain names.
#             If provided, initialize only these ``Domain`` instances.
#         :return: (a list of populated proteins, a DataFrame as fetched from UniProt)
#         """
#
#         def fetcher(acc: t.Iterable[str]) -> pd.DataFrame:
#             df = self.fetch_tsv(acc, fields=fields, **kwargs)
#             num_ids = len(df['accession'].unique())
#             LOGGER.debug(f'Obtained {len(df)} rows for {num_ids} IDs')
#             return df
#
#         def get_remaining(
#                 fetched: pd.DataFrame,
#                 remaining: t.Sequence[str]
#         ) -> t.List[str]:
#             remaining_ = list(set(remaining) - set(fetched['accession']))
#             LOGGER.debug(f'Fetcher has {len(remaining_)} IDs to fetch')
#             return remaining_
#
#         def check_duplicates(fetched: pd.DataFrame):
#             idx = fetched['accession'].duplicated()
#             duplicates = fetched.loc[idx, 'accession']
#             if len(duplicates):
#                 LOGGER.warning(
#                     f'Found duplicated IDs during metadata fetching: {list(duplicates)}. '
#                     f'Excluding these from metadata to avoid ambiguity')
#             return fetched[~idx]
#
#         def populate_meta(p: Chain, m: pd.Series) -> None:
#             meta = [(k, m[k]) for k in meta_fields.split(',')]
#             if p.metadata is not None:
#                 if complement:
#                     p.metadata += list(filterfalse(lambda x: x in meta, meta))
#                 else:
#                     if overwrite:
#                         p.metadata = meta
#             else:
#                 p.metadata = meta
#             return
#
#         def populate_seq(p: Chain, m: pd.Series) -> None:
#             if p.uniprot_seq is not None and not overwrite:
#                 return
#             seq_id = f"{m['id']}|{m['accession']}"
#             seq_rec = SeqRec(Seq(m['sequence']), seq_id, seq_id, seq_id)
#             p.uniprot_seq = seq_rec
#             return
#
#         def populate_dom(p: Chain, dom_info: t.Tuple[int, int, str]) -> None:
#             # Populate, but don't extract any data yet.
#             start, end, name = dom_info
#             dom = p.spawn_child(start, end, name, keep=overwrite or name not in p.children)
#             if not overwrite and complement and name in p.children:
#                 p.children[name].start = start
#                 p.children[name].end = end
#                 if dom.uniprot_seq is not None:
#                     p.children[name].uniprot_seq = dom.uniprot_seq
#             return
#
#         fields = base_fields
#         if meta_fields:
#             fields = f'{base_fields},{meta_fields}'
#
#         LOGGER.info(f'Attempting {len(fields.split(","))} metadata fields '
#                     f'for {len(proteins)} proteins')
#
#         # Fetch and join `DataFrame`s
#         results = self._fetch(proteins, fetcher, get_remaining, 'Fetching protein data')
#         df = check_duplicates(pd.concat(results))
#
#         # Group proteins based on UniProt ID
#         groups = groupby(lambda p: p.uniprot_id, proteins)
#
#         for _, row in df.iterrows():
#             acc = row['accession']
#             if acc not in groups:
#                 LOGGER.warning(f'Fetched {acc}, but it was not in queries')
#                 continue
#             group = groups[acc]
#             domains = row['ft_domain']
#             if isinstance(domains, str):
#                 domains = self._parse_domains(domains)
#             else:
#                 domains = []
#             for prot in group:
#                 populate_seq(prot, row)
#                 if meta_fields:
#                     populate_meta(prot, row)
#                 for d in domains:
#                     if keep_expected_if_any:
#                         if prot.expected_domains:
#                             if d[-1] in prot.expected_domains:
#                                 LOGGER.debug(f'{prot.id}: Keeping expected domain {d[-1]}')
#                                 populate_dom(prot, d)
#                             else:
#                                 LOGGER.debug(f'{prot.id}: Omitting unexpected domain {d[-1]}')
#                     else:
#                         populate_dom(prot, d)
#
#         return list(flatten(groups.values())), df
#
#     def fetch_tsv(
#             self, accessions: t.Iterable[str],
#             fields: str = 'accession,id,sequence,ft_domain',
#             explode_domains: bool = False,
#             **kwargs
#     ) -> pd.DataFrame:
#
#         def wrap_into_df(result: str):
#             df = pd.read_csv(StringIO(result), sep='\t', skiprows=1, names=fields.split(','))
#             if explode_domains and 'ft_domain' in fields:
#                 df['DomainTmp'] = df['ft_domain'].apply(self._parse_domains)
#                 df = df.explode('DomainTmp')
#                 idx = ~df['DomainTmp'].isna()
#                 df.loc[idx, 'DomainStart'] = df.loc[idx, 'DomainTmp'].apply(lambda x: x[0][0])
#                 df.loc[idx, 'DomainEnd'] = df.loc[idx, 'DomainTmp'].apply(lambda x: x[0][1])
#                 df.loc[idx, 'DomainName'] = df.loc[idx, 'DomainTmp'].apply(lambda x: x[1])
#                 df.drop(columns=['DomainTmp'], inplace=True)
#             return df
#
#         res = fetch_uniprot(accessions, fmt='tsv', num_threads=self.num_threads,
#                             chunk_size=self.chunk_size, fields=fields, **kwargs)
#
#         if not res:
#             raise MissingData(f'Fetched nothing! fields={fields}')
#
#         return wrap_into_df(res)
#
#     def fetch_fasta(self, proteins: t.Collection[Chain]) -> t.List[Chain]:
#         """
#         Fetch fasta sequences for each of the given collection of proteins.
#         Write the fetched sequences in terms of ``SeqRecord`` objects
#         into ``uniprot_seq`` attribute of each ``Protein``.
#
#         :param proteins: A collection of ``Protein`` objects.
#         """
#
#         def get_id(seq: SeqRec) -> str:
#             return seq.id.split('|')[1]
#
#         def fetcher(acc: t.Iterable[str]) -> t.List:
#             res = fetch_uniprot(acc, num_threads=self.num_threads, chunk_size=self.chunk_size)
#             res = list(SeqIO.parse(StringIO(res), 'fasta'))
#             LOGGER.debug(f'Fasta fetcher fetched {len(res)} sequences')
#             return res
#
#         def get_remaining(
#                 fetched: t.List,
#                 remaining: t.Sequence[str]
#         ) -> t.List[str]:
#             current_ids = set(map(get_id, fetched))
#             remaining_ = list(set(remaining) - current_ids)
#             LOGGER.debug(f'Fasta fetcher has {len(remaining_)} IDs to fetch')
#             return remaining_
#
#         # Fetch the FASTA sequences
#         results = self._fetch(proteins, fetcher, get_remaining, 'Fetching FASTAs')
#         # Split proteins into groups based on UniProt ID
#         groups = groupby(lambda p: p.uniprot_id, proteins)
#         # Fill `uniprot_seq` attribute of protein within
#         # groups of successfully fetched IDs
#         for seq_rec in flatten(results):
#             seq_id = get_id(seq_rec)
#             group = groups[seq_id]
#             for prot in group:
#                 prot.uniprot_seq = seq_rec
#
#         # Return all the received proteins
#         return list(flatten(groups.values()))
#
#     def _fetch(
#             self, proteins: t.Collection[Chain],
#             fetcher: _Fetcher, getter: _Getter,
#             desc: t.Optional[str],
#             alt_ids: t.Optional[t.Sequence[str]] = None
#     ) -> t.List[T]:
#         # Check whether all proteins have UniProt IDs
#         self._check_input(proteins)
#
#         # Prepare unique UniProt accessions
#         acc = alt_ids or set(p.uniprot_id for p in proteins)
#         LOGGER.debug(f'Expecting to fetch {len(acc)} UniProt IDs')
#
#         # Try fetching IDs using `fetcher` until the `getter` outputs
#         # an emtpy list of remaining elements to fetch
#         results, remaining = try_fetching_until(
#             acc,
#             fetcher=fetcher,
#             get_remaining=getter,
#             max_trials=self.max_retries,
#             desc=desc if self.verbose else None)
#
#         # Warn about the remaining IDs, if any
#         if remaining:
#             LOGGER.warning(f'Failed to fetch {len(remaining)}: {remaining}')
#         return results
#
#     def fetch_gff(self, proteins: t.Collection[Chain]) -> str:
#         """
#         Fetch GFF entries. Joins the fetched results in a single GFF file.
#         Returns the latter as a string.
#
#         :param proteins: A collection of ``Protein`` objects.
#         """
#
#         def fetcher(acc: t.Iterable[str]) -> str:
#             res = fetch_uniprot(acc, num_threads=self.num_threads,
#                                 chunk_size=self.chunk_size, fmt='gff')
#             LOGGER.debug(f'GFF fetcher fetched {len(res)} lines')
#             return res
#
#         def get_remaining(
#                 fetched: str,
#                 remaining: t.Sequence[str]
#         ) -> t.List[str]:
#             lines = fetched.split('\n')
#             lines = filter(
#                 lambda line: line and line != '\n' and not line.startswith('#'),
#                 lines)
#             current_ids = {line.split()[0] for line in lines}
#             remaining_ = list(set(remaining) - current_ids)
#             LOGGER.debug(f'GFF fetcher has {len(remaining_)} IDs to fetch')
#             return remaining_
#
#         results = self._fetch(proteins, fetcher, get_remaining, 'Fetching GFFs')
#         LOGGER.debug(f'Got {len(results)} chunks of GFF fetching results')
#
#         return "\n".join(results)
#
#     def fetch_orthologs(
#             self,
#             proteins: t.Sequence[Chain],
#             output_columns: t.Sequence[str] = (
#                     'id', 'reviewed', 'existence', 'organism')
#     ) -> pd.DataFrame:
#
#         @curry
#         def fetcher(
#                 acc: t.Iterable[str], column_names,
#                 from_db='ACC+ID', id_col='UniProt_ID'
#         ) -> pd.DataFrame:
#             results = fetch_uniprot(
#                 acc, from_db=from_db, fmt='tab', num_threads=self.num_threads,
#                 columns=','.join(column_names))
#             res = pd.read_csv(
#                 StringIO(results), sep='\t', skiprows=1,
#                 names=column_names + [id_col])
#             return res
#
#         @curry
#         def get_remaining(
#                 fetched: pd.DataFrame,
#                 remaining: t.Sequence[str],
#                 id_col: str,
#         ) -> t.List[str]:
#             remaining_ = list(set(remaining) - set(fetched[id_col]))
#             return remaining_
#
#         genes = pd.concat(
#             self._fetch(proteins, fetcher(column_names=['genes(PREFERRED)']), get_remaining(id_col='UniProt_ID'),
#                         'Fetching gene names')
#             ).rename(
#             columns={'genes(PREFERRED)': 'gene'}
#         ).dropna()
#
#         ortholist = pd.concat(
#             self._fetch(proteins, fetcher(
#                 column_names=list(output_columns),
#                 from_db='GENENAME', id_col='gene'), get_remaining(id_col='gene'), 'Fetching a list of orthologs',
#                         alt_ids=list(genes['gene']))
#         ).merge(
#             genes, on='gene'
#         )
#
#         return ortholist


def fetch_uniprot(
        acc: t.Iterable[str], fmt: str = 'fasta',
        chunk_size: int = 100, fields: t.Optional[str] = None,
        **kwargs
) -> str:
    """
    An interface for the UniProt's search.

    Base URL: https://rest.uniprot.org/uniprotkb/stream

    Available DB identifiers: https://www.uniprot.org/help/api_idmapping

    Will use :func:`fetch_chunks lXtractor.util.io.fetch_chunks` internally.

    :param acc: an iterable over valid UniProt accessions.
    :param fmt: download format (e.g., "fasta", "gff", "tab", ...).
    :param chunk_size: how many accessions to download in a chunk.
    :param fields: if the ``fmt`` is "tsv", must be provided
        to specify which data columns to fetch.
    :param kwargs: passed to :func:`fetch_chunks lXtractor.util.io.fetch_chunks`.
    :return: the 'utf-8' encoded results as a single chunk of text.
    """

    url = 'https://rest.uniprot.org/uniprotkb/stream'

    def fetch_chunk(chunk: t.Iterable[str]):
        params = {
            'format': fmt,
            'query': ' OR '.join(map(lambda a: f'accession:{a}', chunk))
        }
        if fmt == 'tsv' and fields is not None:
            params['fields'] = fields
        return fetch_text(url, params=urlencode(params).encode('utf-8'))

    results = fetch_chunks(acc, fetch_chunk, chunk_size, **kwargs)

    return "".join(map(op.itemgetter(1), results))


if __name__ == '__main__':
    raise ValueError
