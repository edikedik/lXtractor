"""
This module comprises utils for fetching data from UniProt and parsing the results.
The core functionality is presented via sinlge ``UniProt`` class.
"""
import logging
import typing as t
from copy import deepcopy
from io import StringIO
from urllib.parse import urlencode

import pandas as pd
from Bio import SeqIO
from more_itertools import flatten
from toolz import groupby, curry

from .base import SeqRec, MissingData, Domain
from .protein import Protein
from .utils import download_text, fetch_iterable, try_fetching_until, _Fetcher, _Getter

T = t.TypeVar('T')
LOGGER = logging.getLogger(__name__)


# TODO: implement dir support
# TODO: base calculations on IDs, not on `Protein` objects ==> make it useful by itself


class UniProt:
    """
    Class to retreive and parse UniProt data.
    Methods accept ``Protein`` objects, fetch data,
    and fill in corresponding fields in the ``Protein`` objects.
    Thus, each method is "dirty" from the functional perspective,
    changing the internal state of a received mutable argument.

    Each method uses ``uniprot_id`` attribute to get the database accessions.
    It's assumed that each ``Protein`` has this attribute filled in.
    """

    # TODO: maybe expose `chunk_size` parameter

    def __init__(
            self, max_retries: int = 10,
            num_threads: t.Optional[int] = None,
            verbose: bool = True):
        """
        :param max_retries: a maximum number of fetching attempts.
        :param num_threads: a number of threads for the ``ThreadPoolExecutor``.
        """
        self.max_retries = max_retries
        self.num_threads = num_threads
        self.verbose = verbose

    @staticmethod
    def _check_input(proteins: t.Iterable[Protein]) -> None:
        for p in proteins:
            if p.uniprot_id is None:
                raise MissingData(f'Protein {p.id} misses UniProt ID')

    def _do_fetch(
            self, proteins: t.Collection[Protein],
            fetcher: _Fetcher, getter: _Getter,
            desc: t.Optional[str],
            alt_ids: t.Optional[t.Sequence[str]] = None
    ) -> t.List[T]:
        # Check whether all proteins have UniProt IDs
        self._check_input(proteins)

        # Prepare unique UniProt accessions
        acc = alt_ids or set(p.uniprot_id for p in proteins)
        LOGGER.debug(f'Expecting to fetch {len(acc)} UniProt IDs')

        # Try fetching IDs using `fetcher` until the `getter` outputs
        # an emtpy list of remaining elements to fetch
        results, remaining = try_fetching_until(
            acc,
            fetcher=fetcher,
            get_remaining=getter,
            max_trials=self.max_retries,
            desc=desc if self.verbose else None)

        # Warn about the remaining IDs, if any
        if remaining:
            LOGGER.warning(f'Failed to fetch {len(remaining)}: {remaining}')
        return results

    def fetch_fasta(self, proteins: t.Collection[Protein]) -> t.List[Protein]:
        """
        Fetch fasta sequences for each of the given collection of proteins.
        Write the fetched sequences in terms of ``SeqRecord`` objects
        into ``uniprot_seq`` attribute of a each ``Protein``.

        :param proteins: A collection of ``Protein`` objects.
        """

        def get_id(seq: SeqRec) -> str:
            return seq.id.split('|')[1]

        def fetcher(acc: t.Iterable[str]) -> t.List[SeqRec]:
            res = fetch_uniprot(acc, num_threads=self.num_threads)
            res = list(SeqIO.parse(StringIO(res), 'fasta'))
            LOGGER.debug(f'Fasta fetcher fetched {len(res)} sequences')
            return res

        def get_remaining(
                fetched: t.List[SeqRec],
                remaining: t.Sequence[str]
        ) -> t.List[str]:
            current_ids = set(map(get_id, fetched))
            remaining_ = list(set(remaining) - current_ids)
            LOGGER.debug(f'Fasta fetcher has {len(remaining_)} IDs to fetch')
            return remaining_

        # Fetch the FASTA sequences
        results = self._do_fetch(
            proteins, fetcher, get_remaining,
            'Fetching FASTAs')
        # Split proteins into groups based on UniProt ID
        groups = groupby(lambda p: p.uniprot_id, proteins)
        # Fill `uniprot_seq` attribute of protein within
        # groups of successfully fetched IDs
        for seq_rec in flatten(results):
            seq_id = get_id(seq_rec)
            group = groups[seq_id]
            for prot in group:
                prot.uniprot_seq = seq_rec

        # Return back all the received proteins
        return list(flatten(groups.values()))

    def fetch_gff(self, proteins: t.Collection[Protein]) -> str:
        """
        Fetch GFF entries. Joins the fetched results in a single GFF file.
        Returns the latter as a string.

        :param proteins: A collection of ``Protein`` objects.
        """

        def fetcher(acc: t.Iterable[str]) -> str:
            res = fetch_uniprot(acc, fmt='gff')
            LOGGER.debug(f'GFF fetcher fetched {len(res)} lines')
            return res

        def get_remaining(
                fetched: str,
                remaining: t.Sequence[str]
        ) -> t.List[str]:
            lines = fetched.split('\n')
            lines = filter(
                lambda line: line and line != '\n' and not line.startswith('#'),
                lines)
            current_ids = {line.split()[0] for line in lines}
            remaining_ = list(set(remaining) - current_ids)
            LOGGER.debug(f'GFF fetcher has {len(remaining_)} IDs to fetch')
            return remaining_

        results = self._do_fetch(
            proteins, fetcher, get_remaining, 'Fetching GFFs')
        LOGGER.debug(f'Got {len(results)} chunks of GFF fetching results')

        return "\n".join(results)

    def fetch_domains(
            self, proteins: t.Sequence[Protein],
    ) -> t.List[Protein]:
        """
        Calls ``fetch_gff`` method and filters the GFF output to lines
        specifying the domain name and boundaries.
        Wraps each parsed domain into a ``Segment`` object.
        A dict of ``Segment`` objects is then saved to ``domains``
        attribute of each ``Protein``.

        :param proteins: a collection of ``Protein`` objects.
        """

        def parse_domain_line(line: str) -> Domain:
            line_split = line.split()
            start, end = map(int, line_split[3:5])
            uniprot_id = line_split[0]
            domain_name = line.split('Note=')[1].split(';')[0].rstrip()
            return Domain(start, end, domain_name, uniprot_id)

        def parse_output(gff: str) -> t.Dict[str, t.List[Domain]]:
            lines = gff.split('\n')
            domain_lines = filter(lambda line: 'Domain' in line, lines)
            segments = map(parse_domain_line, domain_lines)
            return groupby(lambda seg: seg.parent_name, segments)

        _gff = self.fetch_gff(proteins)
        LOGGER.debug(f'Fetched {len(_gff)} lines of GFF')
        grouped_prots = groupby(lambda p: p.uniprot_id, proteins)
        grouped_domains = parse_output(_gff)
        LOGGER.debug(f'Obtained {len(grouped_domains)} domain groups '
                     f'from GFF output (one per UniProt ID)')

        for _id, domains in grouped_domains.items():
            try:
                group = grouped_prots[_id]
            # Sometimes, for some reason, UniProt seem to return
            # IDs which were not requested
            except KeyError:
                LOGGER.error(
                    f'Failed to find group {_id}')
                continue
            for prot in group:
                if prot.expected_domains:
                    _domains = filter(
                        lambda x: any(
                            name in x.name for name in prot.expected_domains),
                        domains)
                else:
                    _domains = domains
                # Domains are mutable objects, and this fact is exploited
                # by the the extraction process later.
                # Thus, here we copy objects explicitly.
                prot.domains.update(
                    {d.name: deepcopy(d) for d in _domains})

        return list(flatten(grouped_prots.values()))

    def fetch_meta(
            self, proteins: t.Sequence[Protein],
            fields: t.Sequence[str]
    ) -> t.Tuple[t.List[Protein], pd.DataFrame]:
        """
        Fetch any UniProt field in tabular format
        and write the results as ``metadata``.

        :param proteins: A collection of ``Protein`` objects.
        :param fields: a list of valid fields
            (see ``base.MetaColumns`` for available column names).
        :return: a list of proteins with ``metadata`` attribute
            containing the desired UniProt fields.
        """
        columns = ','.join(fields)
        LOGGER.info(f'Will try fetching {len(fields)} metadata fields '
                    f'for {len(proteins)} proteins')
        column_names = list(fields) + ['UniProt_ID']

        def fetcher(acc: t.Iterable[str]) -> pd.DataFrame:
            results = fetch_uniprot(acc, fmt='tab', columns=columns)
            res = pd.read_csv(
                StringIO(results), sep='\t', skiprows=1, names=column_names)
            LOGGER.debug(f'Metadata fetcher obtained metadata for {len(res)} IDs')
            return res

        def get_remaining(
                fetched: pd.DataFrame,
                remaining: t.Sequence[str]
        ) -> t.List[str]:
            remaining_ = list(set(remaining) - set(fetched['UniProt_ID']))
            LOGGER.debug(f'Metadata fetcher has {len(remaining_)} IDs to fetch')
            return remaining_

        # Fetch and join `DataFrame` s with metadata
        _metas = self._do_fetch(
            proteins, fetcher, get_remaining,
            'Fetching metadata')
        df = pd.concat(_metas)
        LOGGER.debug(f'Joined {len(_metas)} metadata chunks into a '
                     f'`DataFrame` with {len(df)} records')

        # Sanity check whether fetcher has returned duplicates
        duplicates_idx = df['UniProt_ID'].duplicated()
        duplicates = df.loc[duplicates_idx, 'UniProt_ID']
        if len(duplicates):
            LOGGER.warning(
                f'Found duplicated IDs during metadata fetching: {list(duplicates)}. '
                f'Excluding these from metadata to avoid ambiguity')
        df = df[~duplicates_idx]

        # Group proteins based on UniProt ID
        groups = groupby(lambda p: p.uniprot_id, proteins)

        # Each row in the `DataFrame` contains metadata corresponding to a UniProt ID
        # Here we put these metadata into a `metadata` attribute of each `Protein`
        for _, row in df.iterrows():
            protein_meta = list(zip(column_names[:-1], row))
            if row['UniProt_ID'] not in groups:
                LOGGER.error(f'Fetched {row["UniProt_ID"]}, '
                             f'but it was not in queries')
                continue
            group = groups[row['UniProt_ID']]
            for prot in group:
                # Append to existing metadata
                if prot.metadata is not None:
                    if not isinstance(prot.metadata, list):
                        LOGGER.error(
                            f'Expected to have a list object in metadata '
                            f'of protein {prot.id}')
                        continue
                    prot.metadata += protein_meta
                # Create new list with metadata
                else:
                    prot.metadata = protein_meta

        return list(flatten(groups.values())), df

    def fetch_orthologs(
            self,
            proteins: t.Sequence[Protein],
            output_columns: t.Sequence[str] = (
                    'id', 'reviewed', 'existence', 'organism')
    ) -> pd.DataFrame:

        @curry
        def fetcher(
                acc: t.Iterable[str], column_names,
                from_db='ACC+ID', id_col='UniProt_ID'
        ) -> pd.DataFrame:
            results = fetch_uniprot(
                acc, from_db=from_db, fmt='tab',
                columns=','.join(column_names))
            res = pd.read_csv(
                StringIO(results), sep='\t', skiprows=1,
                names=column_names + [id_col])
            return res

        @curry
        def get_remaining(
                fetched: pd.DataFrame,
                remaining: t.Sequence[str],
                id_col: str,
        ) -> t.List[str]:
            remaining_ = list(set(remaining) - set(fetched[id_col]))
            return remaining_

        genes = pd.concat(self._do_fetch(
            proteins,
            fetcher(column_names=['genes(PREFERRED)']),
            get_remaining(id_col='UniProt_ID'),
            'Fetching gene names')
        ).rename(
            columns={'genes(PREFERRED)': 'gene'}
        ).dropna()

        ortholist = pd.concat(
            self._do_fetch(
                proteins,
                fetcher(
                    column_names=list(output_columns),
                    from_db='GENENAME', id_col='gene'),
                get_remaining(id_col='gene'),
                'Fetching a list of orthologs',
                alt_ids=list(genes['gene']))
        ).merge(
            genes, on='gene'
        )

        return ortholist


def fetch_uniprot(
        acc: t.Iterable[str], fmt: str = 'fasta', from_db: str = 'ACC+ID',
        to_db: str = 'ACC', chunk_size: int = 100, columns: t.Optional[str] = None,
        num_threads: t.Optional[int] = None, verbose: bool = False
) -> str:
    """
    An interface for the UniProt for programmatic access to the
    `Retrieve ID/Mapping <https://www.uniprot.org/uploadlists/>`_.

    Available DB identifiers: https://www.uniprot.org/help/api_idmapping

    :param acc: an iterable over valid UniProt accessions.
    :param fmt: download format (e.g., "fasta", "gff", "tab", ...).
    :param from_db: database to map from.
    :param to_db: database to map to.
    :param chunk_size: how many accessions to download in a chunk.
    :param columns: if the ``fmt`` is "tab", must be provided
        to specify which data columns to fetch.
    :param num_threads: a number of threads for the ``ThreadPoolExecutor``.
    :param verbose: expose the progress bar.
    :return: the 'utf-8' encoded results as a single chunk of text.
    """
    url = 'https://www.uniprot.org/uploadlists/'

    def fetch_chunk(chunk: t.Iterable[str]):
        params = {
            'from': from_db,
            'to': to_db,
            'format': fmt,
            'query': ' '.join(chunk),
        }
        if fmt == 'tab' and columns is not None:
            params['columns'] = columns
        return download_text(url, params=urlencode(params).encode('utf-8'))

    results = fetch_iterable(
        acc, fetch_chunk, chunk_size,
        num_threads=num_threads,
        verbose=verbose)

    return "".join(results)


if __name__ == '__main__':
    raise ValueError
