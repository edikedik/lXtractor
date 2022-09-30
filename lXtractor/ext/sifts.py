"""
Contains utils allowing to benefit from SIFTS database `UniProt`-`PDB` mapping.

Namely, the `SIFTS` class is build around the file `uniprot_segments_observed.csv.gz`.
The latter contains segment-wise mapping between UniProt sequences and continuous
corresponding regions in PDB structures, and allows us to:

#.  Cross-reference PDB and UniProt databases (e.g., which structures are available
    for a UniProt "PXXXXXX" accession?)
#.  Map between sequence numbering schemes.

"""
import logging
import typing as t
from importlib import resources
from itertools import chain, starmap
from pathlib import Path

import joblib
import pandas as pd
from more_itertools import unzip

from lXtractor.ext import resources as local
from lXtractor.core.base import AbstractResource
from lXtractor.core.segment import Segment
from lXtractor.core.exceptions import MissingData, LengthMismatch, OverlapError
from lXtractor.util.io import download_to_file
from lXtractor.util.misc import col2col

LOGGER = logging.getLogger(__name__)
SIFTS_FTP = 'ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/uniprot_segments_observed.csv.gz'
RAW_SEGMENTS = 'uniprot_segments_observed.csv.gz'
ID_MAPPING = 'id_mapping.joblib'
DF_SIFTS = 'sifts.tsv'


try:
    with resources.path(local, RAW_SEGMENTS) as __path:
        if not __path.exists():
            raise FileNotFoundError
        SIFTS_PATH = __path

except FileNotFoundError:

    with resources.path(local, '') as __path:
        SIFTS_PATH = Path(str(__path)) / RAW_SEGMENTS

    if not SIFTS_PATH.exists():
        LOGGER.info(
            f'Found no SIFTS in resources. '
            f'Starting to download SIFTS from {SIFTS_FTP}')
        SIFTS_PATH = download_to_file(SIFTS_FTP, SIFTS_PATH)
        LOGGER.info(f'Downloaded sifts to {SIFTS_PATH}')

SIFTS_RENAMES = (
    ('PDB', 'PDB'),
    ('CHAIN', 'Chain'),
    ('SP_PRIMARY', 'UniProt_ID'),
    ('PDB_BEG', 'PDB_start'),
    ('PDB_END', 'PDB_end'),
    ('SP_BEG', 'UniProt_start'),
    ('SP_END', 'UniProt_end')
)


def _soft_load_resource(name: str):
    with resources.path(local, name) as path:
        if not path.exists():
            LOGGER.debug(f'{path} does not exist')
            return None
        suffix = path.suffix
        if suffix in ['.tsv', '.gz']:
            return pd.read_csv(path, sep='\t')
        if suffix in ['.joblib']:
            return joblib.load(path)
        LOGGER.warning(f'Failed to load {path}')
        return None


class Mapping(dict):
    """
    A ``dict`` subclass with explicit IDs of keys/values sources.
    """

    def __init__(self, id_from: str, id_to: str, *args, **kwargs):
        """
        :param id_from: ID of an objects a mapping is from (keys).
        :param id_to: ID of an object a mapping is to (values).
        :param args: passed to ``dict``.
        :param kwargs: passed to ``dict``.
        """
        super().__init__(*args, **kwargs)
        self.id_from = id_from
        self.id_to = id_to


class SIFTS(AbstractResource):
    """
    A class for manipulating `PDB`-`UniProt` mapping.
    It parses the file "uniprot_segments_observed" and stores
    it as ``pandas.DataFrame`` object, which is later use to
    obtain mappings.
    """

    def __init__(
            self, resource_path: t.Optional[Path] = None,
            resource_name: str = 'SIFTS',
            load_segments: bool = False,
            load_id_mapping: bool = True,
            df: t.Optional[pd.DataFrame] = None):
        """
        :param resource_path: a path to a file "uniprot_segments_observed".
            If not provided, will try finding this file in the ``resources`` module.
            If the latter fails will attempt fetching the mapping from the FTP server
            and storing it in the ``resources`` for later use.
        :param resource_name: the name of the resource.
        :param load_segments: load pre-parsed segment-level mapping
        :param load_id_mapping: load pre-parsed id mapping
        :param df: in case some specific parsing is needed, one can provide
            a ``DataFrame`` containing columns specified in the ``SIFTS_RENAMES`` constant.
        """
        if load_segments and df is None:
            self.df = _soft_load_resource(DF_SIFTS)
        else:
            self.df = df

        if load_id_mapping:
            self.id_mapping = _soft_load_resource(ID_MAPPING)
        else:
            self.id_mapping = (None if self.df is None else self._prepare_id_map(df))

        resource_path = resource_path or SIFTS_PATH
        self.renames = dict(SIFTS_RENAMES)
        super().__init__(resource_path, resource_name)

    def _prepare_id_map(self, df: pd.DataFrame):
        self.id_mapping = {
            **col2col(df, 'UniProt_ID', 'PDB_Chain'),
            **col2col(df, 'PDB_Chain', 'UniProt_ID'),
            **col2col(df, 'PDB', 'Chain'),
        }
        LOGGER.debug('Created mapping UniProt ID <-> PDB ID')

    def _store(self):
        with resources.path(local, '') as base:
            # Prepare paths
            base = Path(base)
            id_mapping_path = base / ID_MAPPING
            df_path = base / DF_SIFTS

            joblib.dump(self.id_mapping, id_mapping_path)
            LOGGER.debug(f'Saved ID mapping to {id_mapping_path}')
            self.df.to_csv(df_path, sep='\t', index=False)
            LOGGER.debug(f'Saved df to {df_path}')

    def read(self, overwrite: bool = True) -> pd.DataFrame:
        """
        The method how to read the initial file
        "uniprot_segments_observed" into memory.

        :param overwrite: overwrite existing ``df`` attribute.
        :return: pandas ``DataFrame`` object.
        """
        LOGGER.debug(f'Reading SIFTS from {self.path}')
        df = pd.read_csv(self.path, skiprows=1, low_memory=False)
        LOGGER.debug(f'Read `DataFrame` with {len(df)} records')
        if overwrite:
            self.df = df
        return df

    def parse(
            self, overwrite: bool = False,
            store_to_resources: bool = True) -> pd.DataFrame:
        """
        Prepare the resource to be used for mapping:

            - remove records with empty chains
            - select and rename key columns based on the
              ``SIFTS_RENAMES`` constant
            - create a `PDB_Chain` column to speed up the search.

        :param overwrite: write the results into the ``df`` attribute.
        :param store_to_resources: store parsed ``DataFrame`` and id mapping
            in resources for further simplified access
        :return: prepared `DataFrame`.
        """
        df = self.read(False)
        LOGGER.debug(f'Received {len(df)} records')

        df = df[
            list(self.renames)
        ].rename(
            columns=self.renames
        ).drop_duplicates()
        LOGGER.debug(f'Renamed and removed duplicates: records={len(df)}')

        df = df[~df.Chain.isna()]
        LOGGER.debug(f'Removed entries with empty chains: records={len(df)}')

        df['PDB'] = df['PDB'].str.upper()
        df['PDB_Chain'] = [
            f'{x[1]}:{x[2]}' for x in df[['PDB', 'Chain']].itertuples()]
        LOGGER.debug(f'Created PDB_Chain column')

        for col in ['PDB_start', 'PDB_end']:
            df[col] = ["".join(filter(lambda x: x.isdigit(), value)) for value in df[col]]
            df[col] = df[col].astype(int)
        LOGGER.debug('Digitized boundaries')

        sel = ['PDB_Chain'] + list(self.renames.values())
        df = df[sel].sort_values(
            ['UniProt_ID', 'PDB_Chain']
        )
        LOGGER.debug(f'Finished parsing df: records={len(df)}')

        if overwrite:
            self.df = df

        self._prepare_id_map(df)

        if store_to_resources:
            self._store()

        return df

    def dump(self, path: Path, **kwargs):
        """
        :param path: a valid writable path.
        :param kwargs: passed to ``DataFrame.to_csv()`` method.
        :return:
        """
        if self.df is not None:
            self.df.to_csv(path, **kwargs)
        raise RuntimeError('Nothing to dump')

    def fetch(self, url: str):
        # TODO: encapsulate downloading SIFTS in here
        raise NotImplementedError

    @staticmethod
    def _parse_obj_id(obj_id: str) -> str:
        if len(obj_id) == 6 and ':' in obj_id:
            LOGGER.debug(f'Assumed {obj_id} to be a PDB:Chain')
            sel_column = 'PDB_Chain'
        elif len(obj_id) == 4:
            LOGGER.debug(f'Assumed {obj_id} to be a PDB ID')
            sel_column = 'PDB'
        else:
            LOGGER.debug(f'Assumed {obj_id} to be a UniProt ID')
            sel_column = 'UniProt_ID'
        return sel_column

    def map_numbering(self, obj_id: str) -> t.Iterator[Mapping]:
        """
        Retrieve mappings associated with the ``obj_id``. Mapping example::

            1 -> 2
            2 -> 3
            3 -> None
            4 -> 4

        Above, a UniProt sequence maps to two segments of a PDB sequence (2-3 and 4).
        PDB sequence is always considered a subset of a corresponding UniProt sequence.
        Thus, any "holes" between continuous PDB segments are filled with ``None``.

        :param obj_id: a string value in three possible formats:

            1. "PDB ID:Chain ID"
            2. "PDB ID"
            3. "UniProt ID"

        :return: an iterator over the ``Mapping`` objects.
            These are "unidirectional", i.e., the ``Mapping`` is always from
            the UniProt numbering to the PDB numbering
            regardless of the ``obj_id`` nature.
        """
        sel_column = self._parse_obj_id(obj_id)

        if self.df is None:
            raise MissingData('No SIFTS df found; try calling .parse() first')

        sub = self.df[self.df[sel_column] == obj_id]
        LOGGER.debug(
            f'Subset SIFTS by {sel_column}={obj_id}, records={len(sub)}')

        if not len(sub):
            LOGGER.warning(f"Failed to find {obj_id} in SIFTS")
            return None

        groups = sub.groupby(['UniProt_ID', 'PDB_Chain'])
        group_ids, dfs = unzip(groups)
        segments = map(wrap_into_segments, dfs)
        mappings = starmap(map_segment_numbering, segments)
        yield from (
            Mapping(uni_id, pdb_chain, m)
            for (uni_id, pdb_chain), m in zip(group_ids, mappings))

    def map_id(self, _id: str) -> t.Optional[t.List[str]]:
        if self.id_mapping is None:
            self._prepare_id_map(self.df)
        if _id not in self.id_mapping:
            LOGGER.warning(f"Couldn't find {_id} in SIFTS")
            return None
        return self.id_mapping[_id]


def wrap_into_segments(df: pd.DataFrame) -> t.Tuple[t.List[Segment], t.List[Segment]]:
    """
    :param df: ``DataFrame`` -- a subset of a ``SIFTS`` ``DataFrame`` corresponding to a unique
        "UniProt_ID -- PDB_ID:Chain_ID" pair unto ``Segment`` objects
    :return: two lists with the same length (1) UniProt segments, and (2) PDB segments,
        where segments correspond to each other.
    """
    ids = df['UniProt_ID'].unique()
    if len(ids) > 1:
        raise ValueError(f'Expected single UniProt ID for group {df}, got {ids}')
    ids = df['PDB_Chain'].unique()
    if len(ids) > 1:
        raise ValueError(f'Expected single PDB chain for group {df}, got {ids}')

    uniprot_segments, pdb_segments = map(list, unzip(
        (Segment(row.UniProt_start, row.UniProt_end),
         Segment(row.PDB_start, row.PDB_end))
        for _, row in df.iterrows()))

    return uniprot_segments, pdb_segments


def map_segment_numbering(
        segments_from: t.Sequence[Segment],
        segments_to: t.Sequence[Segment],
) -> t.Iterator[t.Tuple[int, int]]:
    """
    Create a continuous mapping between the numberings of two segment collections.
    Collections must contain the same number of equal length non-overlapping ``Segment`` objects.
    Segments in the ``segments_from`` collection are considered to span a continuous sequence,
    possibly interrupted due to discontinuities in a sequence represented by ``segments_to``'s segments.
    Hence, the segments in ``segments_from`` form continuous numbering over which numberings
    of ``segments_to`` segments are "stiched."

    :param segments_from: a sequence of segments to map from.
    :param segments_to: a sequence of segments to map to.
    :return: an iterable over (key, value) pairs. Keys correspond to numberings of
        the ``segments_from``, values -- to numberings of ``segments_to``.
    """
    if len(segments_to) != len(segments_from):
        raise LengthMismatch('Segment collections must be of the same length')
    for s1, s2 in zip(segments_from, segments_to):
        if len(s1) != len(s2):
            raise LengthMismatch(f'Lengths of segments must match. '
                                 f'Got len({s1})={len(s1)}, len({s2})={len(s2)}')
    for s1, s2 in zip(segments_from, segments_from[1:]):
        if s2.overlaps(s1):
            raise OverlapError(f'Segments {s1},{s2} in `segments_from` overlap')
    for s1, s2 in zip(segments_to, segments_to[1:]):
        if s2.overlaps(s1):
            raise OverlapError(f'Segments {s1},{s2} in `segments_to` overlap')

    hole_sizes = chain(
        ((s2.start - s1.end) for s1, s2 in zip(
            segments_to, segments_to[1:])),
        (0,))

    return zip(
        range(segments_from[0].start, segments_from[-1].end + 1),
        chain.from_iterable(
            chain(
                range(s.start, s.end + 1),
                (None for _ in range(s.end + 1, h + s.end)))
            for s, h in zip(segments_to, hole_sizes))
    )


if __name__ == '__main__':
    raise RuntimeError
