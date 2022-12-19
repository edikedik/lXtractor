"""
Contains utils allowing to benefit from SIFTS database `UniProt`-`PDBF;`
mapping.

Namely, the `SIFTS` class is build around the file
`uniprot_segments_observed.csv.gz`.
The latter contains segment-wise mapping between UniProt sequences
and continuous corresponding regions in PDB structures, and allows us to:

#. Cross-reference PDB and UniProt databases (e.g., which structures
are available for a UniProt "PXXXXXX" accession?)
#. Map between sequence numbering schemes.
"""
import json
import logging
import os
import typing as t
from collections import abc
from importlib import resources
from itertools import starmap
from pathlib import Path

import pandas as pd
from more_itertools import unzip

from lXtractor.core.base import AbstractResource
from lXtractor.core.exceptions import MissingData
from lXtractor.core.segment import Segment, map_segment_numbering
from lXtractor.ext import resources as local
from lXtractor.util.io import fetch_to_file
from lXtractor.util.misc import col2col

LOGGER = logging.getLogger(__name__)
SIFTS_FTP = (
    'ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv'
    '/uniprot_segments_observed.csv.gz'
)
RAW_SEGMENTS = 'uniprot_segments_observed.csv.gz'
ID_MAPPING = 'id_mapping.json'
DF_SIFTS = 'sifts.tsv'
RESOURCES = Path(__file__).parent / 'resources'

# TODO: Create a remote interface. (this may include segment-based mappings,
# although for many files it can be slow)


SIFTS_RENAMES = (
    ('PDB', 'PDB'),
    ('CHAIN', 'Chain'),
    ('SP_PRIMARY', 'UniProt_ID'),
    ('PDB_BEG', 'PDB_start'),
    ('PDB_END', 'PDB_end'),
    ('SP_BEG', 'UniProt_start'),
    ('SP_END', 'UniProt_end'),
)


def _soft_load_resource(name: str) -> pd.DataFrame | dict | None:
    with resources.path(local, name) as path:
        if not path.exists():
            LOGGER.debug(f'{path} does not exist')
            return None
        suffix = path.suffix
        if suffix in ['.tsv', '.gz']:
            return pd.read_csv(path, sep='\t')
        if suffix in ['.json']:
            with path.open(encoding='utf-8') as f:
                return json.load(f)
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
    A resource to segment-wise and ID mappings between UniProt and PDB.

    For a first-time usage, you'll need to call :meth:`fetch`
    to download and store the "uniprot_segments_observed" dataset.

    >>> sifts = SIFTS()
    >>> path = sifts.fetch()
    >>> path.name
    'uniprot_segments_observed.csv.gz'

    Next, :meth:`parse` will process the downloaded file to create
    and store the table with segments and ID mappings.

    (We pass ``overwrite=True`` for the doctest to work.
    It's not needed for the first setup).

    >>> df, mapping = sifts.parse(store_to_resources=True, overwrite=True)
    >>> isinstance(df, pd.DataFrame) and isinstance(mapping, dict)
    True
    >>> list(df.columns)[:4]
    ['PDB_Chain', 'PDB', 'Chain', 'UniProt_ID']
    >>> list(df.columns)[4:]
    ['PDB_start', 'PDB_end', 'UniProt_start', 'UniProt_end']

    Now that we parsed SIFTS segments data, we can use it to map IDs
    and numberings between UniProt and PDB.
    Let's reinitalize SIFTS to verify it loads locally stored resources

    >>> sifts = SIFTS(load_segments=True, load_id_mapping=True)
    >>> assert isinstance(sifts.df, pd.DataFrame)
    >>> assert isinstance(sifts.id_mapping, dict)

    SIFTS has three types of mappings stored:

        1) Between  UniProt and PDB Chains

        >>> sifts['P12931'][:4]
        ['1A07:A', '1A07:B', '1A08:A', '1A08:B']

        2) Between PDB Chains and UniProt IDs

        >>> sifts['1A07:A']
        ['P12931']

        3) Between PDB IDs and PDB Chains

        >>> sifts['1A07']
        ['A', 'B']


    The same types of keys are supported to obtain mappings between
    the numbering schemes.
    You'll get a generator yielding mappings from UniProt numbering
    to the PDB numbering.

    In these two cases, we'll get the mappings for each chain.

    >>> mappings = list(sifts.map_numbering('P12931'))
    >>> assert len(mappings) == len(sifts['P12931'])
    >>> mappings = list(sifts.map_numbering('1A07'))
    >>> assert len(mappings) == len(sifts['1A07']) == 2

    If we specify the chain, we get a single mapping.

    >>> m = next(sifts.map_numbering('1A07:A'))
    >>> list(m.items())[:2]
    [(145, 145), (146, 146)]

    """

    def __init__(
        self,
        resource_path: t.Optional[Path] = None,
        resource_name: str = 'SIFTS',
        load_segments: bool = False,
        load_id_mapping: bool = False,
    ):
        """
        :param resource_path: a path to a file "uniprot_segments_observed".
            If not provided, will try finding this file in the ``resources`` module.
            If the latter fails will attempt fetching the mapping from the FTP server
            and storing it in the ``resources`` for later use.
        :param resource_name: the name of the resource.
        :param load_segments: load pre-parsed segment-level mapping
        :param load_id_mapping: load pre-parsed id mapping
        """
        self.df = _soft_load_resource(DF_SIFTS) if load_segments else None

        if load_id_mapping:
            self.id_mapping = _soft_load_resource(ID_MAPPING)
        else:
            self.id_mapping = None if self.df is None else self._prepare_id_map(self.df)

        resource_path = resource_path or RESOURCES / RAW_SEGMENTS
        self.renames = dict(SIFTS_RENAMES)
        super().__init__(resource_path, resource_name)

    def _prepare_id_map(self, df: pd.DataFrame) -> dict[str, list[str]]:
        self.id_mapping = {
            **col2col(df, 'UniProt_ID', 'PDB_Chain'),
            **col2col(df, 'PDB_Chain', 'UniProt_ID'),
            **col2col(df, 'PDB', 'Chain'),
        }
        LOGGER.debug('Created mapping UniProt ID <-> PDB ID')
        return self.id_mapping

    def _store(self):
        with resources.path(local, '') as base:
            # Prepare paths
            base = Path(base)
            id_mapping_path = base / ID_MAPPING
            df_path = base / DF_SIFTS
            with id_mapping_path.open('w') as f:
                json.dump(self.id_mapping, f)
            LOGGER.debug(f'Saved ID mapping to {id_mapping_path}')
            self.df.to_csv(df_path, sep='\t', index=False)
            LOGGER.debug(f'Saved df to {df_path}')

    def __getitem__(self, item: str):
        return self.map_id(item)

    def read(self, overwrite: bool = True) -> pd.DataFrame:
        """
        The method reads the initial file "uniprot_segments_observed" into memory.

        To load parsed files, use :meth:`load`.

        :param overwrite: overwrite existing ``df`` attribute.
        :return: pandas ``DataFrame`` object.
        """
        try:
            LOGGER.debug(f'Reading SIFTS from {self.path}')
            df = pd.read_csv(self.path, skiprows=1, low_memory=False)
        except FileNotFoundError as e:
            raise MissingData(
                f'Missing file {self.path}. Use `fetch` do download this file.'
            ) from e
        LOGGER.debug(f'Read `DataFrame` with {len(df)} records')
        if overwrite:
            self.df = df
        return df

    @staticmethod
    def load() -> tuple[pd.DataFrame | None, dict[str, list[str]] | None]:
        """
        :return: Loaded segments df and name mapping or ``None`` if they don't exist.
        """

        return _soft_load_resource(DF_SIFTS), _soft_load_resource(ID_MAPPING)

    def parse(
        self,
        overwrite: bool = False,
        store_to_resources: bool = True,
        rm_raw: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Prepare the resource to be used for mapping:

            - remove records with empty chains.
            - select and rename key columns based on the ``SIFTS_RENAMES``
            constant.
            - create a `PDB_Chain` column to speed up the search.

        :param overwrite: Overwrite both :attr:`df` and existing id mapping
            and parsed segments.
        :param store_to_resources: Store parsed `DataFrame` and id mapping
            in resources for further simplified access.
        :param rm_raw: After parsing is finished, remove raw SIFTS download.
            **(!) If `store_to_resources` is ``False``, using SIFTS next
            time will require downloading "uniprot_segments_observed".**
        :return: prepared :class:`DataFrame` of Segment-wise mapping between
            UniProt and PDB sequences. Mapping between IDs will be stored
            in :attr:`id_mapping`.
        """
        if (RESOURCES / DF_SIFTS).exists() and (RESOURCES / ID_MAPPING).exists():
            if not overwrite and store_to_resources:
                raise RuntimeError(
                    'Resources was parsed. Pass overwrite if you wan\'t to overwrite.'
                )

        if not (RESOURCES / RAW_SEGMENTS).exists():
            raise MissingData(
                'Missing fetched SIFTS. Try calling `fetch` method first.'
            )

        df = self.read(False)
        LOGGER.debug(f'Received {len(df)} records')

        df = df[list(self.renames)].rename(columns=self.renames).drop_duplicates()
        LOGGER.debug(f'Renamed and removed duplicates: records={len(df)}')

        df = df[~df.Chain.isna()]
        LOGGER.debug(f'Removed entries with empty chains: records={len(df)}')

        df['PDB'] = df['PDB'].str.upper()
        df['PDB_Chain'] = [f'{x[1]}:{x[2]}' for x in df[['PDB', 'Chain']].itertuples()]
        LOGGER.debug('Created PDB_Chain column')

        for col in ['PDB_start', 'PDB_end']:
            df[col] = [
                "".join(filter(lambda x: x.isdigit(), value)) for value in df[col]
            ]
            df[col] = df[col].astype(int)
        LOGGER.debug('Digitized boundaries')

        sel = ['PDB_Chain'] + list(self.renames.values())
        df = df[sel].sort_values(['UniProt_ID', 'PDB_Chain'])
        LOGGER.debug(f'Finished parsing df: records={len(df)}')

        if overwrite or self.df is None:
            self.df = df

        mapping = self._prepare_id_map(df)

        if store_to_resources:
            self._store()

        if rm_raw:
            os.remove(self.path)

        return df, mapping

    def dump(self, path: Path, **kwargs):
        """
        :param path: a valid writable path.
        :param kwargs: passed to ``DataFrame.to_csv()`` method.
        :return:
        """
        if self.df is not None:
            self.df.to_csv(path, **kwargs)
        raise RuntimeError('Nothing to dump')

    def fetch(self, url: str = SIFTS_FTP, overwrite: bool = False):
        raw_path = RESOURCES / RAW_SEGMENTS
        if raw_path.exists() and not overwrite:
            LOGGER.info('Raw SIFTS download exists and will not be overwritten')
            return raw_path

        LOGGER.info(f'Fetching SIFTS to {raw_path}')

        fetch_to_file(SIFTS_FTP, raw_path)

        LOGGER.info(f'Downloaded sifts to {raw_path}')

        return raw_path

    @staticmethod
    def _categorize(obj_id: str) -> str:
        if ':' in obj_id:
            LOGGER.debug(f'Assumed {obj_id} to be a PDB:Chain')
            sel_column = 'PDB_Chain'
        elif len(obj_id) == 4:
            LOGGER.debug(f'Assumed {obj_id} to be a PDB ID')
            sel_column = 'PDB'
        else:
            LOGGER.debug(f'Assumed {obj_id} to be a UniProt ID')
            sel_column = 'UniProt_ID'
        return sel_column

    def map_numbering(self, obj_id: str) -> abc.Generator[Mapping]:
        """
        Retrieve mappings associated with the ``obj_id``. Mapping example::

            1 -> 2
            2 -> 3
            3 -> None
            4 -> 4

        Above, a UniProt sequence maps to two segments of a PDB sequence
        (2-3 and 4). PDB sequence is always considered a subset of
        a corresponding UniProt sequence. Thus, any "holes" between continuous
        PDB segments are filled with ``None``.

        .. figure:: fig/segments.png
           :scale: 50 %
           :alt: segments

           Mapping from PDB segments to UniProt segments accounting
           for discontinuities.

        .. seealso::
            :func:`map_segment_numbering <lXtractor.core.segment.map_
            segment_numbering>`

            :func:`wrap_into_segments`.

        :param obj_id: a string value in three possible formats:

            1. "PDB ID:Chain ID"
            2. "PDB ID"
            3. "UniProt ID"

        :return: an iterator over the ``Mapping`` objects.
            These are "unidirectional", i.e., the ``Mapping`` is always
            from the UniProt numbering to the PDB numbering regardless
            of the ``obj_id`` nature.
        """
        sel_column = self._categorize(obj_id)

        if self.df is None:
            raise MissingData('No SIFTS df found; try calling .parse() first')

        sub = self.df[self.df[sel_column] == obj_id]
        LOGGER.debug(f'Subset SIFTS by {sel_column}={obj_id}, records={len(sub)}')

        if len(sub) == 0:
            LOGGER.warning(f"Failed to find {obj_id} in SIFTS")
            yield None

        group_ids, dfs = unzip(sub.groupby(['UniProt_ID', 'PDB_Chain']))
        segments = map(wrap_into_segments, dfs)
        mappings = starmap(map_segment_numbering, segments)
        yield from (
            Mapping(uni_id, pdb_chain, m)
            for (uni_id, pdb_chain), m in zip(group_ids, mappings)
        )

    def map_id(self, x: str) -> list[str] | None:
        """
        :param x: Identifier to map from.
        :return: A list of IDs that `x` maps to.
        """
        if self.id_mapping is None:
            self._prepare_id_map(self.df)
        res = self.id_mapping.get(x)
        if res is None:
            LOGGER.warning(f"Couldn't find {x} in SIFTS")
        return res

    @property
    def uniprot_ids(self) -> set[str]:
        """
        :return: A set of encompassed UniProt IDs.
        """
        return {x for x in self.id_mapping if self._categorize(x) == 'UniProt_ID'}

    @property
    def pdb_ids(self) -> set[str]:
        """
        :return: A set of encompassed PDB IDs.
        """
        return {x for x in self.id_mapping if self._categorize(x) == 'PDB'}

    @property
    def pdb_chains(self) -> set[str]:
        """
        :return: A set of encompassed PDB Chains (in {PDB_ID}:{PDB_Chain} format).
        """
        return {x for x in self.id_mapping if self._categorize(x) == 'PDB_Chain'}


def wrap_into_segments(df: pd.DataFrame) -> tuple[list[Segment], list[Segment]]:
    """
    :param df: A subset of a :attr:`Sifts.df` corresponding to a unique
        "UniProt_ID -- PDB_ID:Chain_ID" pair.
    :return: Two lists with the same length (1) UniProt segments, and (2) PDB segments,
        where segments correspond to each other.
    """
    ids = df['UniProt_ID'].unique()
    if len(ids) > 1:
        raise ValueError(f'Expected single UniProt ID for group {df}, got {ids}')
    ids = df['PDB_Chain'].unique()
    if len(ids) > 1:
        raise ValueError(f'Expected single PDB chain for group {df}, got {ids}')

    uniprot_segments, pdb_segments = map(
        list,
        unzip(
            (
                Segment(row.UniProt_start, row.UniProt_end),
                Segment(row.PDB_start, row.PDB_end),
            )
            for _, row in df.iterrows()
        ),
    )

    return uniprot_segments, pdb_segments


if __name__ == '__main__':
    raise RuntimeError
