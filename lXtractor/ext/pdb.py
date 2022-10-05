"""
Module comprises utils for fetching, splitting, extracting sub-structures
and sub-sequences of/from PDB files.
"""
import logging
import typing as t
from collections import abc
from itertools import chain
from pathlib import Path

from more_itertools import flatten, peekable
from toolz import curry

from lXtractor.util.io import try_fetching_until, fetch_iterable, download_to_file

# WrappedResult = t.Tuple[str, Structure, t.Optional[t.List[t.Tuple[str, t.Any]]]]
META_FIELDS = (
    'idcode',
    'name',
    'resolution',
    'structure_method',
    'deposition_date',
)
LOGGER = logging.getLogger(__name__)


class PDB:
    """
    An interface to fetch PDB structures (in PDB format) and process the results.
    """

    def __init__(
            self, pdb_dir: Path, max_trials: int = 3, num_threads: t.Optional[int] = None,
            meta_fields: t.Optional[t.Tuple[str, ...]] = META_FIELDS,
            expected_method: t.Optional[str] = 'x-ray diffraction',
            min_resolution: t.Optional[int] = None,
            fmt: str = 'cif', url: str = 'https://files.rcsb.org/download',
            sleep_sec: int = 5, chunk_size: int = 50, verbose: bool = False,
    ):
        """
        :param max_trials: a maximum number of fetching attempts.
        :param num_threads: a number of threads for the :class:`ThreadPoolExecutor`.
        :param meta_fields: a tuple of metadata names (potentially) returned by
            :func:`Bio.PDB.parse_pdb_header`. :meth:`PDB.fetch` will include
            these fields into :attr:`lXtractor.protein.Protein.metadata`.
        :param expected_method: filter to structures with "structure_method"
            annotated by a given value.
        :param min_resolution: filter to structures having "resolution" lower or
            equal than a given value.
        :param verbose: progress bar on/off.
        """
        self.max_trials = max_trials
        self.num_threads = num_threads
        self.meta_fields = tuple(meta_fields)
        self.expected_method = expected_method
        self.min_resolution = min_resolution
        self.pdb_dir = pdb_dir
        self.fmt = fmt
        self.url = url
        self.sleep_sec = sleep_sec
        self.chunk_size = chunk_size
        self.verbose = verbose

        if pdb_dir is not None:
            pdb_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
            self, ids: abc.Iterable[str], *, overwrite: bool = False,
    ) -> tuple[list[Path], list[str]]:

        def fetcher(chunk: t.Iterable[str]) -> list[Path]:
            chunk = peekable(chunk)
            if not chunk.peek(False):
                return []
            urls = (f'{self.url}/{x}.{self.fmt}' for x in chunk)
            _fetcher = curry(download_to_file, root_dir=self.pdb_dir)
            fetched = fetch_iterable(
                urls, fetcher=lambda xs: [download_to_file(x, root_dir=self.pdb_dir) for x in xs],
                num_threads=self.num_threads, chunk_size=self.chunk_size, sleep_sec=self.sleep_sec,
                verbose=self.verbose
            )
            return list(chain.from_iterable(fetched))

        def get_remaining(
                fetched: abc.Iterable[Path], _remaining: t.Sequence[str]
        ) -> list[str]:
            _current = {x.stem for x in fetched}
            return list(set(_remaining) - _current)

        if not isinstance(ids, list):
            ids = list(ids)
        if not overwrite:
            existing_names = {x.name for x in self.pdb_dir.glob(f'*{self.fmt}')}
            current_names = {f'{x}.{self.fmt}' for x in ids}
            ids = [x.split('.')[0] for x in current_names - existing_names]

        if not ids:
            return [], []

        results, remaining = try_fetching_until(
            ids,
            fetcher=fetcher,
            get_remaining=get_remaining,
            max_trials=self.max_trials,
            verbose=self.verbose)

        if remaining:
            LOGGER.warning(f'Failed to fetch {remaining}')
        results = list(flatten(results))

        return results, remaining


if __name__ == '__main__':
    raise RuntimeError
