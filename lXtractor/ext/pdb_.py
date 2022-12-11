import json
import logging
import typing as t
from collections import abc
from itertools import chain
from pathlib import Path

from more_itertools import flatten, peekable
from toolz import curry

from lXtractor.util.io import try_fetching_until, fetch_iterable, download_to_file, download_text

LOGGER = logging.getLogger(__name__)


class PDB:
    """
    Basic RCSB PDB interface to fetch structures and information.
    """

    def __init__(
            self,
            max_trials: int = 3,
            num_threads: t.Optional[int] = None,
            url_files: str = 'https://files.rcsb.org/download',
            url_info: str = 'https://data.rcsb.org/rest/v1/core/entry',
            sleep_sec: int = 5, chunk_size: int = 50, verbose: bool = False,
    ):
        """

        :param max_trials: Max number of fetching attempts for a given query (PDB ID).
        :param num_threads: The number of threads to use for parallel requests.
        :param url_files: Base URL to fetch structure files.
        :param url_info: Base URL to fetch entry info.
        :param sleep_sec: In case of blocking calls, the number of seconds
            to sleep before reattempting to fetch a chunk.
        :param chunk_size: Split inputs into chunks of this size to submit to parallel
            workers.
        :param verbose: Display progress bar.
        """
        self.max_trials = max_trials
        self.num_threads = num_threads
        self.url_files = url_files
        self.url_info = url_info
        self.sleep_sec = sleep_sec
        self.chunk_size = chunk_size
        self.verbose = verbose

    def fetch(
            self, ids: abc.Iterable[str], pdb_dir: Path, fmt: str = 'cif', *, overwrite: bool = False,
    ) -> tuple[list[Path], list[str]]:
        """
        Fetch structure files from RCSB PDB.

        :param ids: An iterable over PDB IDs.
        :param pdb_dir: Dir to save files to.
        :param fmt: Structure format.
        :param overwrite: Overwrite existing files.
        :return: Paths to the fetched files and a list of remaining IDs that failed to fetch
            (empty if all were fetched successfully).
        """

        def fetcher(chunk: t.Iterable[str]) -> list[Path]:
            chunk = peekable(chunk)
            if not chunk.peek(False):
                return []
            urls = (f'{self.url_files}/{x}.{fmt}' for x in chunk)
            _fetcher = curry(download_to_file, root_dir=pdb_dir)
            fetched = fetch_iterable(
                urls, fetcher=lambda xs: [download_to_file(x, root_dir=pdb_dir) for x in xs],
                num_threads=self.num_threads, chunk_size=self.chunk_size, sleep_sec=self.sleep_sec,
                verbose=self.verbose
            )
            return list(chain.from_iterable(fetched))

        def get_remaining(
                fetched: abc.Iterable[Path], _remaining: t.Sequence[str]
        ) -> list[str]:
            _current = {x.stem for x in fetched}
            return list(set(_remaining) - _current)

        if pdb_dir is not None:
            pdb_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(ids, list):
            ids = list(ids)
        if not overwrite:
            existing_names = {x.name for x in pdb_dir.glob(f'*{fmt}')}
            current_names = {f'{x}.{fmt}' for x in ids}
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

    def get_info(self, ids: abc.Iterable[str]):
        """
        Get `PDB entry details <https://data.rcsb.org/redoc/index.html#tag/Entry-Service/operation/getEntryById>`

        :param ids: IDs of entries to fetch.
        :return: List of dictionaries -- parsed json outputs.
        """

        def fetcher(chunk: t.Iterable[str]) -> list[dict]:
            chunk = peekable(chunk)
            if not chunk.peek(False):
                return []
            urls = (f'{self.url_info}/{x}' for x in chunk)
            fetched = fetch_iterable(
                urls, fetcher=lambda xs: list(map(download_text, xs)),
                num_threads=self.num_threads, chunk_size=self.chunk_size,
                sleep_sec=self.sleep_sec, verbose=self.verbose
            )
            return list(map(json.loads, chain.from_iterable(fetched)))

        def get_remaining(
                fetched: abc.Iterable[dict], _remaining: t.Sequence[str]
        ) -> list[str]:
            _current = {x['entry']['id'] for x in fetched}
            return list(set(_remaining) - _current)

        results, remaining = try_fetching_until(
            map(lambda x: x.upper(), ids),
            fetcher=fetcher,
            get_remaining=get_remaining,
            max_trials=self.max_trials,
            verbose=self.verbose)

        if remaining:
            LOGGER.warning(f'Failed to fetch {remaining}')

        results = list(flatten(results))

        return results


if __name__ == '__main__':
    raise RuntimeError
