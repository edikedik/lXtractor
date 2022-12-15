import inspect
import json
import logging
import operator as op
import typing as t
from collections import abc
from itertools import chain
from pathlib import Path

from more_itertools import peekable, unzip
from toolz import curry

from lXtractor.core.base import UrlGetter
from lXtractor.util.io import try_fetching_until, fetch_iterable, download_to_file, download_text

LOGGER = logging.getLogger(__name__)


def _url_getters() -> dict[str, UrlGetter]:

    def _url_getter_factory(name, *args):
        args_fn = ', '.join(args)
        args_url = '/'.join(f'{{{x}}}' for x in args)
        fn = f'lambda {args_fn}: f"{base}/{name}/{args_url}"'
        return eval(fn)

    base = 'https://data.rcsb.org/rest/v1/core'

    staged = [
        # Single argument group
        ('chem_comp', 'comp_id'), ('drugbank', 'comp_id'),
        ('entry', 'entry_id'), ('pubmed', 'entry_id'),
        ('entry_groups', 'group_id'), ('polymer_entity_groups', 'group_id'),
        ('group_provenance', 'group_provenance_id'),
        # Two arguments group
        ('assembly', 'entry_id', 'assembly_id'),
        ('branched_entity', 'entry_id', 'entity_id'),
        ('nonpolymer_entity', 'entry_id', 'entity_id'),
        ('polymer_entity', 'entry_id', 'entity_id'),
        ('branched_entity_instance', 'entry_id', 'asym_id'),
        ('nonpolymer_entity_instance', 'entry_id', 'asym_id'),
        ('polymer_entity_instance', 'entry_id', 'asym_id'),
        ('uniprot', 'entry_id', 'entity_id'),
        # Three argument group
        ('interface', 'entry_id', 'assembly_id', 'interface_id')
    ]

    d = {x[0]: _url_getter_factory(*x) for x in staged}
    d['files'] = (lambda entry_id: f'https://files.rcsb.org/download/{entry_id}')

    return d


class PDB:
    """
    Basic RCSB PDB interface to fetch structures and information.
    """

    def __init__(
            self, max_trials: int = 1, num_threads: t.Optional[int] = None,
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
        #: Upper limit on the number of fetching attempts.
        self.max_trials: int = max_trials
        #: The number of threads passed to the :class:`ThreadPoolExecutor`.
        self.num_threads: int | None = num_threads
        #: Sleep this number of seconds if the request was blocked.
        self.sleep_sec: int = sleep_sec
        #: Passed to :func:`fetch_iterable <lXtractor.util.io.fetch_iterable>`
        self.chunk_size: int = chunk_size
        #: Display progress bar.
        self.verbose: bool = verbose
        #: A dictionary holding functions constructing urls from provided args.
        self.url_getters: dict[str, UrlGetter] = _url_getters()

    @property
    def url_names(self) -> list[str]:
        """
        :return: A list of supported REST API services.
        """
        return list(self.url_getters)

    @property
    def url_args(self) -> list[tuple[str, list[str]]]:
        """
        :return: A list of services and argument names necessary to construct a valid url.
        """
        return [(k, list(inspect.signature(v).parameters)) for k, v in self.url_getters.items()]

    def fetch_files(
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

        def fetcher(chunk: abc.Iterable[str]) -> list[Path]:
            chunk = peekable(chunk)
            if not chunk.peek(False):
                return []
            urls = (f'{url_getter(x)}.{fmt}' for x in chunk)
            _fetcher = curry(download_to_file, root_dir=pdb_dir)
            fetched = fetch_iterable(
                urls, fetcher=lambda xs: [download_to_file(x, root_dir=pdb_dir) for x in xs],
                num_threads=self.num_threads, chunk_size=self.chunk_size, sleep_sec=self.sleep_sec,
                verbose=self.verbose
            )
            return list(chain.from_iterable(fetched))

        def get_remaining(
                fetched: abc.Iterable[Path], _remaining: list[str]
        ) -> list[str]:
            _current = {x.stem for x in fetched}
            return list(set(_remaining) - _current)

        url_getter = self.url_getters['files']

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

        results = list(chain.from_iterable(results))

        return results, remaining

    def get_info(
            self, url_getter: abc.Callable[[str, ...], str],
            url_args: abc.Iterable[tuple[str, ...]]
    ) -> tuple[list[tuple[tuple[str, ...], dict]], list[tuple[str, ...]]]:
        """

        >>> pdb = PDB()
        >>> fetched, remaining = pdb.get_info(pdb.url_getters['entry'], [('2SRC', ), ('2OIQ', )])
        >>> len(remaining) == 0
        True
        >>> len(fetched) == 2
        True
        >>> (args1, res1), (args2, res2) = fetched
        >>> assert {args1, args2} == {('2SRC', ), ('2OIQ', )}
        >>> assert isinstance(res1, dict) and isinstance(res2, dict)

        :param url_getter: A callable accepting strings and returning a valid url to fetch.
        :param url_args: Arguments to a `url_getter`. Check :meth:`url_args` to see which getters
            require which arguments.
        :return: A tuple with fetched and remaining inputs. Fetched inputs are tuples, where the
            first element is the original arguments and the second argument is the dictionary
            with downloaded data. Remaining inputs are arguments that failed to fetch.
        """
        def fetcher(chunk: abc.Iterable[tuple[str, ...]]) -> list[tuple[tuple[str, ...], dict]]:
            chunk = peekable(chunk)
            if not chunk.peek(False):
                return []
            inputs = ((xs, url_getter(*xs)) for xs in chunk)

            fetched = fetch_iterable(
                inputs, fetcher=lambda xs: [(x[0], download_text(x[1])) for x in xs],
                num_threads=self.num_threads, chunk_size=self.chunk_size,
                sleep_sec=self.sleep_sec, verbose=self.verbose
            )

            return list(map(lambda x: (x[0], json.loads((x[1]))), chain.from_iterable(fetched)))

        def get_remaining(
                fetched: abc.Iterable[tuple[tuple[str, ...], dict]],
                _remaining: list[tuple[str, ...]]
        ) -> list[tuple[str, ...]]:
            args, _ = unzip(fetched)
            _current = set(map(op.itemgetter(0), fetched))
            return list(set(_remaining) - set(args))

        results, remaining = try_fetching_until(
            url_args,
            fetcher=fetcher,
            get_remaining=get_remaining,
            max_trials=self.max_trials,
            verbose=self.verbose
        )

        if remaining:
            LOGGER.warning(f'Failed to fetch {remaining}')

        results = list(chain.from_iterable(results))

        return results, remaining


if __name__ == '__main__':
    raise RuntimeError
