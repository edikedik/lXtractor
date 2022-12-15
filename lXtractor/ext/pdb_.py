import inspect
import json
import logging
from collections import abc
from itertools import chain
from pathlib import Path

from more_itertools import peekable, unzip

from lXtractor.core.base import UrlGetter
from lXtractor.util.io import fetch_max_trials, download_to_file, download_text, fetch_iterable

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
    d['files'] = (lambda entry_id, fmt: f'https://files.rcsb.org/download/{entry_id}.{fmt}')

    return d


class PDB:
    """
    Basic RCSB PDB interface to fetch structures and information.
    """

    def __init__(
            self, max_trials: int = 1, num_threads: int | None = None, verbose: bool = False,
    ):
        """
        :param max_trials: Max number of fetching attempts for a given query (PDB ID).
        :param num_threads: The number of threads to use for parallel requests. If ``None``,
            will send requests sequentially.
        :param verbose: Display progress bar.
        """
        #: Upper limit on the number of fetching attempts.
        self.max_trials: int = max_trials
        #: The number of threads passed to the :class:`ThreadPoolExecutor`.
        self.num_threads: int | None = num_threads
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

    def fetch_structures(
            self, ids: abc.Iterable[str], pdb_dir: Path | None = None, fmt: str = 'cif', *,
            overwrite: bool = False,
    ) -> tuple[list[tuple[str, Path | str]], list[str]]:
        """
        Fetch structure files from RCSB PDB.

        >>> pdb = PDB()
        >>> fetched, failed = pdb.fetch_structures(['2src', '2oiq'])
        >>> len(fetched) == 2 and len(failed) == 0
        True
        >>> (id1, res1), (id2, res2) = fetched
        >>> assert {id1, id2} == {'2src', '2oiq'}
        >>> isinstance(res1, str) and isinstance(res2, str)
        True

        :param ids: An iterable over PDB IDs.
        :param pdb_dir: Dir to save files to. If ``None``, will keep downloaded structures as strings.
        :param fmt: Structure format.
        :param overwrite: Overwrite existing files if `pdb_dir` is provided.
        :return: A tuple with fetched results and the remaining IDs. The former is a list of tuples,
            where the first element is the original ID, and the second element is either the path to
            a downloaded file or downloaded data as string. The order may differ.
            The latter is a list of IDs that failed to fetch.
        """

        def fetch_one(_id):
            url = url_getter(_id, fmt)
            if pdb_dir is None:
                return download_text(url)
            return download_to_file(url, root_dir=pdb_dir)

        def fetcher(chunk: abc.Iterable[str]) -> list[tuple[str, Path | str]]:
            chunk = peekable(chunk)
            if not chunk.peek(False):
                return []
            return list(fetch_iterable(
                chunk, fetcher=fetch_one, num_threads=self.num_threads, verbose=self.verbose,
            ))

        def get_remaining(
                fetched: abc.Iterable[tuple[str, Path | str]], _remaining: list[str]
        ) -> list[str]:
            urls, _ = unzip(fetched)
            fetched_ids = {x.split('/')[-1].split('.')[0] for x in urls}
            return list(set(_remaining) - fetched_ids)

        url_getter = self.url_getters['files']

        if not isinstance(ids, list):
            ids = list(ids)

        if pdb_dir is not None:
            pdb_dir.mkdir(parents=True, exist_ok=True)

            if not overwrite:
                existing_names = {x.name for x in pdb_dir.glob(f'*{fmt}')}
                current_names = {f'{x}.{fmt}' for x in ids}
                ids = [x.split('.')[0] for x in current_names - existing_names]

        if not ids:
            return [], []

        results, remaining = fetch_max_trials(
            ids, fetcher=fetcher, get_remaining=get_remaining,
            max_trials=self.max_trials, verbose=self.verbose)

        if remaining:
            LOGGER.warning(f'Failed to fetch {remaining}')

        results = list(chain.from_iterable(results))

        return results, remaining

    def fetch_info(
            self, url_getter: abc.Callable[[str, ...], str],
            url_args: abc.Iterable[tuple[str, ...]]
    ) -> tuple[list[tuple[tuple[str, ...], dict]], list[tuple[str, ...]]]:
        """

        >>> pdb = PDB()
        >>> fetched, failed = pdb.fetch_info(pdb.url_getters['entry'],[('2SRC', ), ('2OIQ', )])
        >>> len(failed) == 0 and len(fetched) == 2
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

        def fetch_one(args: tuple[str, ...]) -> dict:
            return json.loads(download_text(url_getter(*args)))

        def fetcher(chunk: abc.Iterable[tuple[str, ...]]) -> list[tuple[tuple[str, ...], dict]]:
            chunk = peekable(chunk)
            if not chunk.peek(False):
                return []

            return list(fetch_iterable(
                chunk, fetcher=fetch_one, num_threads=self.num_threads, verbose=self.verbose)
            )

        def get_remaining(
                fetched: abc.Iterable[tuple[tuple[str, ...], dict]],
                _remaining: list[tuple[str, ...]]
        ) -> list[tuple[str, ...]]:
            args, _ = unzip(fetched)
            return list(set(_remaining) - set(args))

        results, remaining = fetch_max_trials(url_args, fetcher=fetcher, get_remaining=get_remaining,
                                              max_trials=self.max_trials, verbose=self.verbose)

        if remaining:
            LOGGER.warning(f'Failed to fetch {remaining}')

        results = list(chain.from_iterable(results))

        return results, remaining


if __name__ == '__main__':
    raise RuntimeError
