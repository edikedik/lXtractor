import inspect
import typing as t
from collections import abc
from itertools import chain
from pathlib import Path

from more_itertools import peekable, unzip

from lXtractor.core.base import UrlGetter
from lXtractor.core.chain import CT
from lXtractor.util.io import fetch_max_trials, download_text, download_to_file, fetch_iterable


@t.runtime_checkable
class SupportsAnnotate(t.Protocol[CT]):
    """
    A class that serves as basis for annotators -- callables accepting a `Chain*`-type
    object and returning a single or multiple objects derived from an initial `Chain*`,
    e.g., via :meth:`spawn_child <lXtractor.core.chain.Chain.spawn_child`.
    """

    def annotate(self, c: CT, *args, keep: bool = True, **kwargs) -> CT | abc.Iterable[CT]: ...


class ApiBase:
    """
    Base class for simple APIs for webservices.
    """

    def __init__(
            self, url_getters: dict[str, UrlGetter],
            max_trials: int = 1, num_threads: int | None = None, verbose: bool = False,
    ):
        """
        :param url_getters: A dictionary holding functions constructing urls from provided args.
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
        self.url_getters: dict[str, UrlGetter] = url_getters

    @property
    def url_names(self) -> list[str]:
        """
        :return: A list of supported services.
        """
        return list(self.url_getters)

    @property
    def url_args(self) -> list[tuple[str, list[str]]]:
        """
        :return: A list of services and argument names necessary to construct a valid url.
        """
        return [(k, list(inspect.signature(v).parameters)) for k, v in self.url_getters.items()]


def fetch_files(
        url_getter: UrlGetter,
        filenames: abc.Iterable[str], fmt: str, dir_: Path | None, *,
        overwrite: bool = False, max_trials: int = 1, num_threads: int | None = None,
        verbose: bool = False,
) -> tuple[list[tuple[str, Path | str]], list[str]]:
    """
    :param url_getter: A callable accepting strings and returning a valid url to fetch.
    :param filenames: An iterable over file names without the extension (fmt).
    :param dir_: Dir to save files to. If ``None``, will keep downloaded structures as strings.
    :param fmt: File format. It is used construct a full file name "{id}.{fmt}".
    :param overwrite: Overwrite existing files if `dir_` is provided.
    :param max_trials: Max number of fetching attempts for a given id.
    :param num_threads: The number of threads to use for parallel requests. If ``None``,
        will send requests sequentially.
    :param verbose: Display progress bar.
    :return: A tuple with fetched results and the remaining file names. The former is a list of tuples,
        where the first element is the original name, and the second element is either the path to
        a downloaded file or downloaded data as string. The order may differ.
        The latter is a list of names that failed to fetch.
    """

    def fetch_one(_id):
        url = url_getter(_id, fmt)
        if dir_ is None:
            return download_text(url)
        return download_to_file(url, root_dir=dir_)

    def fetcher(chunk: abc.Iterable[str]) -> list[tuple[str, Path | str]]:
        chunk = peekable(chunk)
        if not chunk.peek(False):
            return []
        return list(fetch_iterable(
            chunk, fetcher=fetch_one, num_threads=num_threads, verbose=verbose,
        ))

    def get_remaining(
            fetched: abc.Iterable[tuple[str, Path | str]], _remaining: list[str]
    ) -> list[str]:
        urls, _ = unzip(fetched)
        fetched_ids = {x.split('/')[-1].split('.')[0] for x in urls}
        return list(set(_remaining) - fetched_ids)

    if not isinstance(filenames, list):
        filenames = list(filenames)

    if dir_ is not None:
        dir_.mkdir(parents=True, exist_ok=True)

        if not overwrite:
            existing_names = {x.name for x in dir_.glob(f'*{fmt}')}
            current_names = {f'{x}.{fmt}' for x in filenames}
            filenames = [x.split('.')[0] for x in current_names - existing_names]

    if not filenames:
        return [], []

    results, remaining = fetch_max_trials(
        filenames, fetcher=fetcher, get_remaining=get_remaining,
        max_trials=max_trials, verbose=verbose)

    results = list(chain.from_iterable(results))

    return results, remaining


if __name__ == '__main__':
    raise RuntimeError
