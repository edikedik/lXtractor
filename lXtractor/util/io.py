"""
Various utilities for IO.
"""
from __future__ import annotations

import logging
import subprocess as sp
import sys
import typing as t
import urllib
from collections import abc
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from itertools import filterfalse, starmap, chain
from os import PathLike
from pathlib import Path

import pandas as pd
import requests
from more_itertools import chunked_even, peekable, unzip
from tqdm.auto import tqdm

from lXtractor.core.base import UrlGetter

T = t.TypeVar('T')
V = t.TypeVar('V')
_U = t.TypeVar('_U', str, tuple[str, ...])
_F = t.TypeVar('_F', str, Path)
LOGGER = logging.getLogger(__name__)


# =================================== Fetching =========================================


def fetch_text(url: str, decode: bool = True, **kwargs) -> t.Union[str, bytes]:
    """
    :param url: Link.
    :param decode: Decode text to utf-8 (while receiving bytes).
    :param kwargs: Passed to :func:requests.get`.
    :return: Fetched text as a single string.
    """
    r = requests.get(url, stream=True, **kwargs)
    chunk_size = 1024 * 8
    if r.ok:
        decoded = (
            chunk.decode('utf-8') if decode else chunk
            for chunk in r.iter_content(chunk_size=chunk_size)
        )
        return "".join(decoded) if decode else b"".join(decoded)
    raise RuntimeError(
        f'Downloading url {url} failed with status code '
        f'{r.status_code} and output {r.text}'
    )


def fetch_to_file(
    url: str,
    fpath: t.Optional[Path] = None,
    fname: t.Optional[str] = None,
    root_dir: t.Optional[Path] = None,
    text: bool = True,
    **kwargs,
) -> Path:
    """
    :param url: Link to a file.
    :param fpath: Path to a file for saving. If provided, `fname` and
        `root_dir` are ignored. Otherwise, will use ``.../{this}`` from the
        link for the file name and save into the current dir.
    :param fname: Name of the file to save.
    :param root_dir: Dir where to save the file.
    :param text: File is expected to contain text. If ``True``, will use
        :func:`download_text` to fetch text and save it to the file.
    :param kwargs: Passed to :func:`urllib.request.urlretrieve`
        if ``text=False`` else to :func:`download_text`.
    :return: Local path to the file.
    """
    if fpath is None:
        root_dir = root_dir or Path().cwd()
        fname = fname or url.split('/')[-1]
        fpath = root_dir / fname
    if not fpath.parent.exists():
        raise ValueError(f'Directory {fpath.parent} must exist')
    if not text or url.startswith('ftp'):
        urllib.request.urlretrieve(url, fpath, **kwargs)
    else:
        text = fetch_text(url, decode=True, **kwargs)
        with fpath.open('w') as f:
            print(text, file=f)
    return fpath


def fetch_chunks(
    it: abc.Iterable[V],
    fetcher: abc.Callable[[list[V]], T],
    chunk_size: int = 100,
    **kwargs,
) -> abc.Generator[tuple[list[V], T | Future]]:
    """
    A wrapper for fetching multiple links with :class:`ThreadPoolExecutor`.

    :param it: Iterable over some objects accepted by the `fetcher`,
        e.g., links.
    :param fetcher: A callable accepting a chunk of objects from `it`,
        fetching and returning the result.
    :param chunk_size: Split iterable into this many chunks for the executor.
    :param kwargs: Passed to :func:`fetch_iterable`.
    :return: A list of results
    """

    chunks = chunked_even(it, chunk_size)

    yield from fetch_iterable(chunks, fetcher, **kwargs)


def fetch_iterable(
    it: abc.Iterable[V],
    fetcher: abc.Callable[[V], T],
    num_threads: t.Optional[int] = None,
    verbose: bool = False,
    blocking: bool = True,
    allow_failure: bool = True,
) -> abc.Generator[tuple[V, T | Future]]:
    """
    :param it: Iterable over some objects accepted by the `fetcher`,
        e.g., links.
    :param fetcher: A callable accepting a chunk of objects from `it`, fetching
        and returning the result.
    :param num_threads: The number of threads for :class:`ThreadPoolExecutor`.
    :param verbose: Enable progress bar and warnings/exceptions on fetching
        failures.
    :param blocking: If ``True``, will wait for each result.
        Otherwise, will return :class:`Future` objects instead of fetched data.
    :param allow_failure: If ``True``, failure to fetch will raise a warning
        isntead of an exception. Otherwise, the warning is logged, and the
        results won't contain inputs that failed to fetch.
    :return: A list of tuples where the first object is the input and the
        second object is the fetched data.
    """

    def _try_get_result(inp, future=None):
        try:
            return inp, fetcher(inp) if future is None else future.result()
        except Exception as e:
            if not allow_failure:
                raise e
            if verbose:
                LOGGER.warning(f'Failed to fetch input {inp} due to error {e}')
                LOGGER.exception(e)

    if num_threads is None:
        results = filterfalse(lambda x: x is None, map(_try_get_result, it))
        if verbose:
            results = tqdm(results, desc='Fetching')
        yield from results
    else:
        with ThreadPoolExecutor(num_threads) as executor:
            futures_map = {executor.submit(fetcher, x): x for x in it}
            futures = as_completed(list(futures_map))

            if verbose:
                futures = tqdm(futures, desc='Fetching', total=len(futures_map))

            results = ((futures_map[f], f) for f in futures)

            if not blocking:
                yield from results
            else:
                yield from filterfalse(
                    lambda x: x is None, starmap(_try_get_result, results)
                )


def fetch_max_trials(
    it: abc.Iterable[V],
    fetcher: abc.Callable[[abc.Iterable[V]], list[T]],
    get_remaining: abc.Callable[[list[T], list[V]], list[V]],
    max_trials: int,
    verbose: bool = False,
) -> t.Tuple[list[list[T]], list[V]]:
    """
    A wrapper that attempts to fetch many links several times.

    Firstly, it will pass all `it` objects to `fetcher`.
    It will use `get_remaining` function to process fetched results and obtain
    a list of remaining objects to fetch. It will repeat this `max_trials`
    times.

    :param it: Iterable over objects to fetch, e.g., links to files/webpages.
    :param fetcher: A callable accepting an equivalent of `it` and returning
        a list of fetched chunks. It's best to use the curried version of
        a :func:`fetch_iterable` here.
    :param get_remaining: A callable accepting trial results and returning
        a list of remaining entries that will be supplied to the `fetcher`
        on the next trial.
    :param max_trials: Maximum attempts to call the fetcher.
    :param verbose: Display the progress bar of the ongoing trials.
    :return: A tuple of (1) list of fetched chunks during each trial and (2)
        a list of remaining entries failed to fetch at `max_trials`.
    """
    trials = []
    remaining = list(it)
    trial = 1
    pbar = (
        tqdm(desc='Fetching trials', total=max_trials, position=0, leave=True)
        if verbose
        else None
    )
    while remaining and trial <= max_trials:
        fetched = fetcher(remaining)

        if len(fetched):
            trials.append(fetched)
            remaining = get_remaining(fetched, remaining)
        else:
            LOGGER.warning(f'failed to fetch anything on trial {trial}')
        trial += 1
        if pbar is not None:
            pbar.update()
    if pbar is not None:
        pbar.close()
    return trials, remaining


def fetch_files(
    url_getter: UrlGetter,
    url_getter_args: abc.Iterable[_U],
    fmt: str,
    dir_: Path | None,
    *,
    fname_idx: int = 0,
    callback: abc.Callable[[str], T] | None = None,
    overwrite: bool = False,
    max_trials: int = 1,
    num_threads: int | None = None,
    verbose: bool = False,
) -> tuple[list[tuple[_U, _F | T]], list[_U]]:
    """
    :param url_getter: A callable accepting two or more strings and returning
        a valid url to fetch. The last argument is reserved for `fmt`.
    :param url_getter_args: An iterable over strings or tuple of strings
        supplied to the `url_getter`. Each element must be sufficient for the
        `url_getter` to return a valid URL.
    :param dir_: Dir to save files to. If ``None``, will return either raw
        string or json-derived dictionary if the `fmt` is "json".
    :param fmt: File format. It is used construct a full file name
        "{filename}.{fmt}".
    :param fname_idx: If an element in `url_getter_args` is a tuple, this
        argument is used to index this tuple to construct a file name that is
        used to save file / check if such file exists.
    :param callback: A callable to parse the text right after fetching, e.g.,
        ``json.loads``. It's only used if the `dir_` is ``None``.
    :param overwrite: Overwrite existing files if `dir_` is provided.
    :param max_trials: Max number of fetching attempts for a given id.
    :param num_threads: The number of threads to use for parallel requests.
        If ``None``, will send requests sequentially.
    :param verbose: Display progress bar.
    :return: A tuple with fetched results and the remaining file names.
        The former is a list of tuples, where the first element is the
        original name, and the second element is either the path to
        a downloaded file or downloaded data as string. The order may differ.
        The latter is a list of names that failed to fetch.
    """

    def fetch_one(args: _U) -> _F | T:
        url = url_getter(args) if isinstance(args, str) else url_getter(*args)
        if dir_ is None:
            content = fetch_text(url)
            if callback:
                content = callback(content)
            return content
        fname_base = args if isinstance(args, str) else args[fname_idx]
        return fetch_to_file(url, fname=f'{fname_base}.{fmt}', root_dir=dir_)

    def fetcher(chunk: abc.Iterable[_U]) -> list[tuple[_U, _F | T]]:
        chunk = peekable(chunk)
        if not chunk.peek(False):
            return []
        return list(
            fetch_iterable(
                chunk, fetcher=fetch_one, num_threads=num_threads, verbose=verbose
            )
        )

    def get_remaining(
        fetched: abc.Iterable[tuple[_U, _F | T]], _remaining: list[_U]
    ) -> list[_U]:
        args, _ = unzip(fetched)
        return list(set(_remaining) - set(args))

    def filter_existing(args):
        def exists(arg):
            if isinstance(arg, str):
                return f'{arg}.{fmt}' in existing_names
            return f'{arg[fname_idx]}.{fmt}' in existing_names

        existing_names = {x.name for x in dir_.glob(f'*.{fmt}')}
        return list(filterfalse(exists, args))

    if not isinstance(url_getter_args, list):
        url_getter_args = list(url_getter_args)

    if dir_ is not None:
        dir_.mkdir(parents=True, exist_ok=True)

        if not overwrite:
            url_getter_args = filter_existing(url_getter_args)

    if not url_getter_args:
        return [], []

    results, remaining = fetch_max_trials(
        url_getter_args,
        fetcher=fetcher,
        get_remaining=get_remaining,
        max_trials=max_trials,
        verbose=verbose,
    )

    results = list(chain.from_iterable(results))

    return results, remaining


# =================================== Logging ==========================================


def setup_logger(
    log_path: t.Optional[t.Union[str, Path]] = None,
    file_level: t.Optional[int] = None,
    stdout_level: t.Optional[int] = None,
    stderr_level: t.Optional[int] = None,
    logger: t.Optional[logging.Logger] = None,
) -> logging.Logger:
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger('parso.python.diff').disabled = True

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(module)s--%(funcName)s]: %(message)s'
    )
    if logger is None:
        logger = logging.getLogger(__name__)

    if log_path is not None:
        level = file_level or logging.DEBUG
        handler = logging.FileHandler(log_path, 'w')
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
    if stderr_level is not None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        handler.setLevel(stderr_level)
        logger.addHandler(handler)
    if stdout_level is not None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(stdout_level)
        logger.addHandler(handler)

    return logger


# =================================== Processes ========================================

# def run_handles(handles, bar=None):
#     results = []
#     while handles:
#         done, handles = ray.wait(handles)
#         done = ray.get(done[0])
#         if done is not None:
#             results.append(done)
#         if bar is not None:
#             bar.update(1)
#     if bar is not None:
#         bar.close()
#     return results


def run_sp(cmd: str, split: bool = True):
    """
    It will attempt to run the command as a subprocess returning text.
    If the command returns `CalledProcessError`, it will rerun the command
    with ``check=False`` to capture all the outputs into the result.

    :param cmd: A single string of a command.
    :param split: Split `cmd` before running. If ``False``, will pass
        ``shell=True``.
    :return: Result of a subprocess with captured output.
    """
    cmd = cmd.split() if split else cmd
    try:
        res = sp.run(cmd, capture_output=True, text=True, check=True, shell=not split)
    except sp.CalledProcessError as e:
        res = sp.run(cmd, capture_output=True, text=True, check=False, shell=not split)
        raise ValueError(
            f'Command {cmd} failed with an error {e}, '
            f'stdout {res.stdout}, stderr {res.stderr}'
        ) from e
    return res


# ================================= File system ========================================


def is_open_compatible(file):
    """
    :param file: Something attempting to be a file.
    :return: True if `file` is something that can be passed to :func:`open`.
    """
    return isinstance(file, (str, bytes, PathLike))


def get_files(path: Path) -> dict[str, Path]:
    """
    :param path: Path to a directory.
    :return: Mapping {file name => file path} for each file in `path`.
    """
    return {p.name: p for p in path.iterdir() if p.is_file()}


def get_dirs(path: Path) -> dict[str, Path]:
    """
    :param path: Path to a directory.
    :return: Mapping {dir name => dir path} for each dir in `path`.
    """
    return {p.name: p for p in path.iterdir() if p.is_dir()}


# =================================== Parsing ==========================================


def read_n_col_table(path: Path, n: int, sep='\t') -> t.Optional[pd.DataFrame]:
    """
    Read table from file and ensure it has exactly `n` columns.
    """
    df = pd.read_csv(path, sep=sep, header=None)
    if len(df.columns) != n:
        LOGGER.error(
            f'Expected two columns in the table {path}, ' f'but found {len(df.columns)}'
        )
        return None
    return df


if __name__ == '__main__':
    raise RuntimeError
