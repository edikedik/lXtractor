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
from itertools import filterfalse, starmap, chain, repeat
from os import PathLike, walk
from pathlib import Path
from shutil import copyfileobj

import networkx as nx
import pandas as pd
import requests
from more_itertools import chunked_even, peekable
from tqdm.auto import tqdm

from lXtractor.core.base import UrlGetter
from lXtractor.core.config import DumpNames
from lXtractor.core.exceptions import FormatError

T = t.TypeVar("T")
V = t.TypeVar("V")
_U = t.TypeVar("_U", tuple[str, ...], str)
_F = t.TypeVar("_F", str, Path)
LOGGER = logging.getLogger(__name__)

__all__ = (
    "fetch_text",
    "fetch_to_file",
    "fetch_files",
    "fetch_iterable",
    "fetch_chunks",
    "setup_logger",
    "get_files",
    "get_dirs",
    "run_sp",
    "path_tree",
    "parse_suffix",
    "read_n_col_table"
)


# =================================== Fetching =========================================


def fetch_text(
    url: str, decode: bool = False, chunk_size: int = 1024 * 8, **kwargs
) -> str | bytes:
    """
    Fetch the content as a single string.
    This will use the ``requests.get`` with ``stream=True`` by default to split
    the download into chunks and thus avoid taking too much memory at once.

    :param url: Link to fetch from.
    :param decode: Decode the received bytes to utf-8.
    :param chunk_size: The number of bytes to use when splitting the fetched
        result into chunks.
    :param kwargs: Passed to :func:requests.get`.
    :return: Fetched text as a single string.
    """
    with requests.get(url, stream=True, **kwargs) as r:
        if r.ok:
            res = b"".join(r.iter_content(chunk_size))
            if decode:
                res = res.decode("utf-8")
            return res
    raise RuntimeError(
        f"Downloading url {url} failed with status code "
        f"{r.status_code} and output {r.text}"
    )


def fetch_to_file(
    url: str,
    fpath: Path | None = None,
    fname: str | None = None,
    root_dir: Path | None = None,
    decode: bool = False,
) -> Path:
    """
    :param url: Link to a file.
    :param fpath: Path to a file for saving. If provided, `fname` and
        `root_dir` are ignored. Otherwise, will use ``.../{this}`` from the
        link for the file name and save into the current dir.
    :param fname: Name of the file to save.
    :param root_dir: Dir where to save the file.
    :param decode: If ``True``, try decoding the raw request's content.
    :return: Local path to the file.
    """
    if fpath is None:
        root_dir = root_dir or Path().cwd()
        fname = fname or url.split("/")[-1]
        fpath = root_dir / fname
    if not fpath.parent.exists():
        raise ValueError(f"Directory {fpath.parent} must exist")

    if url.startswith("ftp"):
        urllib.request.urlretrieve(url, fpath)
    else:
        with requests.get(url, stream=True) as r:
            if not r.ok:
                raise RuntimeError(
                    f"Downloading url {url} failed with status code "
                    f"{r.status_code} and output {r.text}"
                )
            if decode:
                r.raw.decode_content = True
            with fpath.open("wb") as f:
                copyfileobj(r.raw, f)

    return fpath


def fetch_chunks(
    it: abc.Iterable[V],
    fetcher: abc.Callable[[list[V]], T],
    chunk_size: int = 100,
    **kwargs,
) -> abc.Generator[tuple[list[V], T | Future], None, None]:
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
) -> (
    abc.Generator[tuple[V, T], None, None]
    | abc.Generator[tuple[V, Future[T]], None, None]
):
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

    def _try_get_result(inp: V, future: Future | None) -> tuple[V, T] | None:
        try:
            return inp, fetcher(inp) if future is None else future.result()
        except Exception as e:
            if not allow_failure:
                raise e
            if verbose:
                LOGGER.warning(f"Failed to fetch input {inp} due to error {e}")
                LOGGER.exception(e)
            return None

    results: (abc.Iterable[tuple[V, T]] | abc.Iterable[tuple[V, Future[T]]])
    if num_threads is None:
        results = (x for x in starmap(_try_get_result, zip(it, repeat(None))) if x)
        if verbose:
            results = tqdm(results, desc="Fetching")
        yield from results
    else:
        with ThreadPoolExecutor(num_threads) as executor:
            futures_map = {executor.submit(fetcher, x): x for x in it}
            futures: abc.Iterable[Future] = as_completed(list(futures_map))

            if verbose:
                futures = tqdm(futures, desc="Fetching", total=len(futures_map))

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
        tqdm(desc="Fetching trials", total=max_trials, position=0, leave=True)
        if verbose
        else None
    )
    while remaining and trial <= max_trials:
        fetched = fetcher(remaining)

        if len(fetched):
            trials.append(fetched)
            remaining = get_remaining(fetched, remaining)
        else:
            LOGGER.warning(f"failed to fetch anything on trial {trial}")
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
    callback: abc.Callable[[_U, str | bytes], T] | None = None,
    overwrite: bool = False,
    decode: bool = False,
    max_trials: int = 1,
    num_threads: int | None = None,
    verbose: bool = False,
) -> tuple[list[tuple[_U, _F] | tuple[_U, T]], list[_U]]:
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
    :param callback: A callable to parse content right after fetching, e.g.,
        ``json.loads``. It's only used if `dir_` is not provided.
    :param overwrite: Overwrite existing files if `dir_` is provided.
    :param decode: Decode the fetched content (bytes to utf-8). Should be
        ``True`` if expecting text content.
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

    # TODO: fix typing issues

    def fetch_one(args: _U) -> str | Path | T:
        url = url_getter(args) if isinstance(args, str) else url_getter(*args)
        if dir_ is None:
            res = fetch_text(url, decode=decode)
            return callback(args, res) if callback else res
        fname_base: str = args if isinstance(args, str) else args[fname_idx]
        return fetch_to_file(
            url, fname=f"{fname_base}.{fmt}", root_dir=dir_, decode=decode
        )

    def fetcher(
        chunk: abc.Iterable[_U],
    ) -> list[tuple[_U, str]] | list[tuple[_U, Path]] | list[tuple[_U, T]]:
        chunk = peekable(chunk)
        if not chunk.peek(False):
            return []
        fetched: abc.Iterable[tuple[_U, str | Path | T]] = fetch_iterable(
            chunk, fetcher=fetch_one, num_threads=num_threads, verbose=verbose
        )
        return list(fetched)

    def get_remaining(
        fetched: abc.Iterable[tuple[_U, _F | T]], _remaining: list[_U]
    ) -> list[_U]:
        args = [x[0] for x in fetched]
        return list(set(_remaining) - set(args))

    def filter_existing(args):
        def exists(arg):
            if isinstance(arg, str):
                return f"{arg}.{fmt}" in existing_names
            return f"{arg[fname_idx]}.{fmt}" in existing_names

        existing_names = {x.name for x in dir_.glob(f"*.{fmt}")}
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
    logging.getLogger("parso.python.diff").disabled = True

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(module)s--%(funcName)s]: %(message)s"
    )
    if logger is None:
        logger = logging.getLogger(__name__)
    handler: logging.FileHandler | logging.StreamHandler
    if log_path is not None:
        level = file_level or logging.DEBUG
        handler = logging.FileHandler(log_path, "w")
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
    command: str | list[str] = cmd.split() if split else cmd
    try:
        res = sp.run(
            command, capture_output=True, text=True, check=True, shell=not split
        )
    except sp.CalledProcessError as e:
        res = sp.run(
            command, capture_output=True, text=True, check=False, shell=not split
        )
        raise ValueError(
            f"Command {command} failed with an error {e}, "
            f"stdout {res.stdout}, stderr {res.stderr}"
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


def _is_valid_seq_path(path: Path) -> bool:
    files = get_files(path)
    return DumpNames.meta in files and DumpNames.sequence in files


def _is_valid_str_path(path: Path) -> bool:
    base_names = [x.split(".")[0] for x in get_files(path)]
    return _is_valid_seq_path(path) and DumpNames.structure_base_name in base_names


# TODO: make doctests
def path_tree(path: Path) -> nx.DiGraph:
    """
    Create a tree graph from Chain*-type objects saved to the filesystem.

    The function will recursively walk starting from the provided path,
    connecting parent and children paths (residing within "segments" directory).
    If it meets a path containing "structures" directory, it will save valid
    structure paths under a node's "structures" attribute. In that case,
    such structures are assumed to be nested under a chain, and they do not
    form nodes in this graph.

    A path to a Chain*-type object is valid if it contains "sequence.tsv"
    and "meta.tsv" files. A valid structure path must contain "sequence.tsv",
    "meta.tsv", and "structure.*" files.

    :param path: A root path to start with.
    :return: An undirected graph with paths as nodes and edges representing
        parent-child relationships.
    """

    d = nx.DiGraph()
    d.add_node(path)

    for root, dirs, files in walk(path):
        root_path = Path(root)
        if (
            DumpNames.meta in files
            and DumpNames.structures_dir != root_path.parent.name
        ):
            d.add_node(root_path)
        else:
            parent, child = root_path.parent.name, root_path.name
            if child == DumpNames.segments_dir:
                for seg in dirs:
                    if _is_valid_seq_path(root_path / seg):
                        d.add_edge(root_path.parent, root_path / seg)
                    else:
                        LOGGER.warning(f"Invalid segment object in {root_path / seg}")
            if child == DumpNames.structures_dir:
                structures = []
                for s_dir in dirs:
                    if _is_valid_str_path(root_path / s_dir):
                        structures.append(root_path / s_dir)
                    else:
                        LOGGER.warning(f"Invalid structure in {root_path / s_dir}")
                if structures:
                    d.nodes[root_path.parent]["structures"] = structures
    return d


def parse_suffix(path: Path) -> str:
    """
    Parse a file suffix.

        #. If there are no suffixes: raise an error.
        #. If there is one suffix, return it.
        #. If there are more than one suffixes, join the last two and return.

    :param path: Input path.
    :return: Parsed suffix.
    :raise FormatError: If not suffix is present.
    """
    suffixes = path.suffixes
    if len(suffixes) == 0:
        raise FormatError(f"No suffixes to infer file type in path {path}")
    elif len(suffixes) == 1:
        suffix = suffixes.pop()
    else:
        suffix = "".join(suffixes[-2:])
    return suffix


# =================================== Parsing ==========================================


def read_n_col_table(path: Path, n: int, sep="\t") -> pd.DataFrame | None:
    """
    Read table from file and ensure it has exactly `n` columns.
    """
    df = pd.read_csv(path, sep=sep, header=None)
    if len(df.columns) != n:
        LOGGER.error(
            f"Expected two columns in the table {path}, " f"but found {len(df.columns)}"
        )
        return None
    return df


if __name__ == "__main__":
    raise RuntimeError
