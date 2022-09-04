import logging
import subprocess as sp
import sys
import typing as t
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from shutil import copyfileobj
from time import sleep

import ray
import requests
from more_itertools import divide
from tqdm.auto import tqdm

from lXtractor.core.base import _Fetcher, _Getter

T = t.TypeVar('T')
LOGGER = logging.getLogger(__name__)


# =================================== Fetching ========================================

def download_text(
        url: str, decode: bool = True, **kwargs
) -> t.Union[str, bytes]:
    r = requests.get(url, stream=True, **kwargs)
    chunk_size = 1024 * 8
    if r.ok:
        decoded = (
            chunk.decode('utf-8') if decode else chunk
            for chunk in r.iter_content(chunk_size=chunk_size))
        return "".join(decoded) if decode else b"".join(decoded)
    else:
        raise RuntimeError(
            f'Downloading url {url} failed with status code '
            f'{r.status_code} and output {r.text}')


def download_to_file(
        url: str, fpath: t.Optional[Path] = None,
        fname: t.Optional[str] = None,
        root_dir: t.Optional[Path] = None) -> Path:
    if fpath is None:
        root_dir = root_dir or Path().cwd()
        fname = fname or url.split('/')[-1]
        fpath = root_dir / fname
    if not fpath.parent.exists():
        raise ValueError(
            f'Directory {fpath.parent} must exist')
    if url.startswith('ftp'):
        with urllib.request.urlopen(url) as r, fpath.open('wb') as f:
            copyfileobj(r, f)
    else:
        with requests.get(url, stream=True) as r:
            with fpath.open('wb') as f:
                copyfileobj(r.raw, f)
    return fpath


def fetch_iterable(
        it: t.Iterable[str],
        fetcher: t.Callable[[t.Iterable[str]], T],
        chunk_size: int = 100,
        num_threads: t.Optional[int] = None,
        verbose: bool = False,
        sleep_sec: int = 5,
) -> t.List[T]:
    unpacked = list(it)
    num_chunks = max(1, len(unpacked) // chunk_size)
    if num_chunks == 1:
        try:
            return [fetcher(unpacked)]
        except Exception as e:
            LOGGER.warning(f'Failed to fetch a single chunk due to error {e}')
            return []

    chunks = divide(num_chunks, unpacked)
    results = []
    with ThreadPoolExecutor(num_threads) as executor:
        futures = as_completed([executor.submit(fetcher, c) for c in chunks])
        if verbose:
            futures = tqdm(futures, desc='Fetching chunks', total=num_chunks)
        for i, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception as e:
                LOGGER.warning(f'Failed to fetch chunk {i} due to error {e}')
                if 'closed' in str(e):
                    LOGGER.warning(f'Closed connection: sleep for {sleep_sec} seconds')
                    sleep(sleep_sec)
    return results


def try_fetching_until(
        it: t.Iterable[str],
        fetcher: _Fetcher,
        get_remaining: _Getter,
        max_trials: int,
        desc: t.Optional[str] = None,
) -> t.Tuple[t.List[t.List[T]], t.Sequence[str]]:
    current_chunks = []
    remaining = list(it)
    trial = 1
    bar = tqdm(desc=desc, total=max_trials, position=0, leave=True) if desc else None
    while remaining and trial <= max_trials:
        if desc is not None:
            chunk_sizes = [len(c) for c in current_chunks]
            LOGGER.debug(f'{desc},trial={trial},chunk_sizes={chunk_sizes}')
        fetched = fetcher(remaining)
        if len(fetched):
            current_chunks.append(fetched)
            remaining = get_remaining(fetched, remaining)
        else:
            LOGGER.warning(f'failed to fetch anything on trial {trial}')
        trial += 1
        if bar is not None:
            bar.update()
    if bar is not None:
        bar.close()
    return current_chunks, remaining


# =================================== Logging ========================================

def setup_logger(
        log_path: t.Optional[t.Union[str, Path]] = None,
        file_level: t.Optional[int] = None,
        stdout_level: t.Optional[int] = None,
        stderr_level: t.Optional[int] = None,
        logger: t.Optional[logging.Logger] = None
) -> logging.Logger:
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger('parso.python.diff').disabled = True

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(module)s--%(funcName)s]: %(message)s')
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

def run_handles(handles, bar=None):
    results = []
    while handles:
        done, handles = ray.wait(handles)
        done = ray.get(done[0])
        if done is not None:
            results.append(done)
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return results


def run_sp(cmd: str, split: bool = True):
    cmd = cmd.split() if split else cmd
    try:
        res = sp.run(cmd, capture_output=True, text=True, check=True, shell=not split)
    except sp.CalledProcessError as e:
        res = sp.run(cmd, capture_output=True, text=True, check=False, shell=not split)
        raise ValueError(f'Command {cmd} failed with an error {e}, '
                         f'stdout {res.stdout}, stderr {res.stderr}')
    return res


if __name__ == '__main__':
    raise RuntimeError
