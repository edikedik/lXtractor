import logging
import subprocess as sp
import sys
import typing as t
import urllib
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from time import sleep

import pandas as pd
import ray
import requests
from Bio import SeqIO
from Bio.PDB import PDBIO
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from more_itertools import divide, take, unique_everseen, split_at
from tqdm.auto import tqdm

from .base import _Fetcher, _Getter, FormatError, SeqRec, Variables

T = t.TypeVar('T')
LOGGER = logging.getLogger(__name__)


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


def parse_cdhit(clstr_file: Path) -> t.List[t.List[str]]:
    """
    Parse cd-hit cluster file into a (list of lists of) clusters with seq ids.
    """
    with clstr_file.open() as f:
        return list(map(
            lambda cc: list(map(lambda c: c.split('>')[1].split('...')[0], cc)),
            filter(bool, split_at(f, lambda x: x.startswith('>')))
        ))


def cluster_cdhit(
        seqs: t.Iterable[SeqRec], ts: float,
        cdhit_exec: t.Union[str, Path] = 'cd-hit'
) -> t.List[t.List[SeqRec]]:
    """
    Run cd-hit with params `-A 0.9 -g 1 -T 0 -d 0`.
    :param seqs: Collection of seq records.
    :param ts: Threshold value (`c` parameter).
    :param cdhit_exec: Path or name of the executable.
    :return: clustered seq record objects.
    """
    def get_word_length():
        """
        -n 5 for thresholds 0.7 ~ 1.0
        -n 4 for thresholds 0.6 ~ 0.7
        -n 3 for thresholds 0.5 ~ 0.6
        -n 2 for thresholds 0.4 ~ 0.5
        """
        if ts > 0.7:
            return 5
        if ts > 0.6:
            return 4
        if ts > 0.5:
            return 3
        return 2

    def ungap_seq(seq: SeqRec):
        return SeqRec(
            seq.seq.ungap(), id=seq.id, name=seq.name,
            description=seq.description)

    seqs_map = {s.id: s for s in seqs}
    seqs = list(map(ungap_seq, seqs_map.values()))
    msa_handle = NamedTemporaryFile('w')
    num_aln = SeqIO.write(seqs, msa_handle, 'fasta')
    LOGGER.debug(f'Wrote {num_aln} sequences into {msa_handle.name}')
    msa_handle.seek(0)
    out_handle = NamedTemporaryFile('w')
    cmd = f'{cdhit_exec} -i {msa_handle.name} -o {out_handle.name} ' \
          f'-c {round(ts, 2)} -g 1 -T 0 -M 0 -d 0 -n {get_word_length()}'
    run_sp(cmd)
    LOGGER.debug(f'successfully executed {cmd}')
    clusters = parse_cdhit(Path(f'{out_handle.name}.clstr'))
    return [[seqs_map[x] for x in c] for c in clusters]


def run_sp(cmd: str, split: bool = True):
    cmd = cmd.split() if split else cmd
    try:
        res = sp.run(cmd, capture_output=True, text=True, check=True, shell=not split)
    except sp.CalledProcessError as e:
        res = sp.run(cmd, capture_output=True, text=True, check=False, shell=not split)
        raise ValueError(f'Command {cmd} failed with an error {e}, '
                         f'stdout {res.stdout}, stderr {res.stderr}')
    return res


def split_validate(inp: str, sep: str, parts: int):
    split = inp.split(sep)
    if len(split) != parts:
        raise FormatError(
            f'Expected {parts} "{sep}" separators, '
            f'got {len(split) - 1} in {inp}')
    return split


class SizedDict(UserDict):

    def __init__(self, max_items: int, *args, **kwargs):
        self.max_items = max_items
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        diff = len(self) - self.max_items
        if diff > 0:
            for k in take(diff, iter(self.keys())):
                super().__delitem__(k)
        super().__setitem__(key, value)


class Dumper:
    def __init__(self, dump_dir: Path):
        self.dump_dir = dump_dir

    def dump_pdb(self, structure: Structure, name: str) -> None:
        _path = self.dump_dir / name
        io = PDBIO()
        io.set_structure(structure)
        with _path.open('w') as handle:
            io.save(handle)
        LOGGER.debug(f'Saved PDB structure {structure.id} to {_path}')

    def dump_seq_rec(
            self, seq_rec: t.Union[SeqRec, t.Iterable[SeqRec]],
            name: str
    ) -> None:
        _path = self.dump_dir / name
        SeqIO.write(seq_rec, _path, 'fasta')
        if isinstance(seq_rec, SeqRec):
            LOGGER.debug(f'Saved sequence {seq_rec.id} to {_path}')
        else:
            LOGGER.debug(f'Saved sequences {[s.id for s in seq_rec]} to {_path}')

    def dump_meta(self, meta: t.Iterable[t.Tuple[str, t.Any]], name: str) -> None:
        _path = self.dump_dir / name
        records = list(unique_everseen(meta))
        with _path.open('w') as f:
            for name, value in records:
                print(name, value, sep='\t', file=f)
        LOGGER.debug(f'Saved {len(records)} metadata records to {_path}')

    def dump_variables(
            self, variables: Variables, name: str,
            skip_if_contains: t.Collection[str] = ('ALL',)) -> None:
        _path = self.dump_dir / name
        with _path.open('w') as f:
            for name, (_, value) in variables.items():
                if skip_if_contains and any(
                        x in name for x in skip_if_contains):
                    continue
                print(name, value, sep='\t', file=f)
        LOGGER.debug(f'Saved {len(variables)} variables to {_path}')

    def dump_distance_map(
            self, distances: t.Iterable[t.Tuple[int, int, float]],
            name: str) -> None:
        _path = self.dump_dir / name
        with _path.open('w') as f:
            for pos1, pos2, dist in distances:
                print(pos1, pos2, dist, sep='\t', file=f)
        LOGGER.debug(f'Saved distance map to {_path}')

    def dump_mapping(self, mapping: t.Dict[t.Any, t.Any], name: str) -> None:
        _path = self.dump_dir / name
        with _path.open('w') as f:
            for k, v in mapping.items():
                print(k, v, sep='\t', file=f)
        LOGGER.debug(f'Saved mapping to {_path}')

    def dump_pdb_raw(self, seq: t.Tuple[str, ...], name: str) -> None:
        _path = self.dump_dir / name
        with _path.open('w') as f:
            print(*seq, sep='\n', file=f)
        LOGGER.debug(f'Saved raw sequence to {_path}')


def setup_logger(
        log_path: t.Optional[t.Union[str, Path]] = None, file_level: t.Optional[int] = None,
        stdout_level: t.Optional[int] = None, stderr_level: t.Optional[int] = None,
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


def subset_by_idx(seq: SeqRec, idx: t.Sequence[int], start=1):
    sub = ''.join(c for i, c in enumerate(seq, start=start) if i in idx)
    start, end = idx[0], idx[-1]
    new_id = f'{seq.id}/{start}-{end}'
    return SeqRec(Seq(sub), new_id, new_id, new_id)


def col2col(df: pd.DataFrame, col_fr: str, col_to: str):
    sub = df[[col_fr, col_to]].drop_duplicates().sort_values([col_fr, col_to])
    groups = groupby(zip(sub[col_fr], sub[col_to]), key=lambda x: x[0])
    return {k: [x[1] for x in group] for k, group in groups}

if __name__ == '__main__':
    raise RuntimeError
