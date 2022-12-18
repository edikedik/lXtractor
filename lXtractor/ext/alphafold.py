from collections import abc
from itertools import repeat
from pathlib import Path

from lXtractor.core.base import UrlGetter
from lXtractor.ext.base import ApiBase, fetch_files


def url_getters() -> dict[str, UrlGetter]:
    def _url_getter_factory(name, v='v3'):
        fn = f'lambda _id, fmt: f"{base}/AF-{{_id}}-F1-{name}_{v}.{{fmt}}"'
        return eval(fn)

    base = 'https://alphafold.ebi.ac.uk/files'
    v = 'v3'

    return {
        'model': (lambda _id, fmt: f'{base}/AF-{_id}-F1-model_{v}.{fmt}'),
        'predicted_aligned_error': (lambda _id: f'{base}/AF-{_id}-F1-predicted_aligned_error_{v}.json')
    }


class AlphaFold(ApiBase):
    def __init__(
            self, max_trials: int = 1, num_threads: int | None = None, verbose: bool = False
    ):
        super().__init__(url_getters(), max_trials, num_threads, verbose)

    def fetch_structures(
            self, ids: abc.Iterable[str], fmt: str = 'cif',
            dir_: Path | None = None, *, overwrite: bool = False
    ):
        return fetch_files(
            self.url_getters['model'], zip(ids, repeat(fmt)), fmt, dir_,
            overwrite=overwrite, max_trials=self.max_trials,
            num_threads=self.num_threads, verbose=self.verbose
        )

    def fetch_pae(
            self, ids: abc.Iterable[str], dir_: Path | None = None, *, overwrite: bool = False
    ):
        return fetch_files(
            self.url_getters['predicted_aligned_error'], ids, 'json', dir_, overwrite=overwrite,
            max_trials=self.max_trials, num_threads=self.num_threads, verbose=self.verbose
        )


if __name__ == '__main__':
    raise RuntimeError
