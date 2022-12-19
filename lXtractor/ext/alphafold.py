"""
Interface to the AlphaFold database.
"""
import json
import typing as t
from collections import abc
from itertools import repeat
from pathlib import Path

from lXtractor.core.base import UrlGetter
from lXtractor.ext.base import ApiBase
from lXtractor.util.io import fetch_files

T = t.TypeVar('T')


def url_getters() -> dict[str, UrlGetter]:
    """
    :return: A dictionary with {name: getter} where getter is a function
        accepting string args and returning a valid URL.
    """
    base = 'https://alphafold.ebi.ac.uk/files'
    version = 'v3'

    return {
        'model': (lambda _id, fmt: f'{base}/AF-{_id}-F1-model_{version}.{fmt}'),
        'predicted_aligned_error': (
            lambda _id: f'{base}/AF-{_id}-F1-predicted_aligned_error_{version}.json'
        ),
    }


class AlphaFold(ApiBase):
    """
    A basic interface to AlphaFold2 database.
    """

    def __init__(
        self, max_trials: int = 1, num_threads: int | None = None, verbose: bool = False
    ):
        super().__init__(url_getters(), max_trials, num_threads, verbose)

    def fetch_structures(
        self,
        ids: abc.Iterable[str],
        fmt: str = 'cif',
        dir_: Path | None = None,
        *,
        callback: abc.Callable[[str], T] | None = None,
        overwrite: bool = False,
    ) -> tuple[list[tuple[tuple[str, str], str | Path | T]], list[tuple[str, str]]]:
        """
        Fetch structures from the AlphaFold2 database.

        .. seealso::
            :func:`fetch_files <lXtractor.util.io.fetch_files>`

        >>> AF = AlphaFold()
        >>> fetched, failed = AF.fetch_structures(['P12931'], dir_=None)
        >>> len(fetched) == 1 and len(failed) == 0
        True
        >>> (args, text) = fetched.pop()
        >>> args
        ('P12931', 'cif')
        >>> isinstance(text, str)
        True

        :param ids: (UniProt) IDs to fetch.
        :param fmt: Format of the structure.
        :param dir_: Dir to save files to. If provided, the returned value
            is a Path to fetched file. Otherwise, won't save anywhere, and the returned
            fetched value is either a string or anything returned by `callback`.
        :param callback: A wrapper to imideately parse the fetched text.
        :param overwrite: Overwrite existing files if `dir_` is provided.
        :return: A tuple with two lists: (1) fetched data and (2) arguments that
            failed to fetch.
        """

        return fetch_files(
            self.url_getters['model'],
            zip(ids, repeat(fmt)),
            fmt,
            dir_,
            callback=callback,
            overwrite=overwrite,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )

    def fetch_pae(
        self,
        ids: abc.Iterable[str],
        dir_: Path | None = None,
        *,
        overwrite: bool = False,
    ) -> tuple[list[tuple[str, dict]], list[str]]:
        """

        .. seealso::
            :func:`fetch_files <lXtractor.util.io.fetch_files>`

        :param ids: (UniProt) IDs to fetch.
        :param dir_: Dir to save files to. If provided, the returned value
            is a Path to fetched file.
            Otherwise, won't save anywhere, and the returned fetched value
            is either a string or anything returned by `callback`.
        :param overwrite: Overwrite existing files if `dir_` is provided.
        :return: A tuple with two lists:
            (1) fetched data and
            (2) arguments that failed to fetch.
        """

        return fetch_files(
            self.url_getters['predicted_aligned_error'],
            ids,
            'json',
            dir_,
            overwrite=overwrite,
            callback=json.loads,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )


if __name__ == '__main__':
    raise RuntimeError
