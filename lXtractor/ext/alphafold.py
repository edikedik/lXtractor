"""
Interface to the AlphaFold database.
"""
import typing as t
from collections import abc

from lXtractor.ext.base import StructureApiBase

T = t.TypeVar("T")


def url_getters() -> dict[str, abc.Callable[..., str]]:
    """
    :return: A dictionary with {name: getter} where getter is a function
        accepting string args and returning a valid URL.
    """
    base = "https://alphafold.ebi.ac.uk/files"
    version = "v3"

    return {
        "structures": (lambda _id, fmt: f"{base}/AF-{_id}-F1-model_{version}.{fmt}"),
        "pae": (
            lambda _id: f"{base}/AF-{_id}-F1-predicted_aligned_error_{version}.json"
        ),
    }


class AlphaFold(StructureApiBase):
    """
    A basic interface to the AlphaFold2 database.

    To fetch structures:

    >>> af = AlphaFold()
    >>> fetched, missing = af.fetch_structures(['P00523', ], fmt='cif', dir_=None)
    >>> len(fetched) == 1 and len(missing) == 0
    True
    >>> inp, content = fetched.pop()
    >>> inp
    ('P00523', 'cif')
    >>> isinstance(content, str)
    True

    This will fetch raw structure data in the specified format. One may parse
    the structure immediately by passing ``parse=True``.

    To fetch PAE, please use :meth:`fetch_info`:

    >>> fetched, missing = af.fetch_info('pae', ['P00523'], dir_=None)
    >>> len(fetched) == 1 and len(missing) == 0
    True
    >>> inp, content = fetched.pop()
    >>> inp
    'P00523'
    >>> isinstance(content, list)
    True
    >>> assert len(content) == 1
    >>> content = content.pop()
    >>> content['max_predicted_aligned_error']
    31.75

    By default, the fetched PAE is in json format, parsed here as list of dicts.
    """

    def __init__(
        self, max_trials: int = 1, num_threads: int | None = None, verbose: bool = False
    ):
        super().__init__(url_getters(), max_trials, num_threads, verbose)

    @property
    def supported_str_formats(self) -> list[str]:
        return ["cif", "pdb"]


if __name__ == "__main__":
    raise RuntimeError
