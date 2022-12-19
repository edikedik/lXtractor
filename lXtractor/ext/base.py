"""
Base utilities for the ext module, e.g., base classes and common functions.
"""
import inspect
import typing as t
from collections import abc

from lXtractor.core.base import UrlGetter
from lXtractor.core.chain import CT


@t.runtime_checkable
class SupportsAnnotate(t.Protocol[CT]):
    """
    A class that serves as basis for annotators -- callables accepting
    a `Chain*`-type object and returning a single or multiple objects derived
    from an initial `Chain*`, e.g., via
    :meth:`spawn_child <lXtractor.core.chain.Chain.spawn_child`.
    """

    def annotate(
        self, c: CT, *args, keep: bool = True, **kwargs
    ) -> CT | abc.Iterable[CT]:
        """
        A method must accept a Chain*-type object and return a single or
        multiple Chain*-type objects that are the original chain bounds.
        """


class ApiBase:
    """
    Base class for simple APIs for webservices.
    """

    def __init__(
        self,
        url_getters: dict[str, UrlGetter],
        max_trials: int = 1,
        num_threads: int | None = None,
        verbose: bool = False,
    ):
        """
        :param url_getters: A dictionary holding functions constructing urls
            from provided args.
        :param max_trials: Max number of fetching attempts for a given
            query (PDB ID).
        :param num_threads: The number of threads to use for parallel requests.
            If ``None``,will send requests sequentially.
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
        :return: A list of services and argument names necessary
            to construct a valid url.
        """
        return [
            (k, list(inspect.signature(v).parameters))
            for k, v in self.url_getters.items()
        ]


if __name__ == '__main__':
    raise RuntimeError
