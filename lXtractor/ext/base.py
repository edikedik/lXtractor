"""
Base utilities for the ext module, e.g., base classes and common functions.
"""
import inspect
import json
import typing as t
from abc import abstractmethod
from collections import abc
from itertools import repeat
from pathlib import Path

from lXtractor.core import GenericStructure
from lXtractor.core.base import UrlGetter
from lXtractor.core.exceptions import FormatError
from lXtractor.util import fetch_files

if t.TYPE_CHECKING:
    from lXtractor.chain import ChainSequence, ChainStructure, Chain

    CT = t.TypeVar("CT", ChainSequence, ChainStructure, Chain)
else:
    CT = t.TypeVar("CT")

_ArgT = t.TypeVar("_ArgT", tuple[str, ...], str)
_RT = t.TypeVar("_RT", str, bytes)
_T = t.TypeVar("_T")


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


def parse_structure_callback(
    inp: tuple[str, str], res: str | bytes
) -> GenericStructure:
    """
    Parse the fetched structure.

    :param inp: A pair of (id, fmt).
    :param res: The fetching result. By default, if ``fmt in ["cif", "pdb"]``,
        the result is ``str``, while ``fmt="mmtf"`` will produce ``bytes``.
    :return: Parse generic structure.
    """
    return GenericStructure.read(
        res, structure_id=inp[0], fmt=inp[1].removesuffix(".gz")
    )


def load_json_callback(_: t.Any, res: str) -> dict:
    """
    :param _: Arguments to the ``url_getter()`` (ignored).
    :param res: Fetched string content.
    :return: Parsed json as ``dict``.
    """
    return json.loads(res)


class StructureApiBase(ApiBase):
    """
    A generic abstract API to fetch structures and associated info.

    Child classes must implement :meth:`supported_str_formats` and have a url
    constructor named "structures" in :attr:`url_getters`.
    """

    @property
    @abstractmethod
    def supported_str_formats(self) -> list[str]:
        """
        :return: A list of formats supported by :meth:`fetch_structures`.
        """
        raise NotImplementedError

    def fetch_structures(
        self,
        ids: abc.Iterable[str],
        dir_: Path | None,
        fmt: str = "cif",
        *,
        overwrite: bool = False,
        parse: bool = False,
        callback: abc.Callable[[tuple[str, str], _RT], _T] | None = None,
    ) -> tuple[list[tuple[tuple[str, str], Path | _RT | _T]], list[tuple[str, str]]]:
        """
        Fetch structure files.

        PDB example:

        .. seealso::
            :func:`lXtractor.util.io.fetch_files`.

        .. hint::
            Callbacks will apply in parallel if :attr:`num_threads` is above 1.

        .. note::
            If the provided callback fails, it is equivalent to the fetching
            failure and will be presented as such. Initializing in verbose
            mode will output the stacktrace.

        Reading structures and parsing immediately requires using ``callback``.
        Such callback may be partially evaluated
        :meth:`lXtractor.core.structure.GenericStructure.read` encapsulating
        the correct format.

        :param ids: An iterable over structure IDs.
        :param dir_: Dir to save files to. If ``None``, will keep downloaded
            files as strings.
        :param fmt: Structure format. See :meth:`supported_str_formats`.
            Adding `.gz` will fetch gzipped files.
        :param overwrite: Overwrite existing files if `dir_` is provided.
        :param parse: If ``dir_ is None``, use :func:`parse_callback(fmt=fmt)`
            to parse fetched structures right away. This will override any
            existing `callback`.
        :param callback: If `dir_` is omitted, fetching will result in a
            ``bytes`` or a ``str``. Callback is a single-argument callable
            accepting the fetched content and returning anything.
        :return: A tuple with fetched results and the remaining IDs.
            The former is a list of tuples, where the first element
            is the original ID, and the second element is either the path to
            a downloaded file or downloaded data as string. The order
            may differ. The latter is a list of IDs that failed to fetch.
        """
        if fmt not in self.supported_str_formats:
            raise FormatError(f"Unsupported format {fmt}")

        if fmt == "mmtf":
            decode = False
            fmt += ".gz"
        elif fmt == "mmtf.gz":
            decode = False
        else:
            decode = True

        if parse and callback is None:
            callback = parse_structure_callback

        return fetch_files(
            self.url_getters["structures"],
            zip(ids, repeat(fmt)),
            fmt,
            dir_,
            callback=callback,
            overwrite=overwrite,
            decode=decode,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )

    def fetch_info(
        self,
        service_name: str,
        url_args: abc.Iterable[_ArgT],
        dir_: Path | None,
        *,
        overwrite: bool = False,
        callback: abc.Callable[[_ArgT, _RT], _T] | None = load_json_callback,
    ) -> tuple[list[tuple[_ArgT, dict | Path]], list[_ArgT]]:
        """
        Fetch text information.

        :param service_name: The name of the service to get a `url_getter` from
            :attr:`url_getters`.
        :param dir_: Dir to save files to. If ``None``, will keep downloaded
            files as strings.
        :param url_args: Arguments to a `url_getter`.
        :param overwrite: Overwrite existing files if `dir_` is provided.
        :param callback: Callback to apply after fetching the information file.
            By default, the content is assumed to be in ``json`` format. Thus,
            the default callback will parse the fetched content as ``dict``.
            To disable this behavior, pass ``callback=None``.
        :return: A tuple with fetched and remaining inputs.
            Fetched inputs are tuples, where the first element is the original
            arguments and the second argument is the dictionary with downloaded
            data. Remaining inputs are arguments that failed to fetch.
        """
        return fetch_files(
            self.url_getters[service_name],
            url_args,
            "json",
            dir_,
            overwrite=overwrite,
            callback=callback,
            decode=True,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )


if __name__ == "__main__":
    raise RuntimeError
