"""
Utilities to interact with the RCSB PDB database.
"""
import json
import typing as t
from collections import abc
from itertools import repeat
from pathlib import Path

from toolz import valfilter

from lXtractor.core.base import UrlGetter
from lXtractor.core.exceptions import FormatError
from lXtractor.ext.base import ApiBase
from lXtractor.protocols import LOGGER
from lXtractor.util.io import fetch_files, fetch_text

# ArgT: t.TypeAlias = tuple[str, ...] | str
ArgT = t.TypeVar("ArgT", tuple[str, ...], str)
OBSOLETE_LINK = "https://files.wwpdb.org/pub/pdb/data/status/obsolete.dat"
SERVICES = (
    # Single argument group
    ("chem_comp", "comp_id"),
    ("drugbank", "comp_id"),
    ("entry", "entry_id"),
    ("pubmed", "entry_id"),
    ("entry_groups", "group_id"),
    ("polymer_entity_groups", "group_id"),
    ("group_provenance", "group_provenance_id"),
    # Two arguments group
    ("assembly", "entry_id", "assembly_id"),
    ("branched_entity", "entry_id", "entity_id"),
    ("nonpolymer_entity", "entry_id", "entity_id"),
    ("polymer_entity", "entry_id", "entity_id"),
    ("branched_entity_instance", "entry_id", "asym_id"),
    ("nonpolymer_entity_instance", "entry_id", "asym_id"),
    ("polymer_entity_instance", "entry_id", "asym_id"),
    ("uniprot", "entry_id", "entity_id"),
    # Three argument group
    ("interface", "entry_id", "assembly_id", "interface_id"),
)


def url_getters() -> dict[str, UrlGetter]:
    """
    :return: A dictionary with {name: getter} where getter is a function
        accepting string args and returning a valid URL.
    """

    def url_getter_factory(name, *args):
        args_fn = ", ".join(args)
        args_url = "/".join(f"{{{x}}}" for x in args)
        base = "https://data.rcsb.org/rest/v1/core"
        fn = f'lambda {args_fn}: f"{base}/{name}/{args_url}"'
        return eval(fn)  # pylint: disable=eval-used

    def files_url(entry_id, fmt):
        if fmt in ['pdb', 'cif', 'pdb.gz', 'cif.gz']:
            base = "https://files.rcsb.org/download/"
        elif fmt in ['mmtf', 'mmtf.gz']:
            fmt = 'mmtf.gz'
            base = "https://mmtf.rcsb.org/v1.0/full"
        else:
            raise FormatError(f'Unrecognized file format {fmt} for entry {entry_id}')
        url = f'{base}/{entry_id}.{fmt}'
        return url

    result = {x[0]: url_getter_factory(*x) for x in SERVICES}
    result['files'] = files_url

    return result


class PDB(ApiBase):
    """
    Basic RCSB PDB interface to fetch structures and information.

    Fetch structure files from RCSB PDB.

    >>> pdb = PDB()
    >>> fetched, failed = pdb.fetch_structures(['2src', '2oiq'],)
    >>> len(fetched) == 2 and len(failed) == 0
    True
    >>> (id1, res1), (id2, res2) = fetched
    >>> assert {id1, id2} == {'2src', '2oiq'}
    >>> isinstance(res1, str) and isinstance(res2, str)
    True
    """

    def __init__(
        self, max_trials: int = 1, num_threads: int | None = None, verbose: bool = False
    ):
        super().__init__(url_getters(), max_trials, num_threads, verbose)

    @property
    def supported_str_formats(self) -> list[str]:
        """
        :return: A list of formats supported by :meth:`fetch_structures`.
        """
        return ['.pdb', '.cif', '.mmtf']

    def fetch_structures(
        self,
        ids: abc.Iterable[str],
        dir_: Path | None,
        fmt: str = "cif",
        *,
        overwrite: bool = False,
    ) -> tuple[list[tuple[tuple[str, str], Path | str]], list[tuple[str, str]]]:
        """
        Fetch structure files from the PDB resources.

        >>> pdb = PDB()
        >>> fetched, failed = pdb.fetch_structures(['2src', '2oiq'], dir_=None)
        >>> len(fetched) == 2 and len(failed) == 0
        True
        >>> (args1, res1), (args2, res2) = fetched
        >>> assert {args1, args2} == {('2src', 'cif'), ('2oiq', 'cif')}
        >>> isinstance(res1, str) and isinstance(res2, str)
        True

        .. seealso::
            :func:`fetch_files lXtractor.ext.base.fetch_files`.

        :param ids: An iterable over PDB IDs.
        :param dir_: Dir to save files to. If ``None``, will keep downloaded
            files as strings.
        :param fmt: Structure format. See :meth:`supported_str_formats`.
            Adding `.gz` will fetch gzipped files.
        :param overwrite: Overwrite existing files if `dir_` is provided.
        :return: A tuple with fetched results and the remaining IDs.
            The former is a list of tuples, where the first element
            is the original ID, and the second element is either the path to
            a downloaded file or downloaded data as string. The order
            may differ. The latter is a list of IDs that failed to fetch.
        """
        if fmt == 'mmtf':
            fmt += '.gz'

        return fetch_files(
            self.url_getters['files'],
            zip(ids, repeat(fmt)),
            fmt,
            dir_,
            overwrite=overwrite,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )

    def fetch_info(
        self,
        service_name: str,
        url_args: abc.Iterable[ArgT],
        dir_: Path | None,
        *,
        overwrite: bool = False,
    ) -> tuple[list[tuple[ArgT, dict | Path]], list[ArgT]]:
        """

        >>> pdb = PDB()
        >>> fetched, failed = pdb.fetch_info(
        ...     'entry', [('2SRC', ), ('2OIQ', )], dir_=None)
        >>> len(failed) == 0 and len(fetched) == 2
        True
        >>> (args1, res1), (args2, res2) = fetched
        >>> assert {args1, args2} == {('2SRC', ), ('2OIQ', )}
        >>> assert isinstance(res1, dict) and isinstance(res2, dict)

        .. seealso:
            :meth:`list_services` for a list of services and url args.

        :param service_name: The name of the service to use.
        :param dir_: Dir to save files to. If ``None``, will keep downloaded
            files as strings.
        :param url_args: Arguments to a `url_getter`. Check :meth:`list_services`
            to see which getters require which arguments. Each element of this
            iterable is a tuple of string arguments that should produce a valid
            url for the API.
        :param overwrite: Overwrite existing files if `dir_` is provided.
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
            callback=json.loads,
            overwrite=overwrite,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )

    @staticmethod
    def fetch_obsolete() -> dict[str, str]:
        """
        :return: A dict where keys are obsolete PDB IDs and values are
            replacement PDB IDs or an empty string if no replacement was made.
        """
        text = fetch_text(OBSOLETE_LINK, decode=True)
        lines = map(str.split, text.split("\n")[1:])
        return valfilter(
            bool, {x[2]: (x[3] if len(x) == 4 else "") for x in lines if len(x) >= 3}
        )

    @staticmethod
    def list_services() -> list[tuple[str, ...]]:
        """
        :return: List of services to be used with :meth:`fetch_info`. Each entry
            is a tuple where the first element is the service name and the rest
            are the argument names required to provide when using a service
            (``url_getters``).
        """
        return list(SERVICES)


def filter_by_method(
    pdb_ids: abc.Iterable[str],
    pdb: PDB = PDB(),
    method: str = "X-ray",
    dir_: Path | None = None,
) -> list[str]:
    """
    .. seealso::
        :meth:`PDB.fetch_info <lXtractor.ext.pdb_.PDB.fetch_info>`

    .. note::
        Keys for the info dict are 'rcsb_entry_info' -> 'experimental_method'

    :param pdb_ids: An iterable over PDB IDs.
    :param pdb: Fetcher instance. If not provided, will init with
        default params.
    :param method: Method to match. Must correspond exactly.
    :param dir_: Dir to save info "entry" json dumps.
    :return: A list of PDB IDs obtained by desired experimental
        procedure.
    """

    def method_matches(d: dict) -> bool:
        try:
            return d["rcsb_entry_info"]["experimental_method"] == method
        except KeyError as e:
            LOGGER.warning(f"Missing required key {e}")
            return False

    def get_existing(ids: abc.Iterable[str], _dir: Path) -> list[tuple[str, Path]]:
        res = ((x, (_dir / f"{x}.json")) for x in ids)
        return [x for x in res if x[1].exists()]

    def load_file(inp: str | Path | dict, base: Path | None) -> dict:
        try:
            if isinstance(inp, dict):
                return inp
            if isinstance(inp, str):
                assert base is not None, "base path provided with base filename"
                inp = base / f"{inp}.json"
            with inp.open() as f:
                res = json.load(f)
                assert isinstance(res, dict), "loaded json correctly"
                return res
        except FileNotFoundError:
            LOGGER.warning(f"Missing supposedly fetched {inp}")
            return {}

    pdb_ids = list(pdb_ids)
    existing = get_existing(pdb_ids, dir_) if dir_ is not None else []
    fetched, missed = pdb.fetch_info("entry", pdb_ids, dir_)
    fetched += existing
    fetched = [(x[0], load_file(x[1], dir_)) for x in fetched]

    if missed:
        missed_display = ",".join(missed) if len(missed) < 100 else ""
        LOGGER.warning(f"Failed to fetch {len(missed)} ids: {missed_display}")

    # fails to recognize x[1] must be dict
    return [x[0] for x in fetched if method_matches(x[1])]  # type: ignore


if __name__ == "__main__":
    raise RuntimeError
