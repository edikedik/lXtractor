"""
Utilities to interact with the RCSB PDB database.
"""
import json
import logging
from collections import abc
from pathlib import Path

from toolz import valfilter

import lXtractor.util as util
from lXtractor.core.base import UrlGetter
from lXtractor.core.exceptions import FormatError
from lXtractor.ext.base import StructureApiBase

OBSOLETE_LINK = "https://files.wwpdb.org/pub/pdb/data/status/obsolete.dat"
SERVICES = (
    # Single-argument group
    ("chem_comp", "comp_id"),
    ("drugbank", "comp_id"),
    ("entry", "entry_id"),
    ("pubmed", "entry_id"),
    ("entry_groups", "group_id"),
    ("polymer_entity_groups", "group_id"),
    ("group_provenance", "group_provenance_id"),
    # Two-argument group
    ("assembly", "entry_id", "assembly_id"),
    ("branched_entity", "entry_id", "entity_id"),
    ("nonpolymer_entity", "entry_id", "entity_id"),
    ("polymer_entity", "entry_id", "entity_id"),
    ("branched_entity_instance", "entry_id", "asym_id"),
    ("nonpolymer_entity_instance", "entry_id", "asym_id"),
    ("polymer_entity_instance", "entry_id", "asym_id"),
    ("uniprot", "entry_id", "entity_id"),
    # Three-argument group
    ("interface", "entry_id", "assembly_id", "interface_id"),
)
LOGGER = logging.getLogger(__name__)


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

    def structures_url(entry_id, fmt):
        if fmt in ["pdb", "cif", "pdb.gz", "cif.gz"]:
            base = "https://files.rcsb.org/download/"
        elif fmt in ["mmtf", "mmtf.gz"]:
            fmt = "mmtf.gz"
            base = "https://mmtf.rcsb.org/v1.0/full"
        else:
            raise FormatError(f"Unrecognized file format {fmt} for entry {entry_id}")
        url = f"{base}/{entry_id}.{fmt}"
        return url

    result = {x[0]: url_getter_factory(*x) for x in SERVICES}
    result["structures"] = structures_url

    return result


class PDB(StructureApiBase):
    """
    Basic RCSB PDB interface to fetch structures and information.

    Example of fetching structures:

    >>> pdb = PDB()
    >>> fetched, failed = pdb.fetch_structures(['2src', '2oiq'], dir_=None)
    >>> len(fetched) == 2 and len(failed) == 0
    True
    >>> (args1, res1), (args2, res2) = fetched
    >>> assert {args1, args2} == {('2src', 'cif'), ('2oiq', 'cif')}
    >>> isinstance(res1, str) and isinstance(res2, str)
    True

    Example of fetching information:

    >>> pdb = PDB()
    >>> fetched, failed = pdb.fetch_info(
    ...     'entry', [('2SRC', ), ('2OIQ', )], dir_=None)
    >>> len(failed) == 0 and len(fetched) == 2
    True
    >>> (args1, res1), (args2, res2) = fetched
    >>> assert {args1, args2} == {('2SRC', ), ('2OIQ', )}
    >>> assert isinstance(res1, dict) and isinstance(res2, dict)

    .. hint::
        Check :meth:`list_services` to list available info services.

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
        return ["pdb", "cif", "mmtf", "pdb.gz", "cif.gz", "mmtf.gz"]

    @staticmethod
    def fetch_obsolete() -> dict[str, str]:
        """
        :return: A dict where keys are obsolete PDB IDs and values are
            replacement PDB IDs or an empty string if no replacement was made.
        """
        text = util.fetch_text(OBSOLETE_LINK, decode=True)
        lines = map(str.split, text.split("\n")[1:])
        return valfilter(
            bool, {x[2]: (x[3] if len(x) == 4 else "") for x in lines if len(x) >= 3}
        )


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
