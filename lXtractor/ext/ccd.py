import gzip
import logging
import os
import re
from collections import defaultdict, UserDict
from itertools import takewhile
from pathlib import Path

import msgpack
import pandas as pd
from more_itertools import split_at, split_before
from toolz import valmap, valfilter

from lXtractor.core.base import AbstractResource
from lXtractor.core.exceptions import MissingData
from lXtractor.util import fetch_to_file

LOGGER = logging.getLogger(__name__)
RESOURCES = Path(__file__).parent.parent / "resources"
RAW_CCD_NAME = "components.cif.gz"
PARSED_CCD_NAME = "CCD.msgpack"
MISSING_MSG = (
    "No parsed entries. Use `fetch` to obtain raw data and `parse` "
    "to parse and store the data."
)
URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
KV_PATTERN = re.compile(r"(_\w+)\.(\w+)\s+((?:[^;\n]+|;[\s\S]+?;)\s*)")


class Field(UserDict):
    """
    A straightforward ``dict`` extension with additional representation methods.
    """

    def as_df(self):
        try:
            val = next(iter(self.values()))
        except StopIteration:
            return pd.DataFrame()
        d = self
        if isinstance(val, str):
            d = valmap(lambda x: [x], d)
        return pd.DataFrame(d)


CCD_T = dict[str, dict[str, Field]]


def _wrap_fields(entries):
    return valmap(lambda x: valmap(lambda f: Field(f), x), entries)


def _unwrap_fields(entries):
    return valmap(lambda x: valmap(lambda f: dict(f), x), entries)


def _parse_key_value(data: list[str]):
    values_ = {}

    matches = KV_PATTERN.findall("\n".join(data))
    group = None

    for group, name, val in matches:
        values_[name] = val.replace("\n", "").strip(";").strip()

    return group, Field(values_)


def _parse_loop(data: list[str]) -> tuple[str, Field]:
    def parse_line(line: str):
        return re.findall(r"\"[^\"]+\"|\S+", line)

    group = data[0].split(".")[0]
    header = dict(
        enumerate(
            map(lambda x: x.split(".")[1], takewhile(lambda x: x.startswith("_"), data))
        )
    )
    values = data[len(header) + 1 :]
    values_ = defaultdict(list)

    multiline_mode = False
    multiline_buffer = ""
    all_values_str = ""

    for row in values:
        if row.startswith(";"):
            if multiline_mode:  # End of multiline
                multiline_mode = False
                all_values_str += " " + multiline_buffer
                multiline_buffer = ""
            else:  # Start of multiline
                multiline_mode = True
                # Remove the leading semicolon
                multiline_buffer = row[1:].strip()
            continue

        if multiline_mode:
            multiline_buffer += " " + row.strip()
            continue

        all_values_str += " " + row.strip()

    all_values_list = parse_line(all_values_str)

    for i, v in enumerate(all_values_list):
        header_key = header[i % len(header)]
        values_[header_key].append(v)

    return group, Field(values_)


def _parse_entry(data: list[str]):
    entry_id = data[0].removeprefix("data_")
    splits = filter(lambda x: len(x) > 1, split_at(data, lambda x: x.startswith("#")))
    parsed_splits = dict(
        _parse_loop(x[1:]) if x[0].startswith("loop") else _parse_key_value(x)
        for x in splits
    )
    return entry_id, parsed_splits


class CCD(AbstractResource, UserDict[str, dict[str, Field]]):
    """
    `Chemical Component Dictionary resource <https://www.wwpdb.org/data/ccd>`_.

    CCD is represented as a dictionary, where keys are PDB three-letter codes
    and values are also dictionaries, where keys are CCD groups (e.g.,
    "_chem_comp") that point to :class:`Field` objects.

    On the first use, one has to download and parse the CCD data:

    >>> ccd = CCD()
    >>> _ = ccd.fetch()
    >>> _ = ccd.parse()

    This will fetch, parse, and store the resource locally as a single `msgpack`
    serialized file. These data will be automatically loaded upon future
    initializations and can be accessed using ``dict`` syntax:

    >>> ccd = CCD()
    >>> chem_comp = ccd['ATP']['_chem_comp']
    >>> chem_comp['name']
    '"ADENOSINE-5\\'-TRIPHOSPHATE"'
    >>> bonds = ccd['ATP']['_chem_comp_bond'].as_df()
    >>> list(bonds.columns)[:4]
    ['comp_id', 'atom_id_1', 'atom_id_2', 'value_order']

    """

    def __init__(
        self,
        resource_path: str | Path = RESOURCES / PARSED_CCD_NAME,
        resource_name: str | None = "CCD",
        read_entries: bool = True,
    ):
        AbstractResource.__init__(self, resource_path, resource_name)
        UserDict.__init__(self)

        if read_entries and self.path.exists():
            self.read()

    def fetch(self, url: str = URL, overwrite: bool = False) -> Path:
        raw_path = RESOURCES / RAW_CCD_NAME
        if raw_path.exists() and not overwrite:
            LOGGER.info("Raw CCD download exists and will not be overwritten")
            return raw_path

        LOGGER.info(f"Fetching CCD to {raw_path}")
        fetch_to_file(url, raw_path)

        return raw_path

    def parse(
        self,
        path: Path = RESOURCES / RAW_CCD_NAME,
        overwrite: bool = False,
        store_to_resources: bool = True,
        rm_raw: bool = True,
    ) -> CCD_T:
        if self.path.exists() and store_to_resources and not overwrite:
            raise RuntimeError(
                "Resources was parsed. Pass overwrite if you want to overwrite."
            )
        if not path.exists():
            raise MissingData("Missing raw resource. Try calling `fetch` method first.")

        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = filter(bool, map(lambda x: x.removesuffix("\n"), f))
            splits = list(split_before(lines, lambda x: x.startswith("data_")))
            self.data = dict(map(_parse_entry, splits))

        if store_to_resources:
            self.dump(self.path)

        if rm_raw:
            os.remove(path)

        return self.data

    def dump(self, path: Path) -> Path:
        if self.data is None:
            raise MissingData("No entries to save")
        packed = msgpack.packb(_unwrap_fields(self.data))
        with self.path.open("wb") as f:
            f.write(packed)
            LOGGER.info(f"Stored parsed resource to {self.path}")
        return path

    def read(self) -> CCD_T:
        if not self.path.exists():
            raise MissingData(MISSING_MSG)
        with self.path.open("rb") as f:
            unpacker = msgpack.Unpacker(f)
            self.data = _wrap_fields(unpacker.unpack())

        return self.data

    def make_res_name_map(self, store_to_resources: bool = True) -> dict[str, str]:
        if self.data is None:
            raise MissingData(MISSING_MSG)
        m = {k: v["_chem_comp"]["one_letter_code"] for k, v in self.items()}
        m = valfilter(lambda x: x != "?" and len(x) == 1, m)

        if store_to_resources:
            from lXtractor.core.base import ALL21

            path = ALL21.absolute()
            with path.open("wb") as f:
                f.write(msgpack.packb(m))
                LOGGER.info(f"Stored residue mapping to {path}")

        return m


if __name__ == "__main__":
    raise RuntimeError
