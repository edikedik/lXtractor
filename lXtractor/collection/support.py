from __future__ import annotations

import json
import typing as t
from abc import ABC, abstractmethod
from collections import abc
from dataclasses import dataclass, field
from itertools import chain, groupby
from pathlib import Path

from more_itertools import unique_everseen
from toolz import curry

import lXtractor.chain as lxc
from lXtractor.core.config import Config, DefaultConfig
from lXtractor.core.exceptions import MissingData, FormatError
from lXtractor.ext import AlphaFold, PDB, UniProt, SIFTS

_RESOURCES = Path(__file__).parent.parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCES / "collection_config.json"
_USER_CONFIG_PATH = _RESOURCES / "collection_user_config.json"

_CT = t.TypeVar("_CT", lxc.ChainSequence, lxc.ChainStructure, lxc.Chain)


def _parse_str_id(x: str) -> str | tuple[str, tuple[str, ...]]:
    if ":" in x:
        id_, chains = x.split(":", maxsplit=1)
        return id_, tuple(chains.split(","))
    return x


def _validate_seq_id(x: t.Any) -> str:
    if not isinstance(x, str):
        raise TypeError(f"Expected sequence ID to be of string type, got {type(x)}.")
    return x


class ConstructorConfig(Config):
    def __init__(
        self,
        default_config_path: str | Path = _DEFAULT_CONFIG_PATH,
        user_config_path: str | Path = _USER_CONFIG_PATH,
        **kwargs,
    ):
        self.provided_settings = kwargs
        super().__init__(default_config_path, user_config_path)

    @property
    def nullable_fields(self) -> tuple[str, ...]:
        """
        :return: A tuple of fields that can have ``None``/``null`` values.
        """
        return "child_filter", "parent_filter", "child_callback", "parent_callback"

    def reload(self):
        """
        Reload the configuration from files and initially
        :attr:`provided_settings`
        """
        super().reload()
        self.update_with(self.provided_settings)

    def save(self, user_config_path: str | Path = _USER_CONFIG_PATH):
        super().save(user_config_path)

    @classmethod
    def list_fields(cls) -> list[str]:
        with _DEFAULT_CONFIG_PATH.open("r") as f:
            return list(json.load(f))

    def list_missing_fields(self) -> list[str]:
        return [k for k, v in self.data.items() if v is None]

    def validate(self) -> None:
        none_keys = ", ".join(
            filter(lambda x: x not in self.nullable_fields, self.list_missing_fields())
        )
        if none_keys:
            raise MissingData(f"Missing values for required keys: {none_keys}")


@dataclass
class CollectionPaths:
    output: Path
    references: Path
    sequences: Path
    structures: Path

    str_fmt: str

    @property
    def structure_files(self) -> Path:
        return self.structures / self.str_fmt

    @property
    def sequence_files(self) -> Path:
        return self.sequences / "fasta"

    @property
    def structures_info(self) -> Path:
        return self.structures / "info"

    @property
    def chains(self) -> Path:
        return self.output / "chains"

    def get_all(self):
        return (
            self.output,
            self.references,
            self.sequences,
            self.structures,
            self.structure_files,
            self.structures_info,
            self.sequence_files,
            self.chains,
        )

    def mkdirs(self):
        for d in self.get_all():
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class Interfaces:
    AlphaFold: AlphaFold
    PDB: PDB
    SIFTS: SIFTS | None
    UniProt: UniProt
    Initializer: lxc.ChainInitializer
    IO: lxc.ChainIO

    def get_fetchers(
        self, paths: CollectionPaths, str_fmt: str, **kwargs
    ) -> dict[str, abc.Callable[[abc.Iterable[str],], t.Any]]:
        return {
            "uniprot": curry(self.UniProt.fetch_sequences)(
                dir_=paths.sequence_files, **kwargs
            ),
            "pdb": curry(self.PDB.fetch_structures)(
                dir_=paths.structure_files, fmt=str_fmt, **kwargs
            ),
            "alphafold": curry(self.AlphaFold.fetch_structures)(
                dir_=paths.structure_files, fmt=str_fmt, **kwargs
            ),
        }


def _group_str_items(items: abc.Iterable[StrItem]) -> abc.Iterator[tuple[str, str]]:
    key = lambda s: s.str_id
    for g, gg in groupby(sorted(items, key=key), key=key):
        g_chains = ",".join(x.str_chain for x in gg)
        yield g, g_chains


class ConstructorItem(ABC, t.Generic[_CT]):
    @classmethod
    @abstractmethod
    def from_chain(cls, c: _CT) -> t.Self:
        ...

    @classmethod
    @abstractmethod
    def from_str(cls, inp: str) -> abc.Iterator[t.Self]:
        ...

    @abstractmethod
    def to_str(self) -> str:
        ...

    @abstractmethod
    def make_path(self, paths: CollectionPaths, fmt: str | tuple[str, str]):
        ...


@dataclass(frozen=True)
class SeqItem(ConstructorItem[lxc.ChainSequence]):
    seq_id: str

    @classmethod
    def from_chain(cls, c: lxc.ChainSequence) -> t.Self:
        return cls(c.name)

    @classmethod
    def from_str(cls, inp: str) -> abc.Iterator[t.Self]:
        yield cls(inp)

    def to_str(self) -> str:
        return self.seq_id

    def make_path(self, paths: CollectionPaths, fmt: str = "fasta") -> Path:
        return paths.sequence_files / f"{self.seq_id}.{fmt}"


@dataclass(frozen=True)
class StrItem(ConstructorItem[lxc.ChainStructure]):
    str_id: str
    str_chain: str

    @classmethod
    def from_chain(cls, c: lxc.ChainStructure) -> t.Self:
        return cls(c.meta[DefaultConfig["metadata"]["structure_id"]], c.chain_id)

    @classmethod
    def from_str(cls, inp: str) -> abc.Iterator[t.Self]:
        if ":" not in inp:
            raise FormatError(f"Invalid input format in {inp}.")
        str_id, chain_ids = inp.split(":", maxsplit=1)
        for chain_id in chain_ids.split(","):
            yield cls(str_id, chain_id)

    def to_str(self) -> str:
        return f"{self.str_id}:{self.str_chain}"

    def make_path(self, paths: CollectionPaths, fmt: str) -> Path:
        return paths.sequence_files / f"{self.str_id}.{fmt}"


@dataclass(frozen=True)
class MapItem(ConstructorItem[lxc.Chain]):
    seq_item: SeqItem
    str_items: abc.Sequence[StrItem]

    @classmethod
    def from_chain(cls, c: lxc.Chain) -> t.Self:
        return cls(
            SeqItem.from_chain(c.seq), list(map(StrItem.from_chain, c.structures))
        )

    @classmethod
    def from_str(cls, inp: str) -> abc.Iterator[t.Self]:
        seq_inp, str_inps = inp.split("=>", maxsplit=1)
        yield cls(
            next(SeqItem.from_str(seq_inp)),
            list(chain.from_iterable(map(StrItem.from_str, str_inps.split(";")))),
        )

    def to_str(self) -> str:
        str_joined = ";".join(map(":".join, _group_str_items(self.str_items)))
        return f"{self.seq_item.to_str()}=>{str_joined}"

    def make_path(
        self, paths: CollectionPaths, fmt: tuple[str, str]
    ) -> tuple[Path, list[tuple[Path, abc.Sequence[str]]]]:
        seq_fmt, str_fmt = fmt
        str_paths = [
            (paths.structure_files / str_id, str_chains)
            for str_id, str_chains in _group_str_items(self.str_items)
        ]
        return self.seq_item.make_path(paths, seq_fmt), str_paths


def _parse_inp_ids(inp_ids: abc.Iterable[str]) -> abc.Iterator[str]:
    for x in inp_ids:
        if ":" in x:
            id_, chains = x.split(":", maxsplit=1)
            for c in chains.split(","):
                yield f"{id_}:{c}"
        else:
            yield x


def _group_join_chains(ids: abc.Iterable[str]) -> abc.Iterator[str]:
    def _get_chains(x: str):
        if ":" in x:
            return x.split(":")[1].split(",")
        return []

    groups = groupby(sorted(ids), lambda x: x.split(":")[0])
    for g, gg in groups:
        chains = ",".join(chain.from_iterable(map(_get_chains, gg)))
        if chains:
            yield f"{g}:{chains}"
        else:
            yield g


@dataclass(repr=False)
class BatchData:
    i: int
    ids_in: abc.Sequence[str]
    ids_out: abc.Sequence[str]
    chains: lxc.ChainList[_CT] | None
    failed: bool = False

    def __post_init__(self):
        self._ids_in_parsed = tuple(unique_everseen(_parse_inp_ids(self.ids_in)))
        self._ids_out_parsed = tuple(
            unique_everseen(x.split("|")[0] for x in self.ids_out)
        )

    def filter_done(self) -> abc.Iterator[str]:
        yield from filter(lambda x: x in self._ids_out_parsed, self._ids_in_parsed)

    def filter_missed(self) -> abc.Iterator[str]:
        yield from filter(lambda x: x not in self._ids_out_parsed, self._ids_in_parsed)


@dataclass(repr=False)
class BatchesHistory(t.Generic[_CT]):
    data: list[BatchData] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)

    def join_chains(self) -> lxc.ChainList[_CT]:
        chains = (x.chains for x in self.data if x.chains is not None)
        return lxc.ChainList(chain.from_iterable(chains))

    def last_step(self) -> int:
        if len(self.data) == 0:
            return 0
        return self.data[-1].i

    def cleanup(self) -> None:
        self.data = []

    def iter_done(self) -> abc.Iterator[str]:
        ids = chain.from_iterable(bd.filter_done() for bd in self.data)
        yield from _group_join_chains(ids)

    def iter_tried(self) -> abc.Iterator[str]:
        yield from unique_everseen(chain.from_iterable(b.ids_in for b in self.data))

    def iter_missed(self) -> abc.Iterator[str]:
        ids = chain.from_iterable(bd.filter_missed() for bd in self.data)
        yield from _group_join_chains(ids)

    def iter_failed(self):
        yield from (b for b in self.data if b.failed)

    def iter_failed_ids(self):
        yield from chain.from_iterable(b.ids_in for b in self.iter_failed())


if __name__ == "__main__":
    raise RuntimeError
