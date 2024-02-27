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
        return (
            "default_chain",
            "child_filter",
            "parent_filter",
            "child_callback",
            "parent_callback",
        )

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

    @property
    def references(self) -> Path:
        return self.output / "references"

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


class ConstructorItem(ABC, t.Generic[_CT]):
    @classmethod
    @abstractmethod
    def from_chain(cls, c: _CT) -> abc.Iterator[t.Self]:
        ...

    @classmethod
    @abstractmethod
    def from_str(cls, inp: str) -> abc.Iterator[t.Self]:
        ...


@dataclass(frozen=True)
class SeqItem(ConstructorItem[lxc.ChainSequence]):
    seq_id: str

    @classmethod
    def from_chain(cls, c: lxc.ChainSequence) -> abc.Iterator[t.Self]:
        yield cls(c.name)

    @classmethod
    def from_str(cls, inp: str) -> abc.Iterator[t.Self]:
        yield cls(inp)


@dataclass(frozen=True)
class StrItem(ConstructorItem[lxc.ChainStructure]):
    str_id: str
    chain_id: str

    @classmethod
    def from_chain(cls, c: lxc.ChainStructure) -> abc.Iterator[t.Self]:
        yield cls(c.meta[DefaultConfig["metadata"]["structure_id"]], c.chain_id)

    @classmethod
    def from_str(cls, inp: str) -> abc.Iterator[t.Self]:
        str_id, chain_ids = inp.split(":", maxsplit=1)
        for chain_id in chain_ids.split(","):
            yield cls(str_id, chain_id)

    @classmethod
    def from_tuple(cls, inp: tuple[str, abc.Sequence[str]]) -> abc.Iterator[t.Self]:
        if isinstance(inp[1], str):
            raise FormatError(
                f"Strings are disallowed as a second element in input {inp}"
            )
        for chain_id in inp[1]:
            yield cls(inp[0], chain_id)


@dataclass(frozen=True)
class MapItem(ConstructorItem[lxc.Chain]):
    seq_item: SeqItem
    str_item: StrItem

    @classmethod
    def from_chain(cls, c: lxc.Chain) -> abc.Iterator[t.Self]:
        seq_item = next(SeqItem.from_chain(c.seq))
        for s in c.structures:
            yield cls(seq_item, next(StrItem.from_chain(s)))

    @classmethod
    def from_str(cls, inp: str) -> abc.Iterator[t.Self]:
        seq_inp, str_inps = inp.split("=>", maxsplit=1)
        seq_item = next(SeqItem.from_str(seq_inp))
        for s in str_inps.split(";"):
            for str_item in StrItem.from_str(s):
                yield cls(seq_item, str_item)

    @classmethod
    def from_tuple(
        cls,
        inp: tuple[
            str, str | abc.Sequence[str] | abc.Sequence[tuple[str, abc.Sequence[str]]]
        ],
    ) -> abc.Iterator[t.Self]:
        match inp:
            case (str(), str()):
                seq_item = SeqItem(inp[0])
                str_items = StrItem.from_str(inp[1])
                for str_item in str_items:
                    yield cls(seq_item, str_item)
            case (str(), abc.Sequence()) if all(isinstance(x, str) for x in inp[1]):
                seq_item = SeqItem(inp[0])
                str_items = chain.from_iterable(map(StrItem.from_str, inp[1]))
                for str_item in str_items:
                    yield cls(seq_item, str_item)
            case (str(), abc.Sequence()) if all(
                isinstance(x, tuple)
                and len(x) == 2
                and isinstance(x[0], str)
                and isinstance(x[1], abc.Sequence)
                for x in inp[1]
            ):
                seq_item = SeqItem(inp[0])
                str_items = chain.from_iterable(map(StrItem.from_tuple, inp[1]))
                for str_item in str_items:
                    yield cls(seq_item, str_item)
            case _:
                raise ValueError(f"Invalid input format for {inp}.")


_IT = t.TypeVar("_IT", SeqItem, StrItem, MapItem)


class ItemList(ABC, list, t.Generic[_IT]):
    @abstractmethod
    def prep_for_init(self, paths: CollectionPaths):
        pass

    @abstractmethod
    def as_strings(self):
        pass

    @property
    @abstractmethod
    def item_type(self) -> t.Type[_IT]:
        pass


class SeqItemList(ItemList[SeqItem]):
    @property
    def item_type(self) -> t.Type[SeqItem]:
        return SeqItem

    def prep_for_init(self, paths: CollectionPaths):
        yield from (
            paths.sequence_files / f"{seq_item.seq_id}.fasta" for seq_item in self
        )

    def as_strings(self):
        yield from (s.seq_id for s in self)


class StrItemList(ItemList[StrItem]):
    @property
    def item_type(self) -> t.Type[StrItem]:
        return StrItem

    def iter_groups(self):
        key = lambda s: s.str_id
        yield from groupby(sorted(self, key=key), key=key)

    def prep_for_init(self, paths: CollectionPaths):
        for g, gg in self.iter_groups():
            str_path = paths.structure_files / f"{g}.{paths.str_fmt}"
            yield str_path, [it.chain_id for it in gg]

    def as_strings(self):
        for g, gg in self.iter_groups():
            chains = ",".join(it.chain_id for it in gg)
            yield f"{g}:{chains}"


class MapItemList(ItemList[MapItem]):
    @property
    def item_type(self) -> t.Type[MapItem]:
        return MapItem

    def iter_groups(self):
        key = lambda s: s.seq_item.seq_id
        yield from groupby(sorted(self, key=key), key=key)

    def prep_for_init(self, paths: CollectionPaths):
        for g, gg in self.iter_groups():
            str_items = StrItemList(it.str_item for it in gg)
            seq_path = paths.sequence_files / f"{g}.fasta"
            yield seq_path, list(str_items.prep_for_init(paths))

    def as_strings(self):
        for g, gg in self.iter_groups():
            str_items = StrItemList(it.str_item for it in gg)
            strs = ";".join(str_items.as_strings())
            yield f"{g}=>{strs}"


_ITL = t.TypeVar("_ITL", SeqItemList, StrItemList, MapItemList)


@dataclass(frozen=True, repr=False)
class BatchData(t.Generic[_ITL, _CT]):
    i: int
    items_in: _ITL
    items_out: _ITL
    chains: lxc.ChainList[_CT] | None
    failed: bool = False

    @property
    def item_list_type(self) -> t.Type[_ITL]:
        return self.items_in.__class__

    def items_done(self) -> _ITL:
        return self.item_list_type(set(self.items_out) & set(self.items_in))

    def items_missed(self) -> _ITL:
        return self.item_list_type(set(self.items_in) - set(self.items_out))


@dataclass(repr=False)
class BatchesHistory(t.Generic[_ITL, _CT]):
    _item_list_type: t.Type[_ITL]
    data: list[BatchData[_ITL, _CT]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> BatchData[_ITL, _CT]:
        return self.data.__getitem__(i)

    def _init_item_list(self, items: abc.Iterable[_ITL]) -> _ITL:
        return self._item_list_type(unique_everseen(chain.from_iterable(items)))

    def join_chains(self) -> lxc.ChainList[_CT]:
        chains = (x.chains for x in self.data if x.chains is not None)
        return lxc.ChainList(chain.from_iterable(chains))

    @property
    def last_step(self) -> int:
        if len(self.data) == 0:
            return 0
        return self.data[-1].i

    def cleanup(self) -> None:
        self.data = []

    def items_done(self) -> _ITL:
        return self._init_item_list(x.items_done() for x in self.data)

    def items_missed(self) -> set[str]:
        return self._init_item_list(x.items_missed() for x in self.data)

    def items_tried(self):
        return self._init_item_list(x.items_in for x in self.data)

    def items_failed(self):
        return self._init_item_list(x.items_in for x in self.iter_failed())

    def iter_failed(self):
        yield from (b for b in self.data if b.failed)


if __name__ == "__main__":
    raise RuntimeError
