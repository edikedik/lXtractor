from __future__ import annotations

import json
import os
import sys
import typing as t
from abc import ABCMeta, abstractmethod
from collections import abc
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from loguru import logger
from toolz import curry

import lXtractor.chain as lxc
from lXtractor.collection import (
    ChainCollection,
    StructureCollection,
    SequenceCollection,
)
from lXtractor.core import Alignment
from lXtractor.core.config import Config
from lXtractor.core.exceptions import MissingData, ConfigError
from lXtractor.ext import PyHMMer, Pfam, AlphaFold, PDB, UniProt, SIFTS
from lXtractor.util import read_fasta

_RESOURCES = Path(__file__).parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCES / "collection_config.json"
_USER_CONFIG_PATH = _RESOURCES / "collection_user_config.json"
_CTA: t.TypeAlias = SequenceCollection | StructureCollection | ChainCollection
_CT = t.TypeVar("_CT", SequenceCollection, StructureCollection, ChainCollection)
_SeqIt: t.TypeAlias = str
_StrIt: t.TypeAlias = tuple[str, tuple[str, ...]]
_MapIt: t.TypeAlias = tuple[_SeqIt, abc.Sequence[_StrIt]]
_IT = t.TypeVar("_IT", _SeqIt, _StrIt, _MapIt)
_U = t.TypeVar("_U", bound=int | float | str | None)
_T = t.TypeVar("_T")


# TODO: all configs should be stored within the database to be reusable


def _to_concrete_collection(desc: str) -> t.Type[_CT]:
    match desc[:3].lower():
        case "cha":
            return ChainCollection
        case "seq":
            return SequenceCollection
        case "str":
            return StructureCollection
        case _:
            raise NameError(f"Cannot determine collection from parameter {desc}")


def _setup_collection(
    config: ConstructorConfig, prefix: str, sources: abc.Iterable[str]
) -> _CTA:
    ct = _to_concrete_collection(prefix)

    config["collection_type"] = prefix

    if config["collection_name"] == "auto":
        config["collection_name"] = f"{prefix}Collection"
    if config["source"].lower() not in ("local", *sources):
        raise ConfigError(
            f"Invalid source {config['source']} for a {prefix} collection"
        )
    coll_path = config["out_dir"] / f"{config['collection_name']}.sqlite"
    return ct(coll_path)


def _seqs_to_hmm(
    seqs: abc.Iterable[(str, str)],
    name: str,
    alphabet: str,
) -> PyHMMer:
    if len(set(map(len, (x[1] for x in seqs)))) == 1:
        aln = seqs
    else:
        aln = Alignment.make(seqs)
    return PyHMMer.from_msa(aln, name, alphabet)


def _read_reference(inp: Path | (str, Path), alphabet: str):
    if isinstance(inp, Path):
        name, path = inp.stem, inp
    else:
        name, path = inp

    match path.suffix:
        case ".fasta":
            seqs = list(read_fasta(path))
            return _seqs_to_hmm(seqs, name, alphabet)
        case ".hmm":
            return PyHMMer(path)
        case _:
            raise NameError(
                f"The input path must end with '.fasta' or '.hmm'. "
                f"Got {path.suffix} for path {path}."
            )


def _init_reference(
    inp: Path
    | Alignment
    | lxc.ChainSequence
    | (str, Path)
    | (str, Alignment)
    | (str, lxc.ChainSequence)
    | (str, str)
    | PyHMMer
    | str,
    alphabet: str,
) -> PyHMMer:
    match inp:
        case PyHMMer():
            return inp
        case Path() | (str(), Path()):
            return _read_reference(inp, alphabet)
        case Alignment():
            return PyHMMer.from_msa(inp, inp[0][0], alphabet)
        case lxc.ChainSequence():
            return PyHMMer.from_msa([(inp.name, inp.seq1)], inp.name, alphabet)
        case (str(), Alignment()):
            return PyHMMer.from_msa(inp[1], inp[0], alphabet)
        case (str(), str()) | (str(), lxc.ChainSequence()):
            name, seq = inp
            if isinstance(seq, lxc.ChainSequence):
                seq = seq.seq1
            return PyHMMer.from_msa([(name, seq)], name, alphabet)
        case str():
            return Pfam()[inp]
        case _:
            raise TypeError(f"Failed to parse input reference {inp}")


def _parse_str_id(x: str) -> str | tuple[str, tuple[str, ...]]:
    if ":" in x:
        id_, chains = x.split(":", maxsplit=1)
        return id_, tuple(chains.split(","))
    return x


def _validate_seq_id(x: t.Any) -> str:
    if not isinstance(x, str):
        raise TypeError(f"Expected sequence ID to be of string type, got {type(x)}.")
    return x


def _get_cpu_count(c: int):
    mc = os.cpu_count()
    if c == -1:
        return mc
    elif 0 < c <= mc:
        return c
    else:
        raise ValueError(
            f"Invalid requested CPU count {c}. Must be between 1 and the maximum "
            f"number of available cores {mc}."
        )


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
            "alphafold": curry(self.PDB.fetch_structures)(
                dir_=paths.structure_files, fmt=str_fmt, **kwargs
            ),
        }


class ConstructorBase(t.Generic[_CT, _IT], metaclass=ABCMeta):
    def __init__(self, config: ConstructorConfig):
        self.config = config
        self._max_cpu = _get_cpu_count(config["max_proc"])
        self.paths = self._setup_paths()
        self.collection: _CT = self._setup_collection()
        self.interfaces: Interfaces = self._setup_interfaces()
        self.references: list[PyHMMer] = self._setup_references()
        self._ref_kws: abc.Sequence[abc.Mapping[str, t.Any]] = self._setup_ref_kws()

        self._batches = None
        self._step = 0

        self.config.validate()
        self._setup_logger()
        self._save_references()

        config_path = self.config["out_dir"] / "collection_config.json"
        self.config.save(config_path)
        logger.info("Saved config to {}", config_path)

    def _setup_logger(self):
        if self.config["verbose"]:
            logger.add(sys.stdout, level="INFO")
        logger.add(self.paths.output / "log.txt", backtrace=True, level="DEBUG")

    def _setup_paths(self):
        paths = CollectionPaths(
            output=Path(self.config["out_dir"]),
            references=Path(self.config["ref_dir"]),
            sequences=Path(self.config["seq_dir"]),
            structures=Path(self.config["str_dir"]),
            str_fmt=self.config["str_fmt"],
        )
        paths.mkdirs()
        return paths

    def _setup_ref_kws(
        self,
    ) -> abc.Sequence[abc.Mapping[str, t.Any]]:
        num_refs = len(self.references)
        ref_kw = self.config["references_annotate_kw"]

        if isinstance(ref_kw, abc.Sequence):
            if len(ref_kw) == 0:
                return [{} for _ in range(num_refs)]
            else:
                if not all(isinstance(kw, abc.Mapping) for kw in ref_kw):
                    raise TypeError(
                        "Expected keyword arguments provided for each reference to be "
                        "a mapping."
                    )
                if len(ref_kw) != num_refs:
                    raise ValueError(
                        "If references arguments is a sequence, it must have the same "
                        "length as the number of provided references."
                    )
                return ref_kw
        elif isinstance(ref_kw, abc.Mapping):
            return [ref_kw for _ in range(num_refs)]
        else:
            raise TypeError("Invalid type for a reference arguments. Expected a")

    def _setup_references(self) -> list[PyHMMer]:
        try:
            refs = list(self.config["references"])
        except Exception as e:
            raise TypeError("Failed to convert references to list") from e

        init = curry(_init_reference)(alphabet=self.config["references_alphabet"])

        return list(map(init, refs))

    def _setup_interfaces(self) -> Interfaces:
        sifts = (
            SIFTS(**self.config["SIFTS_kwargs"])
            if self.config["source"] == "SIFTS"
            else None
        )
        io_proc = (
            1 if isinstance(self.collection, SequenceCollection) else self._max_cpu
        )
        init_kw = self.config["Initializer_kwargs"]
        if len(init_kw) == 0:
            init_kw["verbose"] = self.config["verbose"]

        return Interfaces(
            AlphaFold(**self.config["AlphaFold_kwargs"]),
            PDB(**self.config["PDB_kwargs"]),
            sifts,
            UniProt(**self.config["UniProt_kwargs"]),
            lxc.ChainInitializer(**init_kw),
            lxc.ChainIO(num_proc=io_proc, verbose=self.config["verbose"]),
        )

    def _save_references(self):
        for ref in self.references:
            hmm_name = ref.hmm.name.decode("utf-8")
            path = Path(self.config["ref_dir"] / f"{hmm_name}.hmm")
            with path.open("wb") as f:
                ref.hmm.write(f, binary=False)

    def _callback_and_filter(self, chains: lxc.ChainList[_CT]) -> lxc.ChainList[_CT]:
        if self.config["child_callback"] is not None:
            try:
                chains.collapse_children().apply(
                    self.config["child_callback"],
                    desc="Applying callback to children",
                    verbose=self.config["verbose"],
                )
            except Exception as e:
                raise RuntimeError("Failed to apply callback to children") from e
        if self.config["child_filter"] is not None:
            for c in chains:
                try:
                    init_ = len(c.children)
                    c.children = c.children.filter(self.config["child_filter"])
                    logger.info(
                        f"Chain {c}: filtered to {len(c.children)} children "
                        f"out of initial {init_}."
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to filter children of {c}") from e
        if self.config["parent_callback"] is not None:
            try:
                chains = chains.apply(
                    self.config["parent_callback"],
                    desc="Applying callback to parents",
                    verbose=self.config["verbose"],
                )
            except Exception as e:
                raise RuntimeError("Failed to apply callback to parent chains") from e
        if self.config["parent_filter"] is not None:
            try:
                init_ = len(chains)
                chains = chains.filter(self.config["parent_filter"])
                logger.info(f"Filtered to {len(chains)} out of initial {init_}")
            except Exception as e:
                raise RuntimeError("Failed to filter parent chains") from e

        return chains

    def fetch_missing(self, ids: abc.Iterable[_IT]) -> t.Any:
        """
        Fetch sequences/structures that are currently missing in the configured
        directories.

        :param ids: An iterable over input IDs.

        :return: Anything that the configured fetcher returns or ``None`` if
            source is local or unrecognized.
        """

        def parse_id(inp: str | tuple[str, ...]) -> str:
            if not isinstance(inp, str):
                return inp[0]
            return inp

        source = self.config["source"].lower()

        if source in ("af", "af2"):
            source = "alphafold"

        try:
            fetchers = self.interfaces.get_fetchers(self.paths, self.config["str_fmt"])
            return fetchers[source](map(parse_id, ids))
        except KeyError:
            if source != "local":
                raise ConfigError(f"Unrecognized source {self.config['source']}.")

    def run_batch(self, ids: abc.Iterable[str]) -> lxc.ChainList[_CT]:
        ids = list(self.parse_ids(ids))
        logger.info(f"Received batch of {len(ids)}.")

        if self.config["fetch_missing"]:
            logger.debug("Attempting to fetch missing entries.")
            self.fetch_missing(ids)
        chains = self.init_inputs(ids)
        logger.info(f"Initialized {len(chains)} chains.")
        for i, (ref, kw) in enumerate(
            zip(self.references, self._ref_kws, strict=True), start=1
        ):
            logger.info(f"Applying reference {i}.")
            num_hits = sum(1 for _ in ref.annotate(chains, **kw))
            logger.info(f"Reference {i} has {num_hits} hits.")

        chains = self._callback_and_filter(chains)
        logger.debug("Done applying callbacks.")

        if self.config["write_batches"]:
            num_written = sum(
                1 for _ in self.interfaces.IO.write(chains, self.paths.chains)
            )
            logger.info(f"Wrote {num_written} chains to {self.paths.chains}.")

        try:
            self.collection.add(chains)
            logger.debug("Added chains to the collection.")
        except Exception as e:
            logger.error(e)
            raise RuntimeError("Failed to add chains to collection.") from e
        finally:
            return chains

    @abstractmethod
    def _setup_collection(self) -> _CT:
        pass

    @abstractmethod
    def parse_ids(self, ids):
        pass

    @abstractmethod
    def wrap_paths(self, ids):
        pass

    @abstractmethod
    def init_inputs(self, ids):
        pass


class SeqCollectionConstructor(ConstructorBase[SequenceCollection, _SeqIt]):
    def _setup_collection(self) -> SequenceCollection:
        return _setup_collection(self.config, "Seq", ("uniprot",))

    def parse_ids(self, ids: _T) -> _T:
        return ids

    def wrap_paths(self, ids: abc.Iterable[str]) -> abc.Iterator[Path]:
        yield from (self.paths.sequence_files / f"{x}.fasta" for x in ids)

    def init_inputs(self, ids: abc.Iterable[str]) -> lxc.ChainList[lxc.ChainSequence]:
        res = self.interfaces.Initializer.from_iterable(self.wrap_paths(ids))
        return lxc.ChainList(res)


class StrCollectionConstructor(ConstructorBase[StructureCollection, _StrIt]):
    def _setup_collection(self) -> SequenceCollection:
        return _setup_collection(self.config, "Str", ("pdb", "af", "af2"))

    def parse_ids(
        self, ids: abc.Iterable[str]
    ) -> abc.Iterator[str | tuple[str, tuple[str, ...]]]:
        def _parse_id(x: str) -> tuple[str, tuple[str, ...]]:
            if ":" in x:
                id_, chains = x.split(":", maxsplit=1)
                return id_, tuple(chains.split(","))
            match self.config["source"].lower():
                case "af" | "af2":
                    return x, ("A",)
                case "pdb":
                    pdb_chains = self.interfaces.SIFTS.map_id(x)
                    if pdb_chains is None:
                        raise MissingData(
                            f"Failed to find chains valid for PDB ID {x}. "
                            f"Please provide them manually."
                        )
                    return x, tuple(pdb_chains)
                case _:
                    raise MissingData(
                        "For local sources, please provide chains for each structure "
                        "using the {ID}:{Chains} format."
                    )

        yield from map(_parse_id, ids)

    def wrap_paths(
        self, ids: abc.Iterable[tuple[str, _T]]
    ) -> abc.Iterator[tuple[Path, _T]]:
        fmt = self.config["str_fmt"]
        yield from (
            (self.paths.structure_files / f"{id_}.{fmt}", xs) for id_, xs in ids
        )

    def init_inputs(
        self, ids: abc.Iterable[_StrIt]
    ) -> lxc.ChainList[lxc.ChainSequence]:
        res = self.interfaces.Initializer.from_iterable(self.wrap_paths(ids))
        return lxc.ChainList(chain.from_iterable(res))


if __name__ == "__main__":
    raise RuntimeError
