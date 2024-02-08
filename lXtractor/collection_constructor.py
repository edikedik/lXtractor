from __future__ import annotations

import logging
import json
import typing as t
from collections import abc
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from more_itertools import unzip
from toolz import curry

from lXtractor.chain import ChainSequence
from lXtractor.collection import (
    ChainCollection,
    StructureCollection,
    SequenceCollection,
)
from lXtractor.core import Alignment
from lXtractor.core.config import Config
from lXtractor.core.exceptions import MissingData, FormatError
from lXtractor.ext import PyHMMer, Pfam, AlphaFold, PDB, UniProt, SIFTS
from lXtractor.util import read_fasta

LOGGER = logging.getLogger(__name__)
_RESOURCES = Path(__file__).parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCES / "collection_config.json"
_USER_CONFIG_PATH = _RESOURCES / "collection_user_config.json"
_IDS: t.TypeAlias = abc.Iterable[str] | abc.Iterator[tuple[str, str]]
_CTA: t.TypeAlias = SequenceCollection | StructureCollection | ChainCollection
_CT = t.TypeVar("_CT", SequenceCollection, StructureCollection, ChainCollection)
_U = t.TypeVar("_U", bound=int | float | str | None)


def _collection_type_from_source(source: t.Any) -> tuple[t.Type[_CTA], ...]:
    match source:
        case (_, None, None) | "UniProt":
            return (SequenceCollection,)
        case (None, _, None) | "PDB" | "AF2":
            return SequenceCollection, StructureCollection
        case (_, _, _) | "SIFTS":
            return SequenceCollection, ChainCollection
        case _:
            raise FormatError(
                f"Failed to determine collection type from source {source}"
            )


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
    | ChainSequence
    | (str, Path)
    | (str, Alignment)
    | (str, ChainSequence)
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
        case ChainSequence():
            return PyHMMer.from_msa([(inp.name, inp.seq1)], inp.name, alphabet)
        case (str(), Alignment()):
            return PyHMMer.from_msa(inp[1], inp[0], alphabet)
        case (str(), str()) | (str(), ChainSequence()):
            name, seq = inp
            if isinstance(seq, ChainSequence):
                seq = seq.seq1
            return PyHMMer.from_msa([(name, seq)], name, alphabet)
        case str():
            return Pfam()[inp]
        case _:
            raise TypeError(f"Failed to parse input reference {inp}")


class ConstructorConfig(Config):
    def __init__(
        self,
        default_config_path: str | Path = _DEFAULT_CONFIG_PATH,
        user_config_path: str | Path = _USER_CONFIG_PATH,
        **kwargs,
    ):
        self.provided_settings = kwargs
        super().__init__(default_config_path, user_config_path)

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
        none_keys = ", ".join(self.list_missing_fields())
        if none_keys:
            raise MissingData(f"Missing values for required keys: {none_keys}")


@dataclass
class Interfaces:
    AlphaFold: AlphaFold
    PDB: PDB
    SIFTS: SIFTS | None
    UniProt: UniProt


@dataclass
class CollectionPaths:
    output: Path
    references: Path
    sequences: Path
    structures: Path


class CollectionConstructor:
    def __init__(self, config: ConstructorConfig):
        self.config = config
        self.interfaces: Interfaces = self._setup_interfaces()
        self.paths = self._setup_paths()
        self.collection = self._setup_collection()
        self.references: list[PyHMMer] = self._setup_references()

        self.config.validate()
        self._save_references()
        self.config.save(self.config["out_dir"] / "collection_config.json")

    def _setup_collection(self) -> _CT:
        possible_types = _collection_type_from_source(self.config["source"])
        provided_type = _to_concrete_collection(self.config["collection_type"])
        if provided_type not in possible_types:
            raise ValueError(
                f"Collection type {provided_type} is not possible to create from "
                f"source {self.config['source']}."
            )
        coll_name = provided_type.__name__.removesuffix("Collection").lower()

        if self.config["collection_name"] == "auto":
            self.config["collection_name"] = coll_name
        coll_path = self.config["out_dir"] / f"{self.config['collection_name']}.sqlite"
        return provided_type(coll_path)

    def _setup_paths(self):
        # (!) Order here and in CollectionPaths must be the same
        dir_keys = ("out_dir", "ref_dir", "seq_dir", "str_dir")
        for k in dir_keys:
            Path(self.config[k]).mkdir(parents=True, exist_ok=True)
        return CollectionPaths(
            output=Path(self.config["out_dir"]),
            references=Path(self.config["ref_dir"]),
            sequences=Path(self.config["seq_dir"]),
            structures=Path(self.config["str_dir"]),
        )

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
        return Interfaces(
            AlphaFold(**self.config["AlphaFold_kwargs"]),
            PDB(**self.config["PDB_kwargs"]),
            sifts,
            UniProt(**self.config["UniProt_kwargs"]),
        )

    def _save_references(self):
        for ref in self.references:
            hmm_name = ref.hmm.name.decode("utf-8")
            path = Path(self.config["ref_dir"] / f"{hmm_name}.hmm")
            with path.open("wb") as f:
                ref.hmm.write(f, binary=False)

    def fetch_missing(self, ids: _IDS, is_mapping: bool):
        def strip_idx(s: str, at: str = ":", idx: int = 0):
            return s.strip(at)[idx]

        match self.config["source"].lower():
            case "uniprot":
                self.interfaces.UniProt.fetch_sequences(
                    ids, self.paths.sequences, overwrite=False
                )
            case "sifts":
                if is_mapping:
                    uni_ids, vs = unzip(ids)
                    pdb_ids = map(strip_idx, chain.from_iterable(vs))

                else:
                    uni_ids = list(ids)
                    pdb_ids = filter(bool, map(self.interfaces.SIFTS.map_id, uni_ids))
                    pdb_ids = map(strip_idx, chain.from_iterable(pdb_ids))
                self.interfaces.UniProt.fetch_sequences(
                    uni_ids, self.paths.sequences, overwrite=False
                )
                self.interfaces.PDB.fetch_structures(
                    pdb_ids, self.paths.structures, self.config["str_fmt"]
                )
            case "pdb":
                self.interfaces.PDB.fetch_structures(
                    map(strip_idx, ids), self.paths.structures, self.config["str_fmt"]
                )
            case "alphafold":
                self.interfaces.AlphaFold.fetch_structures(
                    map(strip_idx, ids), self.paths.structures, self.config["str_fmt"]
                )
            case "local":
                LOGGER.info("Local source: nothing to fetch")
            case _:
                LOGGER.info("Unrecognized source name; nothing to fetch.")

    def step(self, ids: _IDS, is_mapping: bool):
        ids = list(ids)
        if self.config["fetch_missing"]:
            self.fetch_missing(ids, is_mapping)


if __name__ == "__main__":
    raise RuntimeError
