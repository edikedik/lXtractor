from __future__ import annotations

import json
import typing as t
from collections import abc
from pathlib import Path

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
from lXtractor.ext import PyHMMer, Pfam
from lXtractor.util import read_fasta

_RESOURCES = Path(__file__).parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCES / "collection_config.json"
_USER_CONFIG_PATH = _RESOURCES / "collection_user_config.json"
_CT = t.TypeVar("_CT", SequenceCollection, StructureCollection, ChainCollection)
_CTA: t.TypeAlias = SequenceCollection | StructureCollection | ChainCollection


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


class CollectionConstructor:
    def __init__(self, config: ConstructorConfig):
        self.config = config
        self._setup_paths()
        self.collection = self._setup_collection()
        self.config.validate()
        self.references: list[PyHMMer] = self._setup_references()
        self._save_references()

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
        for k in ("out_dir", "str_dir", "seq_dir", "ref_dir"):
            self.config[k] = Path(self.config[k])
            self.config[k].mkdir(parents=True, exist_ok=True)

    def _setup_references(self) -> list[PyHMMer]:
        try:
            refs = list(self.config["references"])
        except Exception as e:
            raise TypeError("Failed to convert references to list") from e

        init = curry(_init_reference)(alphabet=self.config["references_alphabet"])

        return list(map(init, refs))

    def _save_references(self):
        for ref in self.references:
            hmm_name = ref.hmm.name.decode("utf-8")
            path = Path(self.config["ref_dir"] / f"{hmm_name}.hmm")
            with path.open("wb") as f:
                ref.hmm.write(f, binary=False)


if __name__ == "__main__":
    raise RuntimeError
