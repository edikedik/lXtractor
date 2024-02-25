from __future__ import annotations

import sys
import typing as t
from abc import ABCMeta, abstractmethod
from collections import abc
from itertools import chain
from pathlib import Path

from loguru import logger
from more_itertools import chunked_even, unique_everseen, unzip
from toolz import curry

from lXtractor import chain as lxc
from lXtractor.collection import (
    SequenceCollection,
    StructureCollection,
    MappingCollection,
)
from lXtractor.collection.support import (
    BatchData,
    Interfaces,
    ConstructorConfig,
    BatchesHistory,
    CollectionPaths,
)
from lXtractor.core import Alignment
from lXtractor.core.exceptions import ConfigError, MissingData, FormatError
from lXtractor.ext import PyHMMer, SIFTS, AlphaFold, PDB, UniProt, Pfam
from lXtractor.util import read_fasta
from lXtractor.util.misc import get_cpu_count

_RESOURCES = Path(__file__).parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCES / "collection_config.json"
_USER_CONFIG_PATH = _RESOURCES / "collection_user_config.json"
_CTA: t.TypeAlias = SequenceCollection | StructureCollection | MappingCollection
_ColT = t.TypeVar("_ColT", SequenceCollection, StructureCollection, MappingCollection)
_CT = t.TypeVar("_CT", lxc.ChainSequence, lxc.ChainStructure, lxc.Chain)
_SeqIt: t.TypeAlias = str
_StrIt: t.TypeAlias = tuple[str, tuple[str, ...]]
_MapIt: t.TypeAlias = tuple[_SeqIt, abc.Sequence[_StrIt]]
_IT = t.TypeVar("_IT", _SeqIt, _StrIt, _MapIt)
_U = t.TypeVar("_U", bound=int | float | str | None)
_T = t.TypeVar("_T")

# TODO: all configs should be stored within the database to be reusable


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


def _to_concrete_collection(desc: str) -> t.Type[_ColT]:
    match desc[:3].lower():
        case "map":
            return MappingCollection
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


def _filter_existing_seq(paths: abc.Iterable[Path]) -> abc.Iterator[Path]:
    for p in paths:
        if p.exists():
            yield p
        else:
            logger.warning(f"Path to the sequence {p} does not exist.")


def _filter_existing_str(
    paths: abc.Iterable[tuple[Path, _T]]
) -> abc.Iterator[tuple[Path, _T]]:
    for p, xs in paths:
        if p.exists():
            yield p, xs
        else:
            logger.warning(f"Path to the structure {p} does not exist.")


def _filter_existing_chains(
    paths: abc.Iterable[tuple[Path, abc.Sequence[tuple[Path, _T]]]]
) -> abc.Iterator[tuple[Path, abc.Sequence[tuple[Path, _T]]]]:
    for seq_path, str_paths in paths:
        if seq_path.exists():
            yield seq_path, list(_filter_existing_str(str_paths))
        else:
            logger.warning(f"Path to the sequence {seq_path} does not exist.")


class ConstructorBase(t.Generic[_ColT, _CT, _IT], metaclass=ABCMeta):
    def __init__(self, config: ConstructorConfig):
        self.config = config
        self._max_cpu = get_cpu_count(config["max_proc"])
        self.paths = self._setup_paths()
        self.collection: _ColT = self._setup_collection()
        self.interfaces: Interfaces = self._setup_interfaces()
        self.references: list[PyHMMer] = self._setup_references()
        self._ref_kws: abc.Sequence[abc.Mapping[str, t.Any]] = self._setup_ref_kws()

        self.history = BatchesHistory()
        self._batches: abc.Iterator[list[str]] = iter([])
        self.last_failed_batch: None | list[str] = None

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
                raise RuntimeError("Failed to apply callback to children.") from e
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
                    raise RuntimeError(f"Failed to filter children of {c}.") from e
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

    def fetch_missing(self, ids: abc.Iterable[_SeqIt] | abc.Iterable[_StrIt]) -> t.Any:
        """
        Fetch sequences/structures that are currently missing in the configured
        directories.

        :param ids: An iterable over input IDs.

        :return: Anything that the configured fetcher returns or ``None`` if
            source is local or unrecognized.
        """

        source = self.config["source"].lower()

        if source in ("af", "af2"):
            source = "alphafold"

        logger.debug(f"Fetching source {source}")

        try:
            fetchers = self.interfaces.get_fetchers(self.paths, self.config["str_fmt"])
            ids = [inp if isinstance(inp, str) else inp[0] for inp in ids]
            logger.debug(f"Passing {len(ids)} inputs to {source} fetcher.")
            res = fetchers[source](ids)
            if isinstance(res, tuple) and len(res) == 2:
                fetched, failed = res
                logger.info(f"Fetched {len(fetched)} entries from {source}.")
                if len(failed) > 0:
                    logger.warning(f"Failed on {len(failed)}: {failed}.")
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
        if len(chains) == 0:
            logger.warning("No entries initialized.")
            return chains

        for i, (ref, kw) in enumerate(zip(self.references, self._ref_kws, strict=True)):
            logger.info(f"Applying reference {i}.")
            num_hits = sum(1 for _ in ref.annotate(chains, **kw))
            logger.info(f"Reference {i} has {num_hits} hits.")

        chains = self._callback_and_filter(chains)
        logger.debug("Done applying callbacks.")

        if self.config["write_batches"]:
            paths = list(self.interfaces.IO.write(chains, self.paths.chains))
            self.collection.link(paths)
            logger.info(f"Wrote {len(paths)} chains to {self.paths.chains}.")

        try:
            self.collection.add(chains)
            logger.debug("Added chains to the collection.")
        except Exception as e:
            logger.error(e)
            raise RuntimeError("Failed to add chains to collection.") from e

        return chains

    def _init_batches(self, ids: abc.Iterable[str], start: int):
        self._batches = enumerate(
            chunked_even(unique_everseen(ids), self.config["batch_size"]), start=start
        )

    def _prepend_batch(self, batch_i: int, batch: list[str]):
        self._batches = chain([(batch_i, batch)], self._batches)

    def _run(
        self,
        ids: abc.Iterable[str],
        stop_on_batch_failure: bool,
    ) -> abc.Iterator[list[str]]:
        if self._batches is None:
            self._init_batches(ids, 1)

        for batch_i, batch in self._batches:
            try:
                chains = self.run_batch(batch)
                cs = chains if self.config["keep_chains"] else None
                self.history.data.append(BatchData(batch_i, batch, chains.ids, cs))
            except Exception as e:
                msg = f"Failed on batch {batch_i} of size {len(batch)}."
                self.last_failed_batch = batch
                if stop_on_batch_failure:
                    raise RuntimeError(msg) from e
                else:
                    logger.warning(msg)
                    logger.error(e)
                    self.history.data.append(
                        BatchData(batch_i, batch, [], None, failed=True)
                    )
            yield batch

    def run(
        self, ids: abc.Iterable[str], *, stop_on_batch_failure: bool = True
    ) -> abc.Iterator[list[str]]:
        self.history.cleanup()
        self._batches = None
        yield from self._run(ids, stop_on_batch_failure)

    def resume(self, ids: abc.Iterable[str]) -> abc.Iterator[list[str]]:
        last_step = self.history.last_step()
        self._init_batches(ids, last_step + 1)
        yield from self._run(ids, True)

    def resume_with(self, ids: abc.Iterable[str]) -> abc.Iterator[list[str]]:
        if self._batches is None:
            raise ValueError(
                "No existing batches to resume from. "
                "Please call `run` with the provided ids."
            )
        last_step = self.history.last_step()
        self._batches = chain([(last_step + 1, list(ids))], self._batches)
        yield from self._run([], True)

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


class SeqCollectionConstructor(
    ConstructorBase[SequenceCollection, lxc.ChainSequence, _SeqIt]
):
    def _setup_collection(self) -> SequenceCollection:
        return _setup_collection(self.config, "Seq", ("uniprot",))

    def parse_ids(self, ids: _T) -> _T:
        return ids

    def wrap_paths(self, ids: abc.Iterable[str]) -> abc.Iterator[Path]:
        yield from (self.paths.sequence_files / f"{x}.fasta" for x in ids)

    def init_inputs(self, ids: abc.Iterable[str]) -> lxc.ChainList[lxc.ChainSequence]:
        paths = _filter_existing_seq(self.wrap_paths(ids))
        res = self.interfaces.Initializer.from_iterable(paths)
        return lxc.ChainList(res)


class StrCollectionConstructor(
    ConstructorBase[StructureCollection, lxc.ChainStructure, _StrIt]
):
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
        paths = _filter_existing_str(self.wrap_paths(ids))
        res = self.interfaces.Initializer.from_iterable(paths)
        return lxc.ChainList(chain.from_iterable(res))


class MapCollectionConstructor(ConstructorBase[MappingCollection, lxc.Chain, _StrIt]):
    def _setup_collection(self) -> MappingCollection:
        return _setup_collection(self.config, "Map", ("sifts",))

    def fetch_missing(self, ids: abc.Iterable[_MapIt]) -> t.Any:
        seq_ids, str_ids = unzip(ids)

        with self.config.temporary_namespace():
            self.config['source'] = 'uniprot'
            super().fetch_missing(seq_ids)
            self.config['source'] = 'pdb'
            super().fetch_missing(chain.from_iterable(str_ids))

    def parse_ids(self, ids: abc.Iterable[str]) -> abc.Iterator[_MapIt]:
        def _parse_id(id_: str) -> _MapIt:
            try:
                seq_id, str_ids = id_.split("=>", maxsplit=1)
                str_ids = (x.split(":", maxsplit=1) for x in str_ids)
                str_ids = [
                    (str_id, tuple(chains.split(","))) for str_id, chains in str_ids
                ]
                return seq_id, str_ids
            except Exception as e:
                raise FormatError(f"Invalid mapping format for id {id_}") from e

        yield from map(_parse_id, ids)

    def wrap_paths(
        self, ids: abc.Iterable[tuple[str, abc.Sequence[tuple[str, _T]]]]
    ) -> abc.Iterator[tuple[Path, abc.Sequence[tuple[Path, _T]]]]:
        fmt = self.config["str_fmt"]
        seq_dir = self.paths.sequence_files
        str_dir = self.paths.structure_files
        for seq_id, str_ids in ids:
            str_ids = [(str_dir / f"{x}.{fmt}", xs) for x, xs in str_ids]
            yield seq_dir / f"{seq_id}.fasta", str_ids

    def init_inputs(self, ids: abc.Iterable[_MapIt]) -> lxc.ChainList[lxc.Chain]:
        paths = _filter_existing_chains(self.wrap_paths(ids))
        res = self.interfaces.Initializer.from_mapping(dict(paths))
        return lxc.ChainList(res)


if __name__ == "__main__":
    raise RuntimeError
