from __future__ import annotations

import sys
import typing as t
from abc import ABCMeta, abstractmethod
from collections import abc
from itertools import chain
from pathlib import Path

from loguru import logger
from more_itertools import chunked_even, unique_everseen
from toolz import curry

from lXtractor import chain as lxc
from lXtractor.collection import (
    SequenceCollection,
    StructureCollection,
    MappingCollection,
)
from lXtractor.collection.support import (
    SeqItem,
    StrItem,
    MapItem,
    SeqItemList,
    StrItemList,
    MapItemList,
    BatchData,
    Interfaces,
    ConstructorConfig,
    BatchesHistory,
    CollectionPaths,
    _ITL,
    _IT,
)
from lXtractor.core import Alignment
from lXtractor.core.exceptions import ConfigError, FormatError, MissingData
from lXtractor.ext import PyHMMer, SIFTS, AlphaFold, PDB, UniProt, Pfam
from lXtractor.util import read_fasta
from lXtractor.util.misc import get_cpu_count

_RESOURCES = Path(__file__).parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCES / "collection_config.json"
_USER_CONFIG_PATH = _RESOURCES / "collection_user_config.json"
_CTA: t.TypeAlias = SequenceCollection | StructureCollection | MappingCollection
_ColT = t.TypeVar("_ColT", SequenceCollection, StructureCollection, MappingCollection)
_CT = t.TypeVar("_CT", lxc.ChainSequence, lxc.ChainStructure, lxc.Chain)
_U = t.TypeVar("_U", bound=int | float | str | None)
_T = t.TypeVar("_T")

# TODO: all configs should be stored within the database to be reusable
# TODO: init existing items directly from the database (preserve children?)


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


class ConstructorBase(t.Generic[_ColT, _CT, _IT, _ITL], metaclass=ABCMeta):
    def __init__(self, config: ConstructorConfig):
        self.config = config
        self._max_cpu = get_cpu_count(config["max_proc"])
        self.paths = self._setup_paths()
        self.collection: _ColT = self._setup_collection()
        self.interfaces: Interfaces = self._setup_interfaces()
        self.references: list[PyHMMer] = self._setup_references()
        self._ref_kws: abc.Sequence[abc.Mapping[str, t.Any]] = self._setup_ref_kws()

        self.history = BatchesHistory(self.item_list_type)
        self._batches: abc.Iterator[tuple[int, _ITL]] | None = None
        self.last_failed_batch: _ITL | None = None

        self.config.validate()
        self._setup_logger()
        self._save_references()

        config_path = self.config["out_dir"] / "collection_config.json"
        self.config.save(config_path)
        logger.info("Saved config to {}", config_path)

    @property
    @abstractmethod
    def item_list_type(self) -> t.Type[_ITL]:
        pass

    def _setup_logger(self):
        if self.config["verbose"]:
            logger.add(sys.stdout, level="INFO")
        logger.add(self.paths.output / "log.txt", backtrace=True, level="DEBUG")

    def _setup_paths(self):
        paths = CollectionPaths(
            output=Path(self.config["out_dir"]),
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
            raise TypeError(
                f"Invalid type {type(ref_kw)} for a reference arguments. "
                f"Expected a mapping or a sequence."
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
            path = self.paths.references / f"{hmm_name}.hmm"
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

    def _fetch(self, ids: abc.Iterable[str], source: str) -> t.Any:
        source = source.lower()
        ids = list(ids)

        if source == "local":
            logger.debug("Local source, nothing to fetch.")
            return

        fetchers = self.interfaces.get_fetchers(self.paths, self.config["str_fmt"])
        if source not in fetchers:
            raise ConfigError(f"Unrecognized source {source}.")

        logger.debug(f"Passing {len(ids)} inputs to {source} fetcher.")

        res = fetchers[source](ids)
        if isinstance(res, tuple) and len(res) == 2:
            fetched, failed = res
            logger.info(f"Fetched {len(fetched)} entries from {source}.")
            if len(failed) > 0:
                logger.warning(f"Failed on {len(failed)}: {failed}.")

        return res

    def parse_inputs(self, inputs: abc.Iterable[t.Any]) -> abc.Iterator[_IT]:
        yield from chain.from_iterable(map(self._parse_inp, inputs))

    def run_batch(self, items: _ITL) -> lxc.ChainList[_CT]:
        logger.info(f"Received batch of {len(items)} items.")

        if self.config["fetch_missing"]:
            logger.debug("Fetching missing entries.")
            self.fetch_missing(items)
        else:
            logger.debug("Fetching disabled.")

        chains = self.init_inputs(items)
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

    def _run(
        self,
        items: abc.Iterable[_IT] | None,
        stop_on_batch_failure: bool,
    ) -> abc.Iterator[BatchData[_ITL, _CT]]:
        if self._batches is None:
            if items is None:
                raise MissingData(
                    "No items provided and no batches initialized -> nothing to run."
                )
            self.init_batches(items, 1)

        for batch_i, batch in self._batches:
            try:
                chains = self.run_batch(batch)
                cs = chains if self.config["keep_chains"] else None
                item_type = self.item_list_type().item_type
                items_done = chain.from_iterable(map(item_type.from_chain, chains))
                bd = BatchData(batch_i, batch, self.item_list_type(items_done), cs)
                self.history.data.append(bd)
            except Exception as e:
                msg = f"Failed on batch {batch_i} of size {len(batch)}."
                self.last_failed_batch = batch
                if stop_on_batch_failure:
                    raise RuntimeError(msg) from e
                else:
                    logger.warning(msg)
                    logger.error(e)
                    bd = BatchData(
                        batch_i, batch, self.item_list_type(), None, failed=True
                    )
                    self.history.data.append(bd)
            yield bd

    def make_batches(
        self, items: abc.Iterable[_IT], start: int
    ) -> abc.Iterator[tuple[int, _ITL]]:
        items = unique_everseen(items)
        chunks = chunked_even(unique_everseen(items), self.config["batch_size"])
        chunks = map(self.item_list_type, chunks)
        yield from enumerate(chunks, start=start)

    def init_batches(self, items: abc.Iterable[_IT], start: int) -> None:
        self._batches = self.make_batches(items, start)

    def append_batches(self, batches: abc.Iterator[tuple[int, _ITL]]) -> None:
        if self._batches is None:
            self._batches = batches
        else:
            self._batches = chain(self._batches, batches)

    def prepend_batches(self, batches: abc.Iterator[tuple[int, _ITL]]):
        if self._batches is None:
            self._batches = batches
        else:
            self._batches = chain(batches, self._batches)

    def run(
        self, items: abc.Iterable[_IT], *, stop_on_batch_failure: bool = True
    ) -> abc.Iterator[BatchData[_ITL, _CT]]:
        self.history.cleanup()
        self._batches = None
        yield from self._run(items, stop_on_batch_failure)

    def resume(self) -> abc.Iterator[BatchData[_ITL, _CT]]:
        yield from self._run(None, True)

    def resume_with(
        self, items: abc.Iterable[_IT]
    ) -> abc.Iterator[BatchData[_ITL, _CT]]:
        last_step = self.history.last_step
        batches = self.make_batches(items, start=last_step + 1)
        self.prepend_batches(batches)
        yield from self._run(None, True)

    @abstractmethod
    def _parse_inp(self, x: t.Any) -> abc.Iterator[_IT]:
        pass

    @abstractmethod
    def _setup_collection(self) -> _CT:
        pass

    @abstractmethod
    def fetch_missing(self, items: abc.Iterable[_IT]) -> t.Any:
        pass

    @abstractmethod
    def init_inputs(self, items: _ITL) -> lxc.ChainList[_CT]:
        pass


class SeqCollectionConstructor(
    ConstructorBase[SequenceCollection, lxc.ChainSequence, SeqItem, SeqItemList]
):
    @property
    def item_list_type(self) -> t.Type[SeqItemList]:
        return SeqItemList

    def _parse_inp(self, x: t.Any) -> abc.Iterator[SeqItem]:
        if isinstance(x, SeqItem):
            yield x
        elif isinstance(x, str):
            yield from SeqItem.from_str(x)
        else:
            raise FormatError(f"Invalid value while parsing ID {x}.")

    def _setup_collection(self) -> SequenceCollection:
        return _setup_collection(self.config, "Seq", ("uniprot",))

    def fetch_missing(self, items: abc.Iterable[SeqItem]) -> t.Any:
        return self._fetch((x.seq_id for x in items), self.config["source"])

    def init_inputs(self, items: SeqItemList) -> lxc.ChainList[lxc.ChainSequence]:
        staged = items.prep_for_init(self.paths)
        chains = self.interfaces.Initializer.from_iterable(staged)
        return lxc.ChainList(filter(lambda x: x is not None, chains))


class StrCollectionConstructor(
    ConstructorBase[StructureCollection, lxc.ChainStructure, StrItem, StrItemList]
):
    @property
    def item_list_type(self) -> t.Type[StrItemList]:
        return StrItemList

    def _parse_inp(self, x: t.Any) -> abc.Iterator[StrItem]:
        match x:
            case StrItem():
                yield x
            case str():
                if ":" not in x:
                    if self.config["source"].lower() in ("af2", "af", "alphafold"):
                        x = f"{x}:A"
                    else:
                        if self.config["default_chain"] is None:
                            raise FormatError(
                                f"No chain specified for input {x} and no default "
                                f"chain was set up in config."
                            )
                        x = f"{x}:{self.config['default_chain']}"
                yield from StrItem.from_str(x)
            case (_, _):
                yield from StrItem.from_tuple(x)
            case _:
                raise FormatError(f"Invalid value while parsing ID {x}.")

    def _setup_collection(self) -> SequenceCollection:
        return _setup_collection(self.config, "Str", ("pdb", "af", "af2"))

    def fetch_missing(self, items: abc.Iterable[StrItem]) -> t.Any:
        source = self.config["source"].lower()
        if source in ("af2", "af"):
            source = "alphafold"
        return self._fetch((x.str_id for x in items), source)

    def init_inputs(self, items: StrItemList) -> lxc.ChainList[lxc.ChainStructure]:
        staged = items.prep_for_init(self.paths)
        chains = self.interfaces.Initializer.from_iterable(staged)
        chains = filter(lambda x: x is not None, chains)
        return lxc.ChainList(chain.from_iterable(chains))


class MapCollectionConstructor(
    ConstructorBase[MappingCollection, lxc.Chain, MapItem, MapItemList]
):
    @property
    def item_list_type(self) -> t.Type[MapItemList]:
        return MapItemList

    def _parse_inp(self, x: t.Any) -> abc.Iterator[MapItem]:
        match x:
            case MapItem():
                yield x
            case str():
                yield from MapItem.from_str(x)
            case (_, _):
                yield from MapItem.from_tuple(x)
            case _:
                raise FormatError(f"Invalid value while parsing ID {x}.")

    def _setup_collection(self) -> MappingCollection:
        return _setup_collection(self.config, "Map", ("sifts",))

    def fetch_missing(self, items: abc.Iterable[MapItem]) -> tuple[t.Any, t.Any]:
        items = list(items)
        res_seq = self._fetch((x.seq_item.seq_id for x in items), "uniprot")
        res_str = self._fetch((x.str_item.str_id for x in items), "pdb")
        return res_seq, res_str

    def init_inputs(self, items: MapItemList) -> lxc.ChainList[lxc.Chain]:
        return lxc.ChainList(
            self.interfaces.Initializer.from_mapping(
                dict(items.prep_for_init(self.paths))
            )
        )


if __name__ == "__main__":
    raise RuntimeError
