from __future__ import annotations

import logging
import typing as t
from collections import abc
from concurrent.futures import Future
from dataclasses import dataclass, asdict
from itertools import chain
from pathlib import Path

from toolz import curry, merge, valfilter, valmap

from lXtractor.chain import ChainList, Chain, ChainSequence, ChainStructure
from lXtractor.core.config import DefaultConfig
from lXtractor.util import get_dirs, apply, path_tree

CT = t.TypeVar("CT", ChainSequence, ChainStructure, Chain)
LOGGER = logging.getLogger(__name__)

_CB: t.TypeAlias = abc.Callable[[CT], CT]
_P = t.TypeVar("_P", dict, Path)
_ChildDict: t.TypeAlias = dict[Path, list[_P]]


__all__ = ("ChainIOConfig", "ChainIO", "read_chains")


@dataclass
class ChainIOConfig:
    num_proc: int = 1
    verbose: bool = False
    tolerate_failures: bool = False


@curry
def _read_obj(
    path: Path,
    obj_type: t.Type[CT],
    tolerate_failures: bool,
    callbacks: abc.Iterable[_CB],
    **kwargs,
) -> CT | None:
    try:
        obj = obj_type.read(path, **kwargs)
        for cb in callbacks:
            obj = cb(obj)
        return obj
    except Exception as e:
        LOGGER.warning(f"Failed to initialize {obj_type} from {path}")
        LOGGER.exception(e)
        if not tolerate_failures:
            raise e
    return None


@curry
def _write_obj(obj: CT, base: Path, tolerate_failures: bool, **kwargs) -> Path | None:
    path = base / obj.id
    try:
        return obj.write(path, **kwargs)
    except Exception as e:
        LOGGER.warning(f"Failed to write {obj} to {path}")
        LOGGER.exception(e)
        if not tolerate_failures:
            raise e
    return None


def _read_objs(
    obj_type: t.Type[CT],
    paths: list[Path],
    cfg: ChainIOConfig,
    callbacks: abc.Sequence[_CB],
    kwargs: dict[str, t.Any] | None,
) -> dict[Path, CT]:
    io = ChainIO(**asdict(cfg))
    if kwargs is None:
        kwargs = {}

    return dict(zip(paths, io.read(obj_type, paths, callbacks, **kwargs)))


def read_chains(
    paths: Path | abc.Sequence[Path],
    children: bool,
    *,
    seq_cfg: ChainIOConfig = ChainIOConfig(),
    str_cfg: ChainIOConfig = ChainIOConfig(),
    seq_callbacks: abc.Sequence[_CB] = (),
    str_callbacks: abc.Sequence[_CB] = (),
    seq_kwargs: dict[str, t.Any] | None = None,
    str_kwargs: dict[str, t.Any] | None = None,
) -> ChainList[Chain]:
    """
    Reads saved :class:`lXtractor.core.chain.chain.Chain` objects without
    invoking :meth:`lXtractor.core.chain.chain.Chain.read`. Instead, it will
    use separate :class:`ChainIO` instances to read chain sequences and
    chain structures. The output is identical to :meth:`ChainIO.read_chain_seq`.

    Consider using it for:

        #. For parallel parsing of ``Chain`` objects with many structures.
        #. For separate treatment of chain sequences and chain structures.
        #. For better customization of chain sequences and structures parsing.

    :param paths: A path or a sequence of paths to chains.
    :param children: Search for, parse and integrate all nested children.
    :param seq_cfg: :class:`ChainIO` config for chain sequences parsing.
    :param str_cfg: ... for chain structures parsing.
    :param seq_callbacks: A (potentially empty) sequence passed to the reader.
        Each callback must accept and return a single chain sequence.
    :param str_callbacks: ... Same for the structures.
    :param seq_kwargs: Passed to
        :meth:`lXtractor.core.chain.sequence.ChainSequence.read`.
    :param str_kwargs: Passed to
        :meth:`lXtractor.core.chain.structure.ChainStructure.read`.
    :return: A chain list of parsed chains.
    """
    if isinstance(paths, Path):
        paths = [paths]

    if children:
        trees = list(map(path_tree, paths))

        seq_paths = list(chain.from_iterable(tree.nodes for tree in trees))
        node2data = merge(*(dict(tree.nodes.data()) for tree in trees))
        node2data = valfilter(lambda x: "structures" in x, node2data)
        seq2str = valmap(lambda x: x["structures"], node2data)
    else:
        trees = None
        seq_paths = paths
        seq2str = {p: list(p.glob("structures/*")) for p in seq_paths}
        seq2str = valfilter(bool, seq2str)

    str_paths = list(chain.from_iterable(seq2str.values()))

    path2seq = _read_objs(ChainSequence, seq_paths, seq_cfg, seq_callbacks, seq_kwargs)
    path2str = _read_objs(ChainStructure, str_paths, str_cfg, str_callbacks, str_kwargs)
    path2chain = valmap(Chain, path2seq)

    for seq_path, str_paths in seq2str.items():
        parent_chain = path2chain[seq_path]
        for str_path in str_paths:
            bound_str = path2str.get(str_path, None)
            if bound_str is not None:
                parent_chain.structures.append(bound_str)

    if trees is not None:
        for tree in trees:
            for parent, child in tree.edges:
                parent_chain = path2chain[parent]
                child_chain = path2chain[child]

                if not (parent_chain is None or child_chain is None):
                    parent_chain.children.append(child_chain)
                    child_chain.parent = parent_chain

    return ChainList(c for p, c in path2chain.items() if p in paths)


class ChainIO:
    """
    A class handling reading/writing collections of `Chain*` objects.
    """

    # TODO: implement context manager
    def __init__(
        self,
        num_proc: int = 1,
        verbose: bool = False,
        tolerate_failures: bool = False,
    ):
        """
        :param num_proc: The number of parallel processes. Using more processes
            is especially beneficial for :class:`ChainStructure`'s and
            :class:`Chain`'s with structures. Otherwise, the increasing this
            number may not reduce or actually worsen the time needed to
            read/write objects.
        :param verbose: Output logging and progress bar.
        :param tolerate_failures: Errors when reading/writing do not raise
            an exception.
        """
        #: The number of parallel processes
        self.num_proc = num_proc
        #: Output logging and progress bar.
        self.verbose = verbose
        #: Errors when reading/writing do not raise an exception.
        self.tolerate_failures = tolerate_failures

    def read(
        self,
        obj_type: t.Type[CT],
        path: Path | abc.Iterable[Path],
        callbacks: abc.Sequence[_CB] = (),
        **kwargs,
    ) -> abc.Generator[CT | None, None, None]:
        """
        Read ``obj_type``-type objects from a path or an iterable of paths.

        :param obj_type: Some class with ``@classmethod(read(path))``.
        :param path: Path to the dump to read from. It's a path to directory
            holding files necessary to init a given `obj_type`, or an iterable
            over such paths.
        :param callbacks: Callables applied sequentially to parsed object.
        :param kwargs: Passed to the object's :meth:`read` method.
        :return: A generator over initialized objects or futures.
        """

        if isinstance(path, Path):
            dirs = get_dirs(path)
        else:
            dirs = {p.name: p for p in path if p.is_dir()}

        _read = _read_obj(
            obj_type=obj_type,
            tolerate_failures=self.tolerate_failures,
            callbacks=callbacks,
            **kwargs,
        )
        fnames = DefaultConfig["filenames"]
        if fnames["segments_dir"] in dirs or not dirs and isinstance(path, Path):
            paths = [path]
        else:
            paths = iter(dirs.values())

        yield from apply(
            _read, paths, self.verbose, f"Reading {obj_type.__name__}", self.num_proc
        )

    def read_chain(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> abc.Generator[Chain | None, None, None]:
        """
        Read :class:`Chain`'s from the provided path.

        If `path` contains signature files and directories
        (such as `sequence.tsv` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`Chain` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`read`.
        :return: An iterator over :class:`Chain` objects.
        """
        yield from self.read(Chain, path, **kwargs)

    def read_chain_seq(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> abc.Generator[ChainSequence | None, None, None]:
        """
        Read :class:`ChainSequence`'s from the provided path.

        If `path` contains signature files and directories
        (such as `sequence.tsv` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`ChainSequence` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`read`.
        :return: An iterator over :class:`ChainSequence` objects.
        """
        yield from self.read(ChainSequence, path, **kwargs)

    def read_chain_str(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> abc.Generator[ChainStructure | None, None, None]:
        """
        Read :class:`ChainStructure`'s from the provided path.

        If `path` contains signature files and directories
        (such as `structure.cif` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`ChainStructure` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`read`.
        :return: An iterator over :class:`ChainStructure` objects.
        """
        yield from self.read(ChainStructure, path, **kwargs)

    def write(
        self,
        chains: CT | abc.Iterable[CT],
        base: Path,
        **kwargs,
    ) -> abc.Generator[Path | None | Future, None, None]:
        """
        :param chains: A single or multiple chains to write.
        :param base: A writable dir. For multiple chains, will use
            `base/chain.id` directory.
        :param kwargs: Passed to a chain's `write` method.
        :return: Whatever `write` method returns.
        """
        if isinstance(chains, (ChainSequence, ChainStructure, Chain)):
            yield chains.write(base)
        else:
            fn = _write_obj(
                base=base, tolerate_failures=self.tolerate_failures, **kwargs
            )
            yield from apply(
                fn,
                chains,
                verbose=self.verbose,
                desc="Writing objects",
                num_proc=self.num_proc,
            )


if __name__ == "__main__":
    raise RuntimeError
