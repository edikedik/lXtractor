from __future__ import annotations

import logging
import typing as t
from collections import abc
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from pathlib import Path

from toolz import curry
from tqdm.auto import tqdm

from lXtractor.core.chain.chain import Chain
from lXtractor.core.chain.sequence import ChainSequence
from lXtractor.core.chain.structure import ChainStructure
from lXtractor.core.config import DumpNames, _DumpNames
from lXtractor.util.io import get_dirs

CT = t.TypeVar('CT', ChainSequence, ChainStructure, Chain)
LOGGER = logging.getLogger(__name__)

_CB: t.TypeAlias = abc.Callable[[CT], CT]


@curry
def _read_obj(
    path: Path,
    obj_type: t.Type[CT],
    tolerate_failures: bool,
    callbacks: abc.Iterable[_CB] | None,
    **kwargs,
) -> CT | None:
    try:
        obj = obj_type.read(path, **kwargs)
        if callbacks is not None:
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
def _write_obj(obj: CT, path: Path, tolerate_failures: bool, **kwargs) -> Path | None:
    try:
        obj.write(path, **kwargs)
        return path
    except Exception as e:
        LOGGER.warning(f"Failed to write {obj} to {path}")
        LOGGER.exception(e)
        if not tolerate_failures:
            raise e
    return None


class ChainIO:
    """
    A class handling reading/writing collections of `Chain*` objects.
    """

    # TODO: implement context manager
    def __init__(
        self,
        num_proc: None | int = None,
        verbose: bool = False,
        tolerate_failures: bool = False,
        dump_names: _DumpNames = DumpNames,
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
        :param dump_names: File names container.
        """
        #: The number of parallel processes
        self.num_proc = num_proc
        #: Output logging and progress bar.
        self.verbose = verbose
        #: Errors when reading/writing do not raise an exception.
        self.tolerate_failures = tolerate_failures
        #: File names container.
        self.dump_names = dump_names

    def read(
        self,
        obj_type: t.Type[CT],
        path: Path | abc.Iterable[Path],
        non_blocking: bool = False,
        callbacks: abc.Sequence[_CB] | None = None,
        **kwargs,
    ) -> abc.Generator[CT | Future | None, None, None]:
        """
        Read object dump.

        :param obj_type: Some class with ``@classmethod(read(path))``.
        :param path: Path to the dump to read from. It's a path to directory
            holding files necessary to init a given `obj_type`.
        :param non_blocking: Return `Future` objects without attempting to
            retrieve the result.
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

        if DumpNames.segments_dir in dirs or not dirs and isinstance(path, Path):
            yield _read(path)
            return

        paths = iter(dirs.values())

        if self.num_proc is None:

            if self.verbose:
                yield from map(_read, tqdm(paths, desc=f"Reading {obj_type}"))
            else:
                yield from map(_read, paths)

        else:

            with ProcessPoolExecutor(self.num_proc) as executor:

                futures: abc.Iterator[Future] = as_completed(
                    [executor.submit(_read, d) for d in paths]
                )

                if non_blocking:
                    yield from futures

                _futures = (
                    tqdm(futures, desc=f"Reading {obj_type}")
                    if self.verbose
                    else futures
                )

                for f in _futures:
                    yield f.result()

    def write(
        self,
        objs: CT | abc.Iterable[CT],
        base: Path,
        non_blocking: bool = False,
        **kwargs,
    ) -> abc.Generator[Path | None | Future, None, None]:
        """
        :param objs: A single or multiple objects to write.
            Each must have a `write` method accepting a directory.
        :param base: A writable dir. If `objs` are many, dump into `id`
            directories.
        :param non_blocking: If :attr:`num_proc` is >= 1, return `Future`
            objects instead of waiting for the result.
        :param kwargs: Passed to the `write` method of each object.
        :return: Whatever `write` method returns.
        """
        if isinstance(objs, (ChainSequence, ChainStructure, Chain)):
            objs.write(base)
        else:
            _write = _write_obj(tolerate_failures=self.tolerate_failures, **kwargs)

            if self.num_proc is None:
                if self.verbose:
                    objs = tqdm(objs, desc="Writing objects")
                for obj in objs:
                    yield _write(obj, base / obj.id)
            else:
                with ProcessPoolExecutor(self.num_proc) as executor:

                    futures = as_completed(
                        [executor.submit(_write, obj, base / obj.id) for obj in objs]
                    )

                    if non_blocking:
                        yield from futures

                    _futures = (
                        tqdm(futures, desc="Writing objects")
                        if self.verbose
                        else futures
                    )

                    for f in _futures:
                        yield f.result()

    def read_chain(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> abc.Generator[Chain | Future | None, None, None]:
        """
        Read :class:`Chain`'s from the provided path.

        If `path` contains signature files and directories
        (such as `sequence.tsv` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`Chain` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`read`.
        :return: An single chain or iterator over chain objects read
            sequentially or in parallel.
        """
        return self.read(Chain, path, **kwargs)

    def read_chain_seq(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> abc.Generator[ChainSequence | Future | None, None, None]:
        """
        Read :class:`ChainSequence`'s from the provided path.

        If `path` contains signature files and directories
        (such as `sequence.tsv` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`ChainSequence` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`read`.
        :return: A single chain sequence or iterator over
            :class:`ChainSequence` objects read sequentially or in parallel.
        """
        return self.read(ChainSequence, path, **kwargs)

    def read_chain_struc(
        self, path: Path | abc.Iterable[Path], **kwargs
    ) -> abc.Generator[ChainStructure | Future | None, None, None]:
        """
        Read :class:`ChainStructure`'s from the provided path.

        If `path` contains signature files and directories
        (such as `structure.cif` and `segments`), it is assumed to contain
        a single object. Otherwise, it is assumed to contain multiple
        :class:`ChainStructure` objects.

        :param path: Path to a dump or a dir of dumps.
        :param kwargs: Passed to :meth:`read`.
        :return: A single chain sequence or iterator over
            :class:`ChainStructure` objects read sequentially or in parallel.
        """
        return self.read(ChainStructure, path, **kwargs)


if __name__ == '__main__':
    raise RuntimeError
