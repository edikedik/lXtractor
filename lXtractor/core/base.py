"""
Base classes, commong types and functions for the `core` module.
"""
from __future__ import annotations

import typing as t
from abc import ABCMeta, abstractmethod
from collections import abc, UserDict
from io import TextIOBase
from pathlib import Path
from typing import runtime_checkable

import msgpack

T = t.TypeVar("T")
_T = t.TypeVar("_T", contravariant=True)

_Fetcher = abc.Callable[[abc.Iterable[str]], T]
_Getter = abc.Callable[[T, abc.Sequence[str]], abc.Sequence[str]]
_MapT = dict[int, int | None]

ALL21 = Path(__file__).parent.parent / "resources" / "all21.msgpack"


class ResNameDict(UserDict):
    """
    A dictionary providing mapping between PDB residue names and their
    one-letter codes. The mapping was parsed from the CCD and can be obtained
    by calling :meth:`lXtractor.ext.ccd.CCD.make_res_name_map`.

    >>> d = ResNameDict()
    >>> assert d['ALA'] == 'A'

    """

    def __init__(self):
        from lXtractor.core.exceptions import MissingData

        if not ALL21.exists():
            raise MissingData(f"Missing parsed all-to-one mapping in {ALL21}")

        with ALL21.open("rb") as f:
            unpacker = msgpack.Unpacker(f)
            all21 = unpacker.unpack()
            super().__init__(all21)


class AbstractResource(metaclass=ABCMeta):
    """
    Abstract base class defining basic interface any resource must provide.
    """

    def __init__(self, resource_path: str | Path, resource_name: str | None):
        """
        :param resource_path: Path to parsed resource data.
        :param resource_name: Resource's name.
        """
        self.name = resource_name
        self.path = resource_path

    @abstractmethod
    def read(self):
        """
        Read the resource using the :attr:`resource_path`
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        """
        Parse the read resource, so it's ready for usage.
        """
        raise NotImplementedError

    @abstractmethod
    def dump(self, path: Path):
        """
        Save the resource under the given `path`.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch(self, url: str):
        """
        Download the resource.
        """
        raise NotImplementedError


@t.runtime_checkable
class Ord(t.Protocol[_T]):
    """
    Any objects defining comparison operators.
    """

    def __le__(self, other: _T) -> bool:
        ...

    def __ge__(self, other: _T) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...


@t.runtime_checkable
class SupportsLT(t.Protocol[_T]):
    def __le__(self, other: _T) -> bool:
        ...


@runtime_checkable
class SupportsWrite(t.Protocol):
    """
    Any object with the `write` method.
    """

    def write(self, data):
        """
        Write the supplied data.
        """


@runtime_checkable
class AddMethod(t.Protocol):
    """
    A callable to add sequences to the aligned ones,
    preserving the alignment length.
    """

    def __call__(
        self,
        msa: abc.Iterable[tuple[str, str]] | Path,
        seqs: abc.Iterable[tuple[str, str]],
        # **kwargs,
    ) -> abc.Iterable[tuple[str, str]]:
        ...


@runtime_checkable
class AlignMethod(t.Protocol):
    """
    A callable to align arbitrary sequences.
    """

    def __call__(
        self, seqs: abc.Iterable[tuple[str, str]] | Path
    ) -> abc.Iterable[tuple[str, str]]:
        ...


class SeqReader(t.Protocol):
    """
    A callable reading sequences into tuples of (header, _seq) pairs.
    """

    def __call__(
        self, inp: Path | TextIOBase | abc.Iterable[str]
    ) -> abc.Iterable[tuple[str, str]]:
        ...


class SeqWriter(t.Protocol):
    """
    A callable writing (header, _seq) pairs to disk.
    """

    def __call__(
        self,
        inp: abc.Iterable[tuple[str, str]],
        out: Path | SupportsWrite,
        # **kwargs
    ) -> None:
        ...


class SeqMapper(t.Protocol):
    """
    A callable accepting and returning a pair (header, _seq).
    """

    def __call__(self, seq: tuple[str, str], **kwargs) -> tuple[str, str]:
        ...


class SeqFilter(t.Protocol):
    """
    A callable accepting a pair (header, _seq) and returning a boolean.
    """

    def __call__(self, seq: tuple[str, str], **kwargs) -> bool:
        ...


class UrlGetter(t.Protocol):
    """
    A callable accepting some string arguments and turning them into a
    valid url.
    """

    def __call__(self, *args) -> str:
        ...


@t.runtime_checkable
class ApplyTWithArgs(t.Protocol[T]):
    def __call__(self, x: T, *args, **kwargs) -> T:
        ...


# ApplyT = abc.Callable[[T], T]
@t.runtime_checkable
class ApplyT(t.Protocol[T]):
    def __call__(self, x: T) -> T:
        ...


@t.runtime_checkable
class FilterT(t.Protocol[T]):
    def __call__(self, x: T) -> bool:
        ...


@t.runtime_checkable
class NamedTupleT(t.Protocol, abc.Iterable):
    def __getattr__(self, item):
        ...

    def _asdict(self) -> dict[str, t.Any]:
        ...


if __name__ == "__main__":
    raise RuntimeError
