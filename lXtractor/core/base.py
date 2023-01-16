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

from lXtractor.core.config import DumpNames, _DumpNames

T = t.TypeVar('T')
_T = t.TypeVar('_T', contravariant=True)
KT = t.TypeVar('KT', bound=abc.Hashable)
VT = t.TypeVar('VT')

_Fetcher = t.Callable[[t.Iterable[str]], T]
_Getter = t.Callable[[T, t.Sequence[str]], t.Sequence[str]]

_MapT = t.Dict[int, t.Optional[int]]

_AminoAcids = [
    ('ALA', 'A'),
    ('CYS', 'C'),
    ('THR', 'T'),
    ('GLU', 'E'),
    ('ASP', 'D'),
    ('PHE', 'F'),
    ('TRP', 'W'),
    ('ILE', 'I'),
    ('VAL', 'V'),
    ('LEU', 'L'),
    ('LYS', 'K'),
    ('MET', 'M'),
    ('ASN', 'N'),
    ('GLN', 'Q'),
    ('SER', 'S'),
    ('ARG', 'R'),
    ('TYR', 'Y'),
    ('HIS', 'H'),
    ('PRO', 'P'),
    ('GLY', 'G'),
]
SOLVENTS = (
    'HOH',
    'MPD',
    'EDO',
    'DMS',
    'PEG',
    'PG4',
    'MES',
    '7PE',
    'DTT',
    'EPE',
    '1PE',
    'PHU',
    'MRD',
    'TLA',
    'ACT',
    'BU3',
    'MYR',
    'MLA',
    'DTD',
    'P6G',
    'SRT',
    'TBR',
    'BEN',
    'BME',
    'TRS',
    'PG0',
    'GBL',
    'CXS',
    'MXE',
    'BTB',
    'COM',
    'FLC',
    'EOH',
    'OCT',
    'MLI',
    'CIT',
    'MSE',
    'PGE',
    'GOL',
    'ACT',
    'BOG',
    'MOH',
    'TMA',
    'TFA',
    'MG8',
    '2PE',
    'TAM',
    'HC4',
    'P4G',
    'GLC',
    'DIO',
    'SIN',
    'BUD',
    '2HT',
    'TFA',
    'IPH',
    'SIN',
    'TAR',
    'PGF',
    'P4C',
    'HSJ',
    'DTV',
    'DVT',
    'SGM',
    'TCE',
    'GG5',
    'PTL',
)


# class SoftMapper(UserDict, t.Generic[KT, VT, T]):
#     """
#     A dict with ``[]`` syntax behaving as :meth:`dict.get`.
#     """
#
#     def __init__(self, *args, unk: T, **kwargs):
#         """
#
#         :param args: Passed to :class:`dict`.
#         :param unk: A value returned when `item` is not in dict.
#         :param kwargs: Passed to :class:`dict`.
#         """
#         self.unk = unk
#         super().__init__(*args, **kwargs)
#
#     def __getitem__(self, item: KT) -> VT | T:
#         # !!! Below fails with the recursion error !!!
#         # return super().get(item, self.unk)
#         try:
#             return self.data[item]
#             # return super().__getitem__(item)
#         except KeyError:
#             return self.unk


class AminoAcidDict(UserDict):
    """
    Provides mapping between 3->1 and 1->3-letter amino acid residue codes.

    >>> d = AminoAcidDict(any_unk='X')
    >>> assert d['A'] == 'ALA'
    >>> assert d['ALA'] == 'A'
    >>> assert d['XXX'] == 'X'
    >>> assert d['X'] == 'UNK'
    >>> assert d['CA'] == 'X'

    """

    def __init__(
        self, aa1_unk: str = 'X', aa3_unk: str = 'UNK', any_unk: t.Optional[str] = None
    ):
        """
        :param aa1_unk: Unknown character when mapping 3->1
        :param aa3_unk: Unknown character when mapping 1->3
        :param any_unk: Unknown character when a key doesn't
            meet 1 or 3 length requirements.
        """

        self.any_unk = any_unk
        self.aa1_unk = aa1_unk
        self.aa3_unk = aa3_unk

        self.three21 = dict(_AminoAcids)
        self.one23 = dict((x[1], x[0]) for x in _AminoAcids)

        super().__init__(**self.three21, **self.one23)

    def __getitem__(self, item: str) -> str:
        if len(item) == 3:
            return self.three21.get(item, self.aa1_unk)
        if len(item) == 1:
            return self.one23.get(item, self.aa3_unk)
        if self.any_unk is not None:
            return self.any_unk
        raise KeyError(
            f'Expected 3-sized or 1-sized item, ' f'got {len(item)}-sized {item}'
        )


class AbstractResource(metaclass=ABCMeta):
    """
    Abstract base class defining basic interface any resource must provide.
    """

    def __init__(self, resource_path: t.Optional[Path], resource_name: t.Optional[str]):
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


# class AbstractStructure(metaclass=ABCMeta):
#     """
#     Generic structure abstract interface.
#     """
#
#     __slots__ = ()
#
#     @classmethod
#     @abstractmethod
#     def read(cls, path: Path):
#         """
#         Read an object.
#         """
#
#     @abstractmethod
#     def write(self, path: Path):
#         """
#         Write an object to disk.
#         """
#
#     @abstractmethod
#     def get_sequence(self) -> abc.Iterable[tuple[str, int]]:
#         """
#         Get sequence (e.g., residues) and its numbering.
#         """


# class AbstractChain(metaclass=ABCMeta):
#     """
#     Chain basic interface definition.
#     """
#
#     __slots__ = ()
#
#     @classmethod
#     @abstractmethod
#     def read(cls, path: Path, dump_names: _DumpNames = DumpNames, **kwargs):
#         """
#         Read an object.
#         """
#
#     @abstractmethod
#     def write(self, path: Path, dump_names: _DumpNames = DumpNames, **kwargs):
#         """
#         Write an object to disk.
#         """
#
#     @property
#     @abstractmethod
#     def id(self) -> str:
#         """
#         Unique identifier.
#         """


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
        self,
        seqs: abc.Iterable[tuple[str, str]] | Path,
    ) -> abc.Iterable[tuple[str, str]]:
        ...


class SeqReader(t.Protocol):
    """
    A callable reading sequences into tuples of (header, seq) pairs.
    """

    def __call__(
        self,
        inp: Path | TextIOBase | abc.Iterable[str],
    ) -> abc.Iterable[tuple[str, str]]:
        ...


class SeqWriter(t.Protocol):
    """
    A callable writing (header, seq) pairs to disk.
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
    A callable accepting and returning a pair (header, seq).
    """

    def __call__(self, seq: tuple[str, str], **kwargs) -> tuple[str, str]:
        ...


class SeqFilter(t.Protocol):
    """
    A callable accepting a pair (header, seq) and returning a boolean.
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
class ApplyTWithArgs(t.Protocol, t.Generic[T]):
    def __call__(self, x: T, *args, **kwargs) -> T:
        ...


# ApplyT = abc.Callable[[T], T]
@t.runtime_checkable
class ApplyT(t.Protocol, t.Generic[T]):
    def __call__(self, x: T) -> T:
        ...


@t.runtime_checkable
class FilterT(t.Protocol, t.Generic[T]):
    def __call__(self, x: T) -> bool:
        ...


@t.runtime_checkable
class NamedTupleT(t.Protocol, abc.Iterable):
    def __getattr__(self, item):
        ...

    def _asdict(self) -> dict[str, t.Any]:
        ...


if __name__ == '__main__':
    raise RuntimeError
