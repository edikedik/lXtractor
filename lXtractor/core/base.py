from __future__ import annotations

import typing as t
from abc import ABCMeta, abstractmethod
from collections import abc
from io import TextIOBase
from pathlib import Path
from typing import runtime_checkable

from lXtractor.core.config import DumpNames

T = t.TypeVar('T')
_Fetcher = t.Callable[[t.Iterable[str]], T]
_Getter = t.Callable[[T, t.Sequence[str]], t.Sequence[str]]

_MapT = t.Dict[int, t.Optional[int]]


class AminoAcidDict:
    """
    Complete and flexible amino acid dictionary, mapping between
    3->1 and 1->3-letter codes.

    >>> d = AminoAcidDict()
    >>> assert d['A'] == 'ALA'
    >>> assert d['ALA'] == 'A'
    >>> assert d['XXX'] == 'X'
    >>> assert d['X'] == 'UNK'

    """

    def __init__(
            self,
            aa1_unk: str = 'X',
            aa3_unk: str = 'UNK',
            any_unk: t.Optional[str] = None):
        """
        :param aa1_unk: unknown character when mapping 3->1
        :param aa3_unk: unknown character when mapping 1->3
        :param any_unk: unknown character when a key doesn't
            meet 1 or 3 length requirements.
        """
        self.aa1_unk = aa1_unk
        self.aa3_unk = aa3_unk
        self.any_unk = any_unk
        self._aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'THR': 'T', 'GLU': 'E',
            'ASP': 'D', 'PHE': 'F', 'TRP': 'W', 'ILE': 'I',
            'VAL': 'V', 'LEU': 'L', 'LYS': 'K', 'MET': 'M',
            'ASN': 'N', 'GLN': 'Q', 'SER': 'S', 'ARG': 'R',
            'TYR': 'Y', 'HIS': 'H', 'PRO': 'P', 'GLY': 'G',
            'A': 'ALA', 'C': 'CYS', 'T': 'THR', 'E': 'GLU',
            'D': 'ASP', 'F': 'PHE', 'W': 'TRP', 'I': 'ILE',
            'V': 'VAL', 'L': 'LEU', 'K': 'LYS', 'M': 'MET',
            'N': 'ASN', 'Q': 'GLN', 'S': 'SER', 'R': 'ARG',
            'Y': 'TYR', 'H': 'HIS', 'P': 'PRO', 'G': 'GLY'}

    @property
    def aa_dict(self) -> t.Dict[str, str]:
        return self._aa_dict

    @property
    def proto_mapping(self) -> t.Dict[str, str]:
        """
        :return: unprotonated version of an amino acid code
        """
        return {'e': 'E', 'd': 'D', 'k': 'K', 'y': 'Y', 'j': 'H', 'h': 'H',
                'GLH': 'GLU', 'ASH': 'ASP', 'LYN': 'LYS',
                'TYD': 'TYR', 'HID': 'HIP', 'HIE': 'HIP'}

    @property
    def three_letter_codes(self) -> t.List[str]:
        """
        :return: list of available 3-letter codes
        """
        return list(filter(lambda x: len(x) == 3, self._aa_dict))

    @property
    def one_letter_codes(self) -> t.List[str]:
        """
        :return: list of available 1-letter codes
        """
        return list(filter(lambda x: len(x) == 1, self._aa_dict))

    def __getitem__(self, item):
        if item in self._aa_dict:
            return self._aa_dict[item]
        if len(item) == 3:
            return self.aa1_unk
        elif len(item) == 1:
            return self.aa3_unk
        else:
            if self.any_unk is not None:
                return self.any_unk
            raise KeyError(
                f'Expected 3-sized or 1-sized item, '
                f'got {len(item)}-sized {item}')

    def __contains__(self, item):
        return item in self.aa_dict


class AbstractResource(metaclass=ABCMeta):
    """
    Abstract base class defining basic interface any resource must provide.
    """

    def __init__(self, resource_path: t.Optional[Path],
                 resource_name: t.Optional[str]):
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


class AbstractStructure(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def read(cls, path: Path): pass

    @abstractmethod
    def write(self, path: Path): pass

    @abstractmethod
    def get_sequence(self) -> abc.Iterable[tuple[str, str, int]]: pass


class AbstractChain(metaclass=ABCMeta):
    """
    Protein basic interface definition.
    """

    @classmethod
    @abstractmethod
    def read(cls, path: Path, dump_names: DumpNames = DumpNames): pass

    @abstractmethod
    def write(self, path: Path, dump_names: DumpNames = DumpNames): pass

    @property
    @abstractmethod
    def id(self) -> str: pass


class Ord(t.Protocol[T]):
    def __le__(self, other: T) -> bool: pass

    def __ge__(self, other: T) -> bool: pass

    def __eq__(self, other: T) -> bool: pass


@runtime_checkable
class SupportsWrite(t.Protocol):
    def write(self, data): ...


@runtime_checkable
class AddMethod(t.Protocol):
    def __call__(
            self,
            msa: abc.Iterable[tuple[str, str]] | Path,
            seqs: abc.Iterable[tuple[str, str]], **kwargs
    ) -> abc.Iterable[tuple[str, str]]: ...


@runtime_checkable
class AlignMethod(t.Protocol):
    def __call__(
            self, seqs: abc.Iterable[tuple[str, str]] | Path, **kwargs
    ) -> abc.Iterable[tuple[str, str]]: ...


class SeqReader(t.Protocol):
    def __call__(
            self, inp: Path | TextIOBase | abc.Iterable[str], **kwargs
    ) -> abc.Iterable[tuple[str, str]]: ...


class SeqWriter(t.Protocol):
    def __call__(
            self, inp: abc.Iterable[tuple[str, str]],
            out: Path | SupportsWrite, **kwargs
    ) -> None: ...


class SeqMapper(t.Protocol):
    def __call__(self, seq: tuple[str, str], **kwargs) -> tuple[str, str]: ...


class SeqFilter(t.Protocol):
    def __call__(self, seq: tuple[str, str], **kwargs) -> bool: ...


if __name__ == '__main__':
    raise RuntimeError
