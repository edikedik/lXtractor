"""
A module handling multiple sequence alignments.
"""
from __future__ import annotations

import operator as op
import typing as t
from collections import abc
from io import TextIOBase
from pathlib import Path

from more_itertools import interleave, chunked, islice_extended
from toolz import identity

from lXtractor.core.base import (
    AddMethod,
    AlignMethod,
    SeqReader,
    SeqWriter,
    SupportsWrite,
    SeqMapper,
    SeqFilter,
)
from lXtractor.core.exceptions import InitError, MissingData
from lXtractor.util.seq import (
    mafft_add,
    mafft_align,
    read_fasta,
    write_fasta,
    remove_gap_columns,
    partition_gap_sequences,
)

_Idx = t.Union[int, t.Tuple[int, ...]]


class Alignment:
    """
    An MSA resource: a collection of aligned sequences.
    """

    __slots__ = ('seqs', 'add_method', 'align_method', '_seqs_map')

    def __init__(
        self,
        seqs: abc.Iterable[tuple[str, str]],
        add_method: AddMethod = mafft_add,
        align_method: AlignMethod = mafft_align,
    ):
        """
        :param seqs: An iterable with (id, seq) pairs.
        :param add_method: A callable adding sequences.
            Check the type for a signature.
        :param align_method: A callable aligning sequences.
        """
        self.add_method: AddMethod = add_method
        self.align_method: AlignMethod = align_method
        self.seqs: list[tuple[str, str]] = list(seqs)
        self._seqs_map: dict[str, str] = dict(self.seqs)
        self._verify()

    @property
    def shape(self) -> t.Tuple[int, int]:
        """
        :return: (# sequences, # columns)
        """
        return len(self.seqs), len(self.seqs[0][1])

    def __contains__(self, item: str | tuple[str, str]) -> bool:
        if isinstance(item, str):
            return item in self._seqs_map
        if isinstance(item, tuple):
            return item in self.seqs
        return False

    def __len__(self) -> int:
        return len(self.seqs)

    def __iter__(self) -> abc.Iterator[tuple[str, str]]:
        return iter(self.seqs)

    def __getitem__(self, item: str | int | slice) -> tuple[str, str] | str:
        match item:
            case str():
                return self._seqs_map[item]
            case int() | slice():
                return self.seqs[item]
            case _:
                raise TypeError(f'Unsupported item type {type(item)}')

    def __eq__(self, other: Alignment):
        return self.seqs == other.seqs

    def __add__(self, other: Alignment) -> Alignment:
        return Alignment(self.seqs + other.seqs)

    def __sub__(self, other: Alignment) -> Alignment:
        return self.remove(other)

    def _verify(self) -> None:
        if len(self.seqs) < 1:
            raise InitError('Alignment must contain at least one sequence')
        lengths = set(map(len, self._seqs_map.values()))
        if len(lengths) > 1:
            raise InitError(
                f'Expected all _seqs to have the same length, got {lengths}'
            )

    def itercols(
        self, *, join: bool = True
    ) -> abc.Iterator[str] | abc.Iterator[list[str]]:
        """
        Iterate over the Alignment columns.

        >>> a = Alignment([('A', 'ABCD'), ('X', 'XXXX')])
        >>> list(a.itercols())
        ['AX', 'BX', 'CX', 'DX']

        :param join: Join columns into a string.
        :return: An iterator over columns.
        """
        cols = chunked(interleave(*map(op.itemgetter(1), self.seqs)), len(self))
        if join:
            cols = map(''.join, cols)
        return cols

    def slice(self, start: int, stop: int, step: t.Optional[int] = None) -> Alignment:
        """
        Slice alignment columns.

        >>> a = Alignment([('A', 'ABCD'), ('X', 'XXXX')])
        >>> aa = a.slice(1, 2)
        >>> aa.shape == (2, 2)
        True
        >>>
        >>> aa.seqs[0]
        ('A', 'AB')
        >>> aa = a.slice(-4, 10)
        >>> aa.seqs[0]
        ('A', 'ABCD')

        To add the aligned sequences to the existing ones,
        use ``+`` or :meth:`add`:

        >>> aaa = a + aa
        >>> aaa.shape
        (3, 4)

        :param start: Start coordinate, boundaries inclusive.
        :param stop: Stop coordinate, boundaries inclusive.
        :param step: Step for slicing, i.e., take every column
            separated by `step - 1` number of columns.
        :return: A new alignment with sequences subset according to
            the slicing params.
        """

        def slice_one(item):
            header, seq = item
            return header, ''.join(
                islice_extended(seq, start - 1 if start > 0 else start, stop, step)
            )

        return Alignment(map(slice_one, self.seqs))

    def align(
        self, seq: abc.Iterable[tuple[str, str]] | tuple[str, str] | Alignment, **kwargs
    ) -> Alignment:
        """
        Align (add) sequences to this alignment via :attr:`add_method`.

        >>> a = Alignment([('A', 'ABCD'), ('X', 'XXXX')])
        >>> aa = a.align(('Y', 'ABXD'))
        >>> aa.shape
        (1, 4)
        >>> aa.seqs
        [('Y', 'ABXD')]

        :param seq: A sequence, iterable over sequences, or another
            :class:`Alignment`.
        :param kwargs: Passed to :attr:`add_method`.
        :return: A new alignment object with sequences from `seq`.
            The original number of columns should be preserved,
            which is true when using the default :attr:`add_method`.
        """
        if isinstance(seq, tuple):
            seq = [seq]
        seqs = self.add_method(self, seq, **kwargs)
        return Alignment(seqs)

    def realign(self):
        """
        Realign sequences in :attr:`seqs` using :attr:`align_method`.

        :return: A new :class:`Alignment` object with realigned sequences.
        """
        return Alignment(
            self.align_method(self.seqs),
            align_method=self.align_method,
            add_method=self.add_method,
        )

    def add(
        self,
        other: abc.Iterable[tuple[str, str]] | tuple[str, str] | Alignment,
        **kwargs,
    ) -> Alignment:
        """
        Add sequences to existing ones using :meth:`add`.
        This is similar to :meth:`align` but automatically adds
        the aligned seqs.

        >>> a = Alignment([('A', 'ABCD'), ('X', 'XXXX')])
        >>> aa = a.add(('Y', 'ABXD'))
        >>> aa.shape
        (3, 4)

        :param other: A sequence, iterable over sequences,
            or another :class:`Alignment`.
        :param kwargs: passed to :meth:`add`
        :return: A new :class:`Alignment` object with added sequences.
        """

        aligned_other = self.align(other, **kwargs)
        return self + aligned_other

    def remove(
        self,
        item: str | tuple[str, str] | t.Iterable[str] | t.Iterable[tuple[str, str]],
        error_if_missing: bool = True,
        realign: bool = False,
        **kwargs,
    ) -> Alignment:
        """
        Remove a sequence or collection of sequences.

        >>> a = Alignment([('A', 'ABCD-'), ('X', 'XXXX-'), ('Y', 'YYYYY')])
        >>> aa = a.remove('A')
        >>> 'A' in aa
        False
        >>> aa = a.remove(('Y', 'YYYYY'))
        >>> aa.shape
        (2, 5)
        >>> aa = a.remove(('Y', 'YYYYY'), realign=True)
        >>> aa.shape
        (2, 4)
        >>> aa['A']
        'ABCD'
        >>> aa = a.remove(['X', 'Y'])
        >>> aa.shape
        (1, 5)

        :param item: One of the following:

            - A ``str``: a sequence's name.
            - A pair ``(str, str)`` -- a name with the sequence itself.
            - An iterable over sequence enames or pairs (not mixed!)
        :param error_if_missing: Raise an error if
            any of the items are missing.
        :param realign: Realign seqs after removal.
        :param kwargs: passed to :attr:`align_method` if
            `realign` is ``True``.
        :return: A new :class:`Alignment` object with
            the remaining sequences.
        """

        if isinstance(item, (str, tuple)):
            items = [item]
        else:
            items = list(item)

        if error_if_missing:
            for _x in items:
                if _x not in self:
                    raise MissingData(f'No such item {_x} to remove')

        getter = op.itemgetter(0) if isinstance(items[0], str) else identity
        seqs = filter(lambda x: getter(x) not in items, self.seqs)
        if realign:
            seqs = self.align_method(seqs, **kwargs)
        return Alignment(seqs)

    def filter(self, fn: SeqFilter) -> Alignment:
        """
        Filter alignment sequences.

        :param fn: A function accepting a sequence
            -- (name, seq) pair -- and returning a boolean.
        :return: A new :class:`Alignment` object with
            filtered sequences.
        """
        return Alignment(filter(fn, self.seqs))

    def filter_gaps(self, max_frac: float = 1.0, dim: int = 0) -> Alignment:
        """
        Filter sequences or alignment columns
        having >= `max_frac` of gaps.

        >>> a = Alignment([('A', 'AB---'), ('X', 'XXXX-'), ('Y', 'YYYY-')])

        By default, the `max_frac` gaps is 1.0,
        which would remove solely gap-only sequences.

        >>> aa = a.filter_gaps(dim=0)
        >>> aa == a
        True

        Specifying `max_frac` removes sequences with over 50% gaps.

        >>> aa = a.filter_gaps(dim=0, max_frac=0.5)
        >>> 'A' not in aa
        True

        The last column is removed.

        >>> a.filter_gaps(dim=1).shape
        (3, 4)

        :param max_frac: a maximum fraction of allowed gaps in a sequence or a column.
        :param dim: ``0`` for sequences, ``1`` for columns.
        :return: A new :class:`Alignment` object with filtered sequences or columns.
        """
        if dim == 0:
            ids, _ = partition_gap_sequences(self.seqs, max_frac)
            return Alignment(
                ((x, self._seqs_map[x]) for x in ids),
                add_method=self.add_method,
                align_method=self.align_method,
            )
        if dim == 1:
            seqs, _ = remove_gap_columns(
                map(op.itemgetter(1), self.seqs), max_gaps=max_frac
            )
            return Alignment(
                zip(map(op.itemgetter(0), self.seqs), seqs),
                add_method=self.add_method,
                align_method=self.align_method,
            )
        raise ValueError(f'Invalid dim {dim}')

    def map(self, fn: SeqMapper) -> Alignment:
        """
        Map a function to sequences.

        >>> a = Alignment([('A', 'AB---')])
        >>> a.map(lambda x: (x[0].lower(), x[1].replace('-', '*'))).seqs
        [('a', 'AB***')]

        :param fn: A callable accepting and returning a sequence.
        :return: A new :class:`Alignment` object.
        """
        return Alignment(map(fn, self.seqs))

    @classmethod
    def read(
        cls,
        inp: Path | TextIOBase | abc.Iterable[str],
        read_method: SeqReader = read_fasta,
        add_method: AddMethod = mafft_add,
        align_method: AlignMethod = mafft_align,
        **kwargs,
    ) -> Alignment:
        """
        Read sequences and create an alignment.

        :param inp: A Path to aligned sequences, or a file handle,
            or iterable over file lines.
        :param read_method: A method accepting `inp` and returning an iterable
            over pairs (header, seq). By default, it's :func:`read_fasta`.
            Hence, the default expected format is fasta.
        :param add_method: A sequence addition method for a new
            :class:`Alignment` object.
        :param align_method: An alignment method for a new
            :class:`Alignment` object.
        :param kwargs: passed to `read_method`
        :return: An alignment with sequences read parsed from
            the provided input.
        """
        return cls(
            read_method(inp, **kwargs), add_method=add_method, align_method=align_method
        )

    @classmethod
    def make(
        cls,
        seqs: abc.Iterable[tuple[str, str]],
        method: AlignMethod = mafft_align,
        add_method: AddMethod = mafft_add,
        align_method: AlignMethod = mafft_align,
        **kwargs,
    ) -> Alignment:
        """
        Create a new alignment from a collection of unaligned sequences.
        For aligned sequences, please utilize :meth:`read`.

        :param seqs: An iterable over (header, seq) objects.
        :param method: A callable accepting unaligned sequences
            and returning the aligned ones.
        :param add_method: A sequence addition method for
            a new :class:`Alignment` object.
        :param align_method: An alignment method for
            a new :class:`Alignment` object.
        :param kwargs: Passed to `method`.
        :return: An alignment created from aligned `seqs`.
        """
        return cls(
            method(seqs, **kwargs), add_method=add_method, align_method=align_method
        )

    @classmethod
    def read_make(
        cls,
        inp: Path | TextIOBase | abc.Iterable[str],
        read_method: SeqReader = read_fasta,
        add_method: AddMethod = mafft_add,
        align_method: AlignMethod = mafft_align,
        kwargs_read: dict | None = None,
        kwargs_align: dict | None = None,
    ) -> Alignment:
        """
        A shortcut combining :meth:`read` and :meth:`make`.

        It parses sequences from `inp`, aligns them and creates
            the :class:`Alignment` object.

        :param inp: A Path to aligned sequences, or a file handle,
            or iterable over file lines.
        :param read_method: A method accepting `inp` and returning an iterable
            over pairs (header, seq). By default, it's :func:`read_fasta`.
            Hence, the default expected format is fasta.
        :param add_method: A sequence addition method for a new
            :class:`Alignment` object.
        :param align_method: An alignment method for a new
            :class:`Alignment` object.
        :param kwargs_read: Passed to the `read_method`.
        :param kwargs_align: Passed to the `align_method`.
        :return: An alignment from parsed and aligned `inp` sequences.
        """

        kwargs_read = kwargs_read or {}
        kwargs_align = kwargs_align or {}
        return cls(
            align_method(read_method(inp, **kwargs_read), **kwargs_align),
            add_method=add_method,
            align_method=align_method,
        )

    def write(
        self, out: Path | SupportsWrite, write_method: SeqWriter = write_fasta, **kwargs
    ) -> t.NoReturn:
        """
        Write an alignment.

        :param out: Any object with the `write` method.
        :param write_method: The writing function itself, accepting sequences
            and `out`. By default, use `read_fasta` to write in fasta format.
        :param kwargs: Passed to `write_method`.
        :return: Nothing.
        """
        write_method(self.seqs, out, **kwargs)


if __name__ == '__main__':
    raise RuntimeError
