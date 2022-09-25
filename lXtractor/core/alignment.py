from __future__ import annotations

import logging
import operator as op
import typing as t
from collections import abc
from io import TextIOBase
from pathlib import Path

from more_itertools import interleave, chunked, islice_extended
from toolz import identity

from lXtractor.core.base import (
    AddMethod, AlignMethod, SeqReader, SeqWriter, SupportsWrite, SeqMapper, SeqFilter)
from lXtractor.core.exceptions import AmbiguousData, InitError, MissingData
from lXtractor.util.seq import (
    mafft_add, mafft_align, read_fasta, write_fasta, remove_gap_columns, partition_gap_sequences)

_Idx = t.Union[int, t.Tuple[int, ...]]
LOGGER = logging.getLogger(__name__)


class Alignment:
    """
    An MSA resource: a list of proteins sequences with the same length.
    """

    def __init__(
            self, seqs: abc.Iterable[tuple[str, str]],
            add_method: AddMethod = mafft_add,
            align_method: AlignMethod = mafft_align
    ):
        """
        :param seqs: An iterable with (id, seq) pairs.
        :param add_method: A callable adding sequences. Check the type for a signature.
        :param align_method: A callable aligning sequences.
        """
        self.add_method = add_method
        self.align_method = align_method
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
        elif isinstance(item, tuple):
            return item in self.seqs
        return False

    def __len__(self) -> int:
        return len(self.seqs)

    def __iter__(self) -> abc.Iterator[tuple[str, str]]:
        return iter(self.seqs)

    def __getitem__(self, item: str | int | slice) -> tuple[str, str] | str:
        if isinstance(item, str):
            return self._seqs_map[item]
        if isinstance(item, (int, slice)):
            return self.seqs[item]
        else:
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
            raise InitError(f'Expected all _seqs to have the same length, got {lengths}')

    def itercols(self, *, join: bool = True) -> abc.Iterator[str] | abc.Iterator[list[str]]:
        cols = chunked(interleave(*map(op.itemgetter(1), self.seqs)), len(self))
        if join:
            cols = map(lambda x: ''.join(x), cols)
        return cols

    def slice(self, start: int, stop: int, step: t.Optional[int] = None) -> Alignment:
        def slice_one(item):
            header, seq = item
            return header, ''.join(islice_extended(seq, start, stop, step))

        return Alignment(map(slice_one, self.seqs))

    def align(
            self, seq: abc.Iterable[tuple[str, str]] | tuple[str, str] | Alignment,
            **kwargs
    ) -> Alignment:
        if isinstance(seq, tuple):
            seq = [seq]
        seqs = self.add_method(self, seq, **kwargs)
        return Alignment(seqs)

    def realign(self):
        return Alignment(self.align_method(self.seqs),
                         align_method=self.align_method,
                         add_method=self.add_method)

    def add(
            self, other: abc.Iterable[tuple[str, str]] | tuple[str, str] | Alignment,
            **kwargs
    ) -> Alignment:
        aligned_other = self.align(other, **kwargs)
        return self + aligned_other

    def remove(
            self, item: str | tuple[str, str] | t.Iterable[str] | t.Iterable[tuple[str, str]],
            error_if_missing: bool = True, realign: bool = False, **kwargs
    ) -> Alignment:

        if isinstance(item, (str, tuple)):
            items = [item]
        else:
            items = list(item)

        if error_if_missing:
            for it in items:
                if it not in self:
                    raise MissingData(f'No such item {it} to remove')

        getter = op.itemgetter(0) if isinstance(items[0], str) else identity
        seqs = filter(lambda x: getter(x) not in items, self.seqs)
        if realign:
            seqs = self.align_method(seqs, **kwargs)
        return Alignment(seqs)

    def filter(self, fn: SeqFilter) -> Alignment:
        return Alignment(filter(fn, self.seqs))

    def filter_gaps(self, max_frac: float = 1.0, dim: int = 0) -> Alignment:
        if dim == 0:
            ids, _ = partition_gap_sequences(self.seqs, max_frac)
            return Alignment(
                ((x, self._seqs_map[x]) for x in ids),
                add_method=self.add_method, align_method=self.align_method
            )
        if dim == 1:
            seqs, _ = remove_gap_columns(
                map(op.itemgetter(1), self.seqs), max_gaps=max_frac)
            return Alignment(
                zip(map(op.itemgetter(0), self.seqs), seqs),
                add_method=self.add_method, align_method=self.align_method
            )
        else:
            raise ValueError(f'Invalid dim {dim}')

    def map(self, fn: SeqMapper) -> Alignment:
        return Alignment(map(fn, self.seqs))

    @classmethod
    def read(cls, inp: Path | TextIOBase | abc.Iterable[str],
             read_method: SeqReader = read_fasta,
             add_method: AddMethod = mafft_add,
             align_method: AlignMethod = mafft_align,
             **kwargs) -> Alignment:
        return cls(read_method(inp, **kwargs),
                   add_method=add_method,
                   align_method=align_method)

    @classmethod
    def make(cls, seqs: abc.Iterable[tuple[str, str]],
             method: AlignMethod = mafft_align,
             add_method: AddMethod = mafft_add,
             align_method: AlignMethod = mafft_align,
             **kwargs) -> Alignment:
        return cls(method(seqs, **kwargs),
                   add_method=add_method,
                   align_method=align_method)

    @classmethod
    def read_make(
            cls, inp: Path | TextIOBase | abc.Iterable[str],
            read_method: SeqReader = read_fasta,
            add_method: AddMethod = mafft_add,
            align_method: AlignMethod = mafft_align,
            **kwargs
    ) -> Alignment:
        return cls(align_method(read_method(inp, **kwargs)),
                   add_method=add_method, align_method=align_method)

    def write(self, out: Path | SupportsWrite,
              write_method: SeqWriter = write_fasta,
              **kwargs) -> None:
        write_method(self.seqs, out, **kwargs)

    def map_seq_numbering(
            self, seq: SeqRec,
            seq_numbering: t.Sequence[int],
            seq_added: bool = False,
    ) -> t.Dict[int, int]:
        # TODO: support matching seq by ID and sequence to avoid unneeded alignments
        """
        Map between a sequence's numbering to the alignment columns.

        :param seq: A sequence whose numbering to map.
        :param seq_numbering: Optional numbering of the provided seq.
            Must be the same length as ``seq``.
        :param seq_added: Do not call :meth:`add_sequences`,
            assuming the sequence is already added.
        :return: A dictionary mapping ``seq_numbering`` to alignment's columns indices.
        """

        if not len(seq_numbering) == len(seq):
            raise AmbiguousData(
                f'Numbering length {len(seq_numbering)} does not match '
                f'the sequence length {len(seq)}')

        LOGGER.debug(
            f'Mapping between sequence {seq.id} (size {len(seq)}) '
            f'and the alignment columns numbering.')

        if seq_added:
            aligned_msa = seq
        else:
            # add a sequence to the _seqs and extract it right away
            _, aligned_msa = self.add_sequences([seq], False)
            aligned_msa = aligned_msa[-1]
            LOGGER.debug(
                f'Aligned and extracted sequence {len(aligned_msa.id)} '
                f'with size {len(aligned_msa)}')

        # align the extracted sequence and the original one
        # extract the (second time) aligned sequence
        aligned_ori = self.align_method([seq, aligned_msa])[-1]
        LOGGER.debug(f'Aligned MSA-extracted and original sequence '
                     f'to obtain a sequence {aligned_ori.id} '
                     f'with size {len(aligned_ori)}')

        # Obtain the filtered numbering with the sequence elements
        # really present in the MSA-aligned sequence
        ori2aligned = (i for i, c in zip(seq_numbering, aligned_ori) if c != '-')

        return {i: next(ori2aligned) for i, c in
                enumerate(aligned_msa, start=1) if c != '-'}


if __name__ == '__main__':
    raise RuntimeError
