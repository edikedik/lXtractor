"""
A module handling multiple sequence alignments.
"""
from __future__ import annotations

import operator as op
import typing as t
from collections import abc
from io import TextIOBase
from itertools import tee
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

if t.TYPE_CHECKING:
    import lXtractor.chain as lxc

    _CT: t.TypeAlias = lxc.ChainSequence | lxc.ChainStructure | lxc.Chain

_Idx = t.Union[int, t.Tuple[int, ...]]
_ST: t.TypeAlias = tuple[str, str]
T = t.TypeVar("T")
_GAPS = ("-", ".")


class Alignment:
    # TODO: consider directly inheriting from MutableSequence
    # TODO: implement vcat(other) or | operator
    # TODO: support column names
    """
    An MSA resource: a collection of aligned sequences.
    """

    __slots__ = ("seqs", "add_method", "align_method", "_seqs_map")

    def __init__(
        self,
        seqs: abc.Iterable[_ST],
        add_method: AddMethod = mafft_add,
        align_method: AlignMethod = mafft_align,
    ):
        """
        :param seqs: An iterable with (id, _seq) pairs.
        :param add_method: A callable adding sequences.
            Check the type for a signature.
        :param align_method: A callable aligning sequences.
        """
        self.add_method: AddMethod = add_method
        self.align_method: AlignMethod = align_method
        self.seqs: list[_ST] = list(seqs)
        self._seqs_map: dict[str, str] = dict(self.seqs)
        self._verify()

    @property
    def shape(self) -> tuple[int, int]:
        """
        :return: (# sequences, # columns)
        """
        return len(self.seqs), len(self.seqs[0][1])

    def __contains__(self, item: t.Any) -> bool:
        if isinstance(item, str):
            return item in self._seqs_map
        if isinstance(item, tuple):
            return item in self.seqs
        return False

    def __len__(self) -> int:
        return len(self.seqs)

    def __iter__(self) -> abc.Iterator[_ST]:
        return iter(self.seqs)

    @t.overload
    def __getitem__(self, item: int) -> _ST:
        ...

    @t.overload
    def __getitem__(self, item: slice) -> list[_ST]:
        ...

    @t.overload
    def __getitem__(self, item: str) -> str:
        ...

    def __getitem__(self, item: str | int | slice) -> _ST | str | list[_ST]:
        match item:
            case str():
                return self._seqs_map[item]
            case int() | slice():
                return self.seqs[item]
            case _:
                raise TypeError(f"Unsupported item type {type(item)}")

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, Alignment):
            return False
        return self.seqs == other.seqs

    def __add__(self, other: Alignment) -> Alignment:
        return Alignment(self.seqs + other.seqs)

    def __sub__(self, other: Alignment) -> Alignment:
        return self.remove(other)

    def _verify(self) -> None:
        if len(self.seqs) < 1:
            raise InitError("Alignment must contain at least one sequence")
        lengths = set(map(len, self._seqs_map.values()))
        if len(lengths) > 1:
            raise InitError(
                f"Expected all _seqs to have the same length, got {lengths}"
            )

    def annotate(
        self,
        objs: abc.Iterable[_CT],
        map_name: str,
        accept_fn: abc.Callable[[_CT], bool] | None = None,
        **kwargs,
    ):
        """
        This function "annotates" sequence segments using MSA.

        Namely, it adds each sequence of the provided chain-type objects to
        sequences currently present in this MSA via :attr:`add_method`.
        The latter is expected to preserve the original number of MSA columns,
        whereas potentially cutting the original sequence, thereby defining
        MSA-imposed boundaries. These are used to extract a child object
        using ``spawn_child`` method, which will have the corresponding MSA
        numbering written under `map_name`.

        :param objs: An iterable over chain-type objects.
        :param map_name: A name to use for storing the derived MSA numbering map.
        :param accept_fn: A function accepting a chain-type object and returning
            a boolean value indicating whether the spawn child sequence should
            be preserved.
        :param kwargs: Additional keyword arguments passed to the
            ``spawn_child()`` method.
        :return: An iterator over spawned child objects. These are automatically
            stored under the ``children`` attribute of each chain-type object,
            in which case it's safe to simply consume the returned iterator.
        """
        def enumerate_aligned(s: str) -> list[int]:
            return [i for i, c in enumerate(s, start=1) if c not in _GAPS]

        def derive_boundaries(seq_full: str, seq_partial: str):
            s = seq_partial
            for c in _GAPS:
                s = s.replace(c, "")
            start_idx = seq_full.find(s) + 1
            if start_idx == -1:
                raise RuntimeError(
                    "An aligned sequence is not a subsequence of the full sequence."
                )
            end_idx = start_idx + len(s) - 1
            return start_idx, end_idx

        objs1, objs2 = tee(objs)
        seqs = ((obj.id, obj.seq.seq1) for obj in objs1)
        seqs_aligned = self.add_method(self, seqs)
        for obj, (_, s_aln) in zip(objs2, seqs_aligned):
            s_aln_enum = enumerate_aligned(s_aln)
            s_start, s_end = derive_boundaries(obj.seq.seq1, s_aln)
            assert s_end - s_start + 1 == len(s_aln_enum)
            child = obj.spawn_child(s_start, s_end, map_name, **kwargs)
            child[map_name] = s_aln_enum
            if accept_fn is not None:
                if accept_fn(child):
                    yield child
                else:
                    obj.children.remove(child)
            else:
                yield child

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
        cols: abc.Iterator[list[str]] | abc.Iterator[str]
        cols = chunked(interleave(*map(op.itemgetter(1), self.seqs)), len(self))
        if join:
            cols = map("".join, cols)
        return cols

    def slice(self, start: int, stop: int, step: int | None = None) -> t.Self:
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
            return header, "".join(
                islice_extended(seq, start - 1 if start > 0 else start, stop, step)
            )

        return self.__class__(map(slice_one, self.seqs))

    def align(self, seq: abc.Iterable[_ST] | _ST | Alignment) -> t.Self:
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
        :return: A new alignment object with sequences from `_seq`.
            The original number of columns should be preserved,
            which is true when using the default :attr:`add_method`.
        """

        def is_pair(x: object) -> t.TypeGuard[_ST]:
            return isinstance(x, tuple) and len(x) == 2

        if isinstance(seq, tuple):
            if is_pair(seq):
                seq = [seq]
            else:
                raise ValueError(f"Expected two-element tuple, got {len(seq)} elements")

        seqs = self.add_method(self, seq)
        return self.__class__(seqs)

    def realign(self):
        """
        Realign sequences in :attr:`seqs` using :attr:`align_method`.

        :return: A new :class:`Alignment` object with realigned sequences.
        """
        return self.__class__(
            self.align_method(self.seqs),
            align_method=self.align_method,
            add_method=self.add_method,
        )

    def add(
        self,
        other: abc.Iterable[_ST] | _ST | Alignment,
    ) -> t.Self:
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
        :return: A new :class:`Alignment` object with added sequences.
        """

        aligned_other = self.align(other)
        return self + aligned_other

    def remove(
        self,
        item: str | _ST | t.Iterable[str] | t.Iterable[_ST],
        error_if_missing: bool = True,
        realign: bool = False,
    ) -> t.Self:
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
                    raise MissingData(f"No such item {_x} to remove")

        getter = op.itemgetter(0) if isinstance(items[0], str) else identity
        seqs = filter(lambda x: getter(x) not in items, self.seqs)
        if realign:
            return self.__class__(self.align_method(seqs))
        return self.__class__(seqs)

    def filter(self, fn: SeqFilter) -> t.Self:
        """
        Filter alignment sequences.

        :param fn: A function accepting a sequence
            -- (name, _seq) pair -- and returning a boolean.
        :return: A new :class:`Alignment` object with
            filtered sequences.
        """
        return self.__class__(filter(fn, self.seqs))

    def filter_gaps(self, max_frac: float = 1.0, dim: int = 0) -> t.Self:
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
            return self.__class__(
                ((x, self._seqs_map[x]) for x in ids),
                add_method=self.add_method,
                align_method=self.align_method,
            )
        if dim == 1:
            seqs, _ = remove_gap_columns(
                map(op.itemgetter(1), self.seqs), max_gaps=max_frac
            )
            return self.__class__(
                zip(map(op.itemgetter(0), self.seqs), seqs),
                add_method=self.add_method,
                align_method=self.align_method,
            )
        raise ValueError(f"Invalid dim {dim}")

    def map(self, fn: SeqMapper) -> t.Self:
        """
        Map a function to sequences.

        >>> a = Alignment([('A', 'AB---')])
        >>> a.map(lambda x: (x[0].lower(), x[1].replace('-', '*'))).seqs
        [('a', 'AB***')]

        :param fn: A callable accepting and returning a sequence.
        :return: A new :class:`Alignment` object.
        """
        return self.__class__(map(fn, self.seqs))

    @classmethod
    def read(
        cls,
        inp: Path | TextIOBase | abc.Iterable[str],
        read_method: SeqReader = read_fasta,
        add_method: AddMethod = mafft_add,
        align_method: AlignMethod = mafft_align,
    ) -> t.Self:
        """
        Read sequences and create an alignment.

        :param inp: A Path to aligned sequences, or a file handle,
            or iterable over file lines.
        :param read_method: A method accepting `inp` and returning an iterable
            over pairs (header, _seq). By default, it's :func:`read_fasta`.
            Hence, the default expected format is fasta.
        :param add_method: A sequence addition method for a new
            :class:`Alignment` object.
        :param align_method: An alignment method for a new
            :class:`Alignment` object.
        :return: An alignment with sequences read parsed from
            the provided input.
        """
        return cls(read_method(inp), add_method=add_method, align_method=align_method)

    @classmethod
    def make(
        cls,
        seqs: abc.Iterable[_ST],
        method: AlignMethod = mafft_align,
        add_method: AddMethod = mafft_add,
        align_method: AlignMethod = mafft_align,
    ) -> Alignment:
        """
        Create a new alignment from a collection of unaligned sequences.
        For aligned sequences, please utilize :meth:`read`.

        :param seqs: An iterable over (header, _seq) objects.
        :param method: A callable accepting unaligned sequences
            and returning the aligned ones.
        :param add_method: A sequence addition method for
            a new :class:`Alignment` object.
        :param align_method: An alignment method for
            a new :class:`Alignment` object.
        :return: An alignment created from aligned `seqs`.
        """
        return cls(method(seqs), add_method=add_method, align_method=align_method)

    @classmethod
    def read_make(
        cls,
        inp: Path | TextIOBase | abc.Iterable[str],
        read_method: SeqReader = read_fasta,
        add_method: AddMethod = mafft_add,
        align_method: AlignMethod = mafft_align,
    ) -> t.Self:
        """
        A shortcut combining :meth:`read` and :meth:`make`.

        It parses sequences from `inp`, aligns them and creates
            the :class:`Alignment` object.

        :param inp: A Path to aligned sequences, or a file handle,
            or iterable over file lines.
        :param read_method: A method accepting `inp` and returning an iterable
            over pairs (header, _seq). By default, it's :func:`read_fasta`.
            Hence, the default expected format is fasta.
        :param add_method: A sequence addition method for a new
            :class:`Alignment` object.
        :param align_method: An alignment method for a new
            :class:`Alignment` object.
        :return: An alignment from parsed and aligned `inp` sequences.
        """

        return cls(
            align_method(read_method(inp)),
            add_method=add_method,
            align_method=align_method,
        )

    def write(
        self, out: Path | SupportsWrite, write_method: SeqWriter = write_fasta
    ) -> None:
        """
        Write an alignment.

        :param out: Any object with the `write` method.
        :param write_method: The writing function itself, accepting sequences
            and `out`. By default, use `read_fasta` to write in fasta format.
        :return: Nothing.
        """
        write_method(self.seqs, out)


if __name__ == "__main__":
    raise RuntimeError
