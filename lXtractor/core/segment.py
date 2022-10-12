from __future__ import annotations

import typing as t
from collections import namedtuple, abc
from copy import deepcopy
from itertools import islice

from more_itertools import zip_equal, nth, always_reversible

from lXtractor.core.base import Ord
from lXtractor.core.config import Sep
from lXtractor.core.exceptions import LengthMismatch, NoOverlap
from lXtractor.variables.base import Variables

_I = t.TypeVar('_I', bound=t.Union[int, slice])
# _IterType = t.Union[abc.Iterator[tuple], abc.Iterator[namedtuple]]
DATA_HANDLE_MODES = ('merge', 'self', 'other')


class Segment(abc.Sequence):
    """
    An arbitrary segment with boundaries included.
    """

    __slots__ = ('start', 'end', 'name', 'parent', 'children', 'meta', '_seqs', 'variables')

    def __init__(
            self, start: int, end: int,
            name: t.Optional[str] = None,
            seqs: t.Optional[t.Dict[str, t.Sequence[t.Any]]] = None,
            parent: t.Optional[Segment] = None,
            children: t.Optional[dict[str, Segment]] = None,
            meta: t.Optional[t.Dict[str, t.Any]] = None,
            variables: t.Optional[Variables] = None
    ):
        self.start = start
        self.end = end
        self.name = name
        self.parent = parent
        self.children = children or {}
        self.meta = meta or {}
        self._seqs = seqs or {}
        self.variables: Variables = variables or Variables()

        self._setup_and_validate()

    @property
    def id(self) -> str:
        parent = f'<-({self.parent.id})' if self.parent else ''
        return f'{self.name}{Sep.start_end}{self.start}-{self.end}{parent}'

    @property
    def item_type(self) -> namedtuple:
        return namedtuple('Item', ['i', *self._seqs.keys()])

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def __iter__(self) -> abc.Iterator[tuple] | abc.Iterator[namedtuple]:
        items = range(self.start, self.end + 1)
        item_type = self.item_type
        if self._seqs:
            items = (
                item_type(i, *x) for i, x in
                zip_equal(items, zip(*self._seqs.values())))
        yield from items

    def __getitem__(self, idx: slice | int | str) -> (
            abc.Iterator[tuple] | abc.Iterator[namedtuple] |
            tuple | namedtuple | t.Sequence):
        idx = translate_idx(idx, self.start)
        if isinstance(idx, slice):
            stop = idx.stop + 1 if isinstance(idx.stop, int) else idx.stop
            idx = slice(idx.start, stop, idx.step)
            if idx.start and idx.start < 0:
                return iter([])
            return islice(iter(self), idx.start, idx.stop, idx.step)
        elif isinstance(idx, int):
            return nth(iter(self), idx)
        elif isinstance(idx, str):
            return self._seqs[idx]
        else:
            raise TypeError(f'Unsupported idx type {type(idx)}')

    def __setitem__(self, key: str, value: t.Sequence[t.Any]) -> None:
        self._validate_seq(key, value)
        self._seqs[key] = value

    def __reversed__(self) -> abc.Iterator[tuple] | abc.Iterator[namedtuple]:
        return always_reversible(iter(self))

    def __contains__(self, item: str | Ord) -> bool:
        match item:
            case str():
                return item in self._seqs
            case _:
                return self.start <= item <= self.end

    def __len__(self):
        return self.end - self.start + 1

    def __and__(self, other: Segment) -> Segment:
        return self.overlap_with(other, True, 'self')

    def __rshift__(self, idx) -> Segment:
        self.start += idx
        self.end += idx
        return self

    def __lshift__(self, idx) -> Segment:
        self.start -= idx
        self.end -= idx
        return self

    def _validate_seq(self, name: str, seq: t.Sequence):
        if len(seq) != len(self):
            raise LengthMismatch(
                f"Len({name})={len(seq)} doesn't match the segment's length {len(self)}")

    def _setup_and_validate(self):
        if self.start > self.end:
            raise ValueError(f'Invalid boundaries {self.start}, {self.end}')

        for k, seq in self._seqs.items():
            if len(seq) != len(self):
                self._validate_seq(k, seq)

    def add_seq(self, name: str, seq: t.Sequence[t.Any]):
        if name not in self:
            self[name] = seq
        else:
            raise ValueError(f'Segment already contains {name}. '
                             f'To overwrite existing sequences, use [] syntax')

    def bounds(self, other: Segment) -> bool:
        """
        self: +-------+

        other:  +----+

        => True
        """
        return other.start >= self.start and self.end >= other.end

    def bounded_by(self, other: Segment) -> bool:
        """
        self:   +----+

        other: +------+

        => True
        """
        return self.start >= other.start and other.end >= self.end

    def overlaps(self, other: Segment) -> bool:
        """
        Check whether a segment overlaps with the other segment.
        Use :meth:`overlap_with` to produce an overlapping child :class:`Segment`.

        :param other: other :class:`Segment` instance.
        :return: ``True`` if segments overlap and ``False`` otherwise.
        """
        return not (other.start > self.end or self.start > other.end)

    def overlap_with(
            self, other: Segment, deep_copy: bool = True,
            handle_mode: str = 'merge', sep: str = '&'
    ) -> t.Optional[Segment]:
        """
        self: +--------+

        other:   +-------+

        =>       +-----+

        :param other: other :class:`Segment` instance.
        :param deep_copy: deepcopy seqs and metadata.
        :param handle_mode: When the child overlapping segment is created,
            this parameter defines how :attr:`name` and :attr:`data` are handled.
            The following values are possible:
                - "merge": merge data and name from `self` and `other`
                - "self": the current instance provides both attributes
                - "other": `other` provides both attributes
        :param sep: If `handle_mode` == "merge", the new name is created by joining
            names of `self` and `other` using this separator.
        :return: New segment instance with inherited name and data.
        """

        def subset_seqs(
                _seqs: t.Dict[str, t.Sequence],
                curr_start: int, ov_start: int, ov_end: int
        ) -> t.Dict[str, t.Sequence]:
            _start, _end = ov_start - curr_start, ov_end - curr_start
            return {k: s[_start: _end + 1] for k, s in _seqs.items()}

        if not self.overlaps(other):
            raise NoOverlap

        start, end = max(self.start, other.start), min(self.end, other.end)

        if handle_mode == 'merge':
            data = {**self.meta, **other.meta}
            seqs = {
                **subset_seqs(self._seqs, self.start, start, end),
                **subset_seqs(other._seqs, other.start, start, end)}
            name = sep.join(map(str, [self.name, other.name]))
        elif handle_mode == 'self':
            data, name = self.meta, self.name
            seqs = subset_seqs(self._seqs, self.start, start, end)
        elif handle_mode == 'other':
            data, name = other.meta, other.name
            seqs = subset_seqs(other._seqs, other.start, start, end)
        else:
            raise ValueError(f'Handle mode {handle_mode} is not supported. '
                             f'Supported modes are {DATA_HANDLE_MODES}')

        if deep_copy:
            data, seqs = deepcopy(data), deepcopy(seqs)

        return self.__class__(start, end, name=name, seqs=seqs, parent=self, meta=data)

    def overlap(self, start: int, end: int) -> Segment:
        """
        Create new segment from the current instance using overlapping boundaries.

        :param start: Starting coordinate.
        :param end: Ending coordinate.
        :return: New overlapping segment with :attr:`data` and :attr:`name`
        """
        other = Segment(start, end)

        if not self.overlaps(other):
            raise NoOverlap

        return self.overlap_with(other, True, 'self')

    def sub_by(self, other: Segment, **kwargs) -> Segment:
        """
        A specialized version of :meth:`overlap_with` used in cases where `other`
        is assumed to be a part of the current segment (hence, a subsegment).

        :param other: Some other segment contained within the (`start`, `end`) boundaries.
        :param kwargs: Passed to :meth:`overlap_with`.
        :return: A new :class:`Segment` object with boundaries of `other`.
            See :meth:`overlap_with` on how to handle segments' names and data.
        :raises NoOverlap: If `other`'s boundaries lie outside the existing
            :attr:`start`, :attr:`end`.
        """
        if not self.bounds(other):
            raise NoOverlap(
                f'Provided ({other.start, other.end}) boundaries are not within the existing '
                f'boundaries ({self.start, self.end})')

        return self.overlap_with(other, **kwargs)

    def sub(self, start: int, end: int, **kwargs) -> Segment:
        """
        Subset current segment using provided boundaries.
        Will create a new segment and call :meth:`sub_by`.

        :param start: new start.
        :param end: new end.
        :param kwargs: passed to :meth:`overlap_with`
        """
        return self.sub_by(Segment(start, end), **kwargs)


def translate_idx(idx: _I, offset: int) -> _I:
    if isinstance(idx, slice):
        start = idx.start
        stop = idx.stop
        if start is not None:
            start -= offset
            if start < 0:
                start = 0
        if stop is not None:
            stop -= offset
        return slice(start, stop, idx.step)
    elif isinstance(idx, int):
        return idx - offset
    else:
        return idx


if __name__ == '__main__':
    raise RuntimeError
