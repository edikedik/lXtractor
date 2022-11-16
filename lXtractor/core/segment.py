from __future__ import annotations

import logging
import typing as t
from collections import namedtuple, abc
from copy import deepcopy, copy
from itertools import islice, combinations, filterfalse

import networkx as nx
from more_itertools import zip_equal, nth, always_reversible, powerset, take
from tqdm.auto import tqdm

from lXtractor.core.base import Ord
from lXtractor.core.config import Sep
from lXtractor.core.exceptions import LengthMismatch, NoOverlap
from lXtractor.variables.base import Variables

_I = t.TypeVar('_I', bound=t.Union[int, slice])
# _IterType = t.Union[abc.Iterator[tuple], abc.Iterator[namedtuple]]
DATA_HANDLE_MODES = ('merge', 'self', 'other')
LOGGER = logging.getLogger(__name__)


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
        :param deep_copy: deepcopy seqs to avoid side effects.
        :param handle_mode: When the child overlapping segment is created,
            this parameter defines how :attr:`name` and :attr:`meta` are handled.
            The following values are possible:
                - "merge": merge meta and name from `self` and `other`
                - "self": the current instance provides both attributes
                - "other": `other` provides both attributes
        :param sep: If `handle_mode` == "merge", the new name is created by joining
            names of `self` and `other` using this separator.
        :return: New segment instance with inherited name and meta.
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
            meta = {**self.meta, **other.meta}
            seqs = {
                **subset_seqs(self._seqs, self.start, start, end),
                **subset_seqs(other._seqs, other.start, start, end)}
            name = sep.join(map(str, [self.name, other.name]))
        elif handle_mode == 'self':
            meta, name = self.meta, self.name
            seqs = subset_seqs(self._seqs, self.start, start, end)
        elif handle_mode == 'other':
            meta, name = other.meta, other.name
            seqs = subset_seqs(other._seqs, other.start, start, end)
        else:
            raise ValueError(f'Handle mode {handle_mode} is not supported. '
                             f'Supported modes are {DATA_HANDLE_MODES}')

        meta = copy(meta)
        if deep_copy:
            seqs = deepcopy(seqs)

        return self.__class__(start, end, name=name, seqs=seqs, parent=self, meta=meta)

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


def segments2graph(segments: abc.Iterable[Segment]) -> nx.Graph:
    """
    Convert segments to an undirected graph such that segments are nodes
    and edges are drawn between overlapping segments.

    :param segments: an iterable with segments objects.
    :return: an undirected graph.
    """
    g = nx.Graph()
    for s in segments:
        g.add_node(s)
        edges = [(s, n) for n in g.nodes if n != s and s.overlaps(n)]
        if edges:
            g.add_edges_from(edges)
    return g


def do_overlap(segments: abc.Iterable[Segment]) -> bool:
    """
    Check if any pair of segments overlap.

    :param segments: an iterable with at least two segments.
    :return: ``True`` if there are overlapping segments, ``False`` otherwise.
    """
    return any(s1.overlaps(s2) for s1, s2 in combinations(segments, 2))


def resolve_overlaps(
        segments: abc.Iterable[Segment],
        value_fn: abc.Callable[[Segment], float] = len,
        max_it: int | None = None,
        verbose: bool = False,
) -> abc.Generator[Segment]:
    """
    Eliminate overlapping segments.

    Convert segments into and undirected graph (see :func:`segments2graph`).
    Iterate over connected components.
    If a component has only a single node (no overlaps§), yield it.
    Otherwise, consider all possible non-overlapping subsets of nodes.
    Find a subset such that the sum of the `value_fn` over the segments is
    maximized and yield nodes from it.

    :param segments: A collection of possibly overlapping segments.
    :param value_fn: A function accepting the segment and returning its value.
    :param max_it: The maximum number of subsets to consider when resolving a
        group of overlapping segments.
    :param verbose: Progress bar and general info.
    :return: A collection of non-overlapping segments with maximum cumulative value.
        Note that the optimal solution is guaranteed iff the number of possible subsets
        for an overlapping group does not exceed `max_it`.
    """
    # TODO: option to fallback to a greedy strategy when reaching `max_it`

    g = segments2graph(segments)
    ccs = nx.connected_components(g)
    if verbose:
        ccs = list(ccs)
        LOGGER.info(f'Found {len(ccs)} connected components with sizes: '
                    f'{list(map(len, ccs))}')
    for i, cc in enumerate(nx.connected_components(g), start=1):
        if len(cc) == 1:
            yield cc.pop()
        else:
            sets = powerset(cc)
            if verbose:
                sets = tqdm(sets, desc=f'Resolving cc {i} with size: {len(cc)}')
            if max_it is not None:
                sets = take(max_it, sets)
            overlapping_subsets = filterfalse(do_overlap, sets)
            yield from max(
                overlapping_subsets,
                key=lambda xs: sum(map(value_fn, xs))
            )


if __name__ == '__main__':
    raise RuntimeError
