"""
Module defines a segment object serving as base class for sequences
in lXtractor.
"""
from __future__ import annotations

import logging
import typing as t
from collections import namedtuple, abc
from copy import deepcopy, copy
from itertools import islice, combinations, filterfalse, chain

import networkx as nx
from more_itertools import nth, always_reversible, powerset, take
from tqdm.auto import tqdm

from lXtractor.core.base import Ord
from lXtractor.core.config import Sep
from lXtractor.core.exceptions import LengthMismatch, NoOverlap, OverlapError
from lXtractor.util.misc import is_valid_field_name
from lXtractor.variables.base import Variables

_I = t.TypeVar('_I', bound=t.Union[int, slice])
# _IterType = t.Union[abc.Iterator[tuple], abc.Iterator[namedtuple]]
DATA_HANDLE_MODES = ('merge', 'self', 'other')
LOGGER = logging.getLogger(__name__)


class Segment(abc.Sequence):
    """
    An arbitrary segment with inclusive boundaries containing arbitrary number
    of sequences.

    Sequences themselves may be retrieved via ``[]`` syntax:

    >>> s = Segment(1, 10, 'S', seqs={'X': list(range(10))})
    >>> s.id == 'S|1-10'
    True
    >>> s['X'] == list(range(10))
    True
    >>> 'X' in s
    True

    One can use the same syntax to check if a Segment contains certain index:

    >>> 1 in s and 10 in s and not 11 in s
    True

    Iteration over the segment yields it's items:

    >>> next(iter(s))
    Item(i=1, X=0)

    One can just get the same item by explicit index:

    >>> s[1]
    Item(i=1, X=0)

    Slicing returns an iterable slice object:

    >>> list(s[1:2])
    [Item(i=1, X=0), Item(i=2, X=1)]

    One can add a new sequence in two ways.

    1) using a method:

    >>> s.add_seq('Y', tuple(range(10, 20)))
    >>> 'Y' in s
    True

    2) using ``[]`` syntax:

    >>> s['Y'] = tuple(range(10, 20))
    >>> 'Y' in s
    True

    Note that using the first method, if ``s`` already contains ``Y``,
    this will cause an exception. To overwrite a sequence with the same name,
    please use explicit ``[]`` syntax.

    Additionally, one can offset Segment indices using ``>>``/``<<`` syntax.
    This operation mutates original Segment!

    >>> s >> 1
    S|2-11
    >>> 11 in s
    True

    """

    __slots__ = (
        'start',
        'end',
        'name',
        'parent',
        'children',
        'meta',
        '_seqs',
        'variables',
    )

    def __init__(
        self,
        start: int,
        end: int,
        name: str | None = None,
        seqs: dict[str, abc.Sequence[t.Any]] | None = None,
        parent: Segment | None = None,
        children: abc.Sequence | None = None,
        meta: dict[str, t.Any] | None = None,
        variables: Variables | None = None,
    ):
        """
        :param start: Start coordinate.
        :param end: End coordinate.
        :param name: The name of the segment. Name with start and end
            coordinates should uniquely specify the segmet. They are used to
            dynamically construct :meth:`id`.
        :param seqs: A dictionary name => sequence, where sequence is some
            sequence (preferably mutable) bounded by segment. Name of a
            sequence must be "simple", i.e., convertable to a field of a
            namedtuple.
        :param parent: Parental segment bounding this instance, typically
            obtained via :meth:`sub` or :meth:`sub_by` methods.
        :param children: A mapping name => :class:`Segment` with child segments
            bounded by this instance.
        :param meta: A dictionary with any meta-information str() => str()
            since reading/writing `meta` to disc will inevitably convert values
            to strings.
        :param variables: A collection of variables calculated or staged for
            calculation for this segment.
        """
        self.start = start
        self.end = end
        self.name = name
        self.parent = parent
        self.children = children or []
        self.meta = meta or {}
        self._seqs = seqs or {}
        self.variables: Variables = variables or Variables()

        self._setup_and_validate()

    @property
    def id(self) -> str:
        """
        :return: Unique segment's identifier encapsulating name, boundaries and
            parents of a segment if it was spawned from another
            :class:`Segment` instance. For example::

              S|1-2<-(P|1-10)

            would specify a segment `S` with boundaries ``[1, 2]``
            descended from `P`.
        """
        parent = f'<-({self.parent.id})' if self.parent else ''
        return f'{self.name}{Sep.start_end}{self.start}-{self.end}{parent}'

    @property
    def item_type(self) -> namedtuple:
        """
        A factory to make an `Item` namedtuple object encapsulating sequence
        names contained within this instance. The first field is reserved
        for "i" -- an index. :return: `Item` namedtuple object.
        """
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
                item_type(i, *x)
                for i, x in zip(items, zip(*self._seqs.values()), strict=True)
            )
        yield from items

    def __getitem__(
        self, idx: slice | int | str
    ) -> (
        abc.Iterator[tuple] | abc.Iterator[namedtuple] | tuple | namedtuple | t.Sequence
    ):
        idx = _translate_idx(idx, self.start)
        if isinstance(idx, slice):
            stop = idx.stop + 1 if isinstance(idx.stop, int) else idx.stop
            idx = slice(idx.start, stop, idx.step)
            if idx.start and idx.start < 0:
                return iter([])
            return islice(iter(self), idx.start, idx.stop, idx.step)
        if isinstance(idx, int):
            return nth(iter(self), idx)
        if isinstance(idx, str):
            return self._seqs[idx]
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
        if not is_valid_field_name(name):
            raise ValueError(
                f'Invalid field name {name}. '
                f'Please use a valid variable name starting with a letter'
            )
        if len(seq) != len(self):
            raise LengthMismatch(
                f"Len({name})={len(seq)} doesn't match the segment's length {len(self)}"
            )

    def _setup_and_validate(self):
        if self.start > self.end:
            raise ValueError(f'Invalid boundaries {self.start}, {self.end}')
        for name in self._seqs:
            if not is_valid_field_name(name):
                raise ValueError(
                    f'Invalid field name {name}. '
                    f'Please use a valid variable name starting with a letter'
                )

        for k, seq in self._seqs.items():
            if len(seq) != len(self):
                self._validate_seq(k, seq)

    def add_seq(self, name: str, seq: t.Sequence[t.Any]) -> t.NoReturn:
        """
        Add sequence to this segment.

        :param name: Sequence's name. Should be convertible to the
            namedtuple's field.
        :param seq: A sequence with arbitrary elements and the length of
            a segment.
        :return: returns nothing. This operation mutates `attr:`seqs`.
        :raise ValueError: If the `name` is reserved by another segment.
        """
        if not is_valid_field_name(name):
            raise ValueError(
                f'Invalid field name {name}. '
                f'Please use a valid variable name starting with a letter'
            )
        if name not in self:
            self[name] = seq
        else:
            raise ValueError(
                f'Segment already contains {name}. '
                f'To overwrite existing sequences, use [] syntax'
            )

    def bounds(self, other: Segment) -> bool:
        """
        Check if this segment bounds other.

        ::

            self: +-------+
            other:  +----+
            => True

        :param other; Another segment.
        """
        return other.start >= self.start and self.end >= other.end

    def bounded_by(self, other: Segment) -> bool:
        """
        Check whether this segment is bounded by other.

        ::

            self:   +----+
            other: +------+
            => True

        :param other; Another segment.
        """
        return self.start >= other.start and other.end >= self.end

    def overlaps(self, other: Segment) -> bool:
        """
        Check whether a segment overlaps with the other segment.
        Use :meth:`overlap_with` to produce an overlapping child
        :class:`Segment`.

        :param other: other :class:`Segment` instance.
        :return: ``True`` if segments overlap and ``False`` otherwise.
        """
        return not (other.start > self.end or self.start > other.end)

    def overlap_with(
        self,
        other: Segment,
        deep_copy: bool = True,
        handle_mode: str = 'merge',
        sep: str = '&',
    ) -> Segment | None:
        """
        Overlap this segment with other over common indices.

        ::

          self: +---------+
          other:    +-------+
          =>:       +-----+

        :param other: other :class:`Segment` instance.
        :param deep_copy: deepcopy seqs to avoid side effects.
        :param handle_mode: When the child overlapping segment is created,
            this parameter defines how :attr:`name` and :attr:`meta`
            are handled. The following values are possible:

                - "merge": merge meta and name from `self` and `other`
                - "self": the current instance provides both attributes
                - "other": `other` provides both attributes

        :param sep: If `handle_mode` == "merge", the new name is created
            by joining names of `self` and `other` using this separator.
        :return: New segment instance with inherited name and meta.
        """

        def subset_seqs(
            _seqs: dict[str, abc.Sequence], curr_start: int, ov_start: int, ov_end: int
        ) -> dict[str, abc.Sequence]:
            _start, _end = ov_start - curr_start, ov_end - curr_start
            return {k: s[_start : _end + 1] for k, s in _seqs.items()}

        if not self.overlaps(other):
            raise NoOverlap

        start, end = max(self.start, other.start), min(self.end, other.end)

        if handle_mode == 'merge':
            meta = {**self.meta, **other.meta}
            seqs = {
                **subset_seqs(self._seqs, self.start, start, end),
                **subset_seqs(other._seqs, other.start, start, end),
            }
            name = sep.join(map(str, [self.name, other.name]))
        elif handle_mode == 'self':
            meta, name = self.meta, self.name
            seqs = subset_seqs(self._seqs, self.start, start, end)
        elif handle_mode == 'other':
            meta, name = other.meta, other.name
            seqs = subset_seqs(other._seqs, other.start, start, end)
        else:
            raise ValueError(
                f'Handle mode {handle_mode} is not supported. '
                f'Supported modes are {DATA_HANDLE_MODES}'
            )

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
        A specialized version of :meth:`overlap_with` used in cases
        where `other` is assumed to be a part of the current segment
        (hence, a subsegment).

        :param other: Some other segment contained within the
            (`start`, `end`) boundaries.
        :param kwargs: Passed to :meth:`overlap_with`.
        :return: A new :class:`Segment` object with boundaries of `other`.
            See :meth:`overlap_with` on how to handle segments' names and data.
        :raises NoOverlap: If `other`'s boundaries lie outside the existing
            :attr:`start`, :attr:`end`.
        """
        if not self.bounds(other):
            raise NoOverlap(
                f'Provided ({other.start, other.end}) boundaries are not '
                f'within the existing boundaries ({self.start, self.end})'
            )

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


def _translate_idx(idx: _I, offset: int) -> _I:
    if isinstance(idx, slice):
        start = idx.start
        stop = idx.stop
        if start is not None:
            start = max(start - offset, 0)
        if stop is not None:
            stop -= offset
        return slice(start, stop, idx.step)
    if isinstance(idx, int):
        return idx - offset
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
    :return: A collection of non-overlapping segments with maximum cumulative
        value. Note that the optimal solution is guaranteed iff the number of
        possible subsets for an overlapping group does not exceed `max_it`.
    """
    # TODO: option to fallback to a greedy strategy when reaching `max_it`

    g = segments2graph(segments)
    ccs = nx.connected_components(g)
    if verbose:
        ccs = list(ccs)
        LOGGER.info(
            f'Found {len(ccs)} connected components with sizes: '
            f'{list(map(len, ccs))}'
        )
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
            yield from max(overlapping_subsets, key=lambda xs: sum(map(value_fn, xs)))


def map_segment_numbering(
    segments_from: t.Sequence[Segment], segments_to: t.Sequence[Segment]
) -> abc.Iterator[tuple[int, int]]:
    """
    Create a continuous mapping between the numberings of two segment
    collections. They must contain the same number of equal length
    non-overlapping segments. Segments in the `segments_from` collection are
    considered to span a continuous sequence, possibly interrupted due to
    discontinuities in a sequence represented by `segments_to`'s segments.
    Hence, the segments in `segments_from` form continuous numbering over
    which numberings of `segments_to` segments are joined.

    :param segments_from: A sequence of segments to map from.
    :param segments_to: A sequence of segments to map to.
    :return: An iterable over (key, value) pairs. Keys correspond to numberings
        of the `segments_from`, values -- to numberings of `segments_to`.
    """
    if len(segments_to) != len(segments_from):
        raise LengthMismatch('Segment collections must be of the same length')
    for s1, s2 in zip(segments_from, segments_to):
        if len(s1) != len(s2):
            raise LengthMismatch(
                f'Lengths of segments must match. '
                f'Got len({s1})={len(s1)}, len({s2})={len(s2)}'
            )
    for s1, s2 in zip(segments_from, segments_from[1:]):
        if s2.overlaps(s1):
            raise OverlapError(f'Segments {s1},{s2} in `segments_from` overlap')
    for s1, s2 in zip(segments_to, segments_to[1:]):
        if s2.overlaps(s1):
            raise OverlapError(f'Segments {s1},{s2} in `segments_to` overlap')

    hole_sizes = chain(
        ((s2.start - s1.end) for s1, s2 in zip(segments_to, segments_to[1:])), (0,)
    )

    return zip(
        range(segments_from[0].start, segments_from[-1].end + 1),
        chain.from_iterable(
            chain(
                range(s.start, s.end + 1), (None for _ in range(s.end + 1, h + s.end))
            )
            for s, h in zip(segments_to, hole_sizes)
        ),
    )


if __name__ == '__main__':
    raise RuntimeError
