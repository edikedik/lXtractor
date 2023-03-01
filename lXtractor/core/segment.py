"""
Module defines a segment object serving as base class for sequences
in lXtractor.
"""
from __future__ import annotations

import logging
import typing as t
import warnings
from collections import namedtuple, abc
from copy import deepcopy, copy
from itertools import combinations, filterfalse, chain

import networkx as nx
from more_itertools import always_reversible, powerset, take, nth
from tqdm.auto import tqdm
from typing_extensions import Self, reveal_type

import lXtractor.variables.base as vs
from lXtractor.core.base import Ord, NamedTupleT
from lXtractor.core.config import Sep
from lXtractor.core.exceptions import (
    LengthMismatch,
    NoOverlap,
    OverlapError,
    FormatError,
)
from lXtractor.util.misc import is_valid_field_name
# from lXtractor.variables.base import Variables

_S = t.TypeVar('_S', bound='Segment', contravariant=True)
T = t.TypeVar('T')
# _IterType = t.Union[abc.Iterator[tuple], abc.Iterator[namedtuple]]
DATA_HANDLE_MODES = ('merge', 'self', 'other')
LOGGER = logging.getLogger(__name__)


class _Item(t.Protocol):
    def __call__(self, *args, **kwargs) -> NamedTupleT:
        ...


def _check_boundary_change(x_orig: int, x_new: int):
    if x_orig != 0 and x_new == 0:
        raise IndexError(f"Can't change none-zero coordinate {x_orig} to zero")
    if x_orig == 0:
        raise IndexError("Can't change boundaries of an empty segment")
    if x_new < 0:
        raise IndexError(f'Attempting to set a negative boundary {x_new}')


class Segment(abc.Sequence[NamedTupleT]):
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
        '_start',
        '_end',
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
        parent: Self | None = None,
        children: abc.MutableSequence[Self] | None = None,
        meta: dict[str, t.Any] | None = None,
        variables: vs.Variables | None = None,
    ):
        """
        :param start: Start coordinate.
        :param end: End coordinate.
        :param name: The name of the segment. Name with start and end
            coordinates should uniquely specify the segment. They are used to
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
        if (start <= 0 or end <= 0) and start != end:
            raise ValueError('Boundaries must start from 1')
        self._start = start
        self._end = end
        self.name = name
        self.parent = parent
        self.children = children or []
        self.meta: dict[str, t.Any] = meta or {}
        self._seqs: dict[str, abc.Sequence[t.Any]] = seqs or {}
        self.variables: vs.Variables = variables or vs.Variables()

        self._setup_and_validate()

    @property
    def start(self) -> int:
        """
        :return: A Segment's start coordinate.
        """
        return self._start

    @start.setter
    def start(self, value: int):
        _check_boundary_change(self.start, value)
        if value > self.end:
            raise IndexError(
                f'Cannot start {value} further than the current end {self.end}'
            )
        if value > self.start:
            idx = _translate_idx(value, self.start)
            self._seqs = {k: s[idx:] for k, s in self._seqs.items()}
            self._start = value
            self._validate_seqs()
        else:
            if value < self.start:
                if not self._seqs:
                    self._start = value
                else:
                    raise IndexError(
                        f'Cannot set start {self.start} to {value} with '
                        'existing sequences'
                    )

    @property
    def end(self) -> int:
        """
        :return: A Segment's end coordinate.
        """
        return self._end

    @end.setter
    def end(self, value):
        _check_boundary_change(self.end, value)
        if value < self.start:
            raise IndexError(
                f'Cannot set end {value} lower than the current start {self.start}'
            )
        if value < self.end:
            idx = _translate_idx(value, self.start) + 1
            self._seqs = {k: s[:idx] for k, s in self._seqs.items()}
            self._end = value
            self._validate_seqs()
        else:
            if value > self.end:
                if not self._seqs:
                    self._end = value
                else:
                    raise IndexError(
                        f'Cannot set end {self.end} to {value} with existing sequences'
                    )

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
    def item_type(self) -> _Item:
        """
        A factory to make an `Item` namedtuple object encapsulating sequence
        names contained within this instance. The first field is reserved
        for "i" -- an index. :return: `Item` namedtuple object.
        """
        # Returns Type[Tuple[Any, ...]]
        return namedtuple('Item', ['i', *self._seqs.keys()])  # type: ignore

    @property
    def is_empty(self) -> bool:
        """
        :return: ``True`` if the segment is empty. Emptiness is a special case,
            in which :class:`Segment` ``has start == end == 0``.
        """
        return self.start == self.end == 0

    @property
    def is_singleton(self) -> bool:
        """
        :return: ``True`` if the segment contains a single element. In this
            special case, ``start == end``.
        """
        return self.start == self.end

    @property
    def seq_names(self) -> list[str]:
        """
        :return: A list of sequence names this segment entails.
        """
        return list(self._seqs)

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def __iter__(self) -> abc.Iterator[NamedTupleT]:
        if self.is_empty:
            return iter([])
        item_type = self.item_type
        enum = range(self.start, self.end + 1)
        if self._seqs:
            return (
                item_type(i, *x)
                for i, x in zip(enum, zip(*self._seqs.values()), strict=True)
            )
        return (item_type(i) for i in enum)

    @t.overload
    def __getitem__(self, idx: int) -> NamedTupleT:
        ...

    @t.overload
    def __getitem__(self, idx: slice) -> Self:
        ...

    @t.overload
    def __getitem__(self, idx: str) -> abc.Sequence[t.Any]:
        ...

    def __getitem__(
        self, idx: slice | int | str
    ) -> NamedTupleT | Self | abc.Sequence[t.Any]:
        idx_py: int | slice
        if isinstance(idx, str):
            return self._seqs[idx]
        if isinstance(idx, (int, slice)):
            if self.is_empty:
                raise IndexError('No slicing/indexing for an empty segment')
        if isinstance(idx, int):
            if idx == 0:
                raise IndexError(
                    'Segment uses 1-based indexing, 0 is reserved for '
                    'an empty segment'
                )
            idx_py = _translate_idx(idx, self.start)
            it = nth(iter(self), idx_py, None)
            if it is None:
                raise IndexError(
                    f'Index {idx}->{idx_py} lies outside of [0, {len(self)}]'
                )
            return it
            # try:
            #     seqs = {k: v[idx_py : idx_py + 1] for k, v in self._seqs.items()}
            # except IndexError as e:
            #     raise IndexError(f'{idx_py} is not in segment') from e
            # return self.__class__(idx, idx, self.name, seqs)
        if isinstance(idx, slice):
            if idx.start == 0 or idx.stop == 0:
                raise IndexError(
                    'Segment uses 1-based indexing, 0 is reserved for '
                    'an empty segment'
                )
            if idx.step is not None:
                raise IndexError(
                    'Cannot create non-consecutive copy of segment => "step" '
                    'is slicing-incompatible'
                )
            if (idx.start, idx.stop) in [
                (None, None),
                (None, self.end),
                (self.start, None),
                (self.start, self.end),
            ]:
                return deepcopy(self)

            idx_py = _translate_idx(idx, self.start)

            try:
                seqs = {k: v[idx_py.start : idx_py.stop] for k, v in self._seqs.items()}
            except IndexError as e:
                raise IndexError(f'Failed to index sequences using {idx}') from e

            start = idx.start or self.start
            end = idx.stop or self.end

            return self.__class__(start, end, self.name, seqs, parent=self)
        else:
            raise TypeError(f'Cannot index with type {type(idx)}')

    def __setitem__(self, key: str, value: abc.Sequence[t.Any]) -> None:
        self._validate_seq(key, value)
        self._seqs[key] = value

    def __reversed__(self) -> abc.Iterator[NamedTupleT]:
        return always_reversible(iter(self))

    def __contains__(self, item: object) -> bool:
        match item:
            case str():
                return item in self._seqs
            case Ord():
                return self.start <= item <= self.end
            case _:
                return False

    def __len__(self) -> int:
        if self.is_empty:
            return 0
        return self.end - self.start + 1

    def __and__(self, other: Segment) -> Segment:
        return self.overlap_with(other, True, 'self')

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            self.start == other.start
            and self.end == other.end
            and self.meta == other.meta
            and all(it_self == it_other for it_self, it_other in zip(self, other))
        )

    def __hash__(self):
        seqs = tuple(tuple(x) for x in self)
        return hash((self.start, self.end, self.name, seqs, tuple(self.meta.items())))

    def __rshift__(self, idx: int) -> Segment:
        if self.is_empty:
            raise ValueError('Cannot shift an empty segment')
        return Segment(self.start + idx, self.end + idx, self.name, seqs=self._seqs)

    def __lshift__(self, idx: int) -> Segment:
        if self.is_empty:
            raise ValueError('Cannot shift an empty segment')
        return Segment(self.start - idx, self.end - idx, self.name, seqs=self._seqs)

    def _validate_seq(self, name: str, seq: abc.Sequence):
        if not is_valid_field_name(name):
            raise FormatError(
                f'Invalid field name {name}. '
                f'Please use a valid variable name starting with a letter'
            )
        if len(seq) != len(self):
            raise LengthMismatch(
                f"Len({name})={len(seq)} doesn't match the segment's length {len(self)}"
            )

    def _validate_seqs(self):
        for k, seq in self._seqs.items():
            if len(seq) != len(self):
                self._validate_seq(k, seq)

    def _setup_and_validate(self):
        if self.start > self.end or self.start < 0 or self.end < 0:
            raise FormatError(f'Invalid boundaries {self.start}, {self.end}')
        self._validate_seqs()

    def add_seq(self, name: str, seq: abc.Sequence[t.Any]):
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
        return other.is_empty or (other.start >= self.start and self.end >= other.end)

    def bounded_by(self, other: Segment) -> bool:
        """
        Check whether this segment is bounded by other.

        ::

            self:   +----+
            other: +------+
            => True

        :param other; Another segment.
        """
        return self.is_empty or (self.start >= other.start and other.end >= self.end)

    def overlaps(self, other: Segment) -> bool:
        """
        Check whether a segment overlaps with the other segment.
        Use :meth:`overlap_with` to produce an overlapping child
        :class:`Segment`.

        :param other: other :class:`Segment` instance.
        :return: ``True`` if segments overlap and ``False`` otherwise.
        """
        return (self.is_empty or other.is_empty) or not (
            other.start > self.end or self.start > other.end
        )

    def overlap_with(
        self,
        other: Segment,
        deep_copy: bool = True,
        handle_mode: str = 'merge',
        sep: str = '&',
    ) -> Self:
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
            raise NoOverlap(f'Segments {self} and {other} do not overlap')

        if self.is_empty:
            warnings.warn('Overlapping empty & non-empty always results in empty')
            return self

        if other.is_empty:
            warnings.warn('Overlapping non-empty & empty always results in empty')
            return self.__class__(0, 0)

        start, end = max(self.start, other.start), min(self.end, other.end)

        if handle_mode == 'merge':
            meta = {**self.meta, **other.meta}
            seqs = {
                **subset_seqs(self._seqs, self.start, start, end),
                **subset_seqs(other._seqs, other.start, start, end),
            }
            name: str | None = sep.join(map(str, [self.name, other.name]))
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

    def overlap(self, start: int, end: int) -> Self:
        """
        Create new segment from the current instance using overlapping boundaries.

        :param start: Starting coordinate.
        :param end: Ending coordinate.
        :return: New overlapping segment with :attr:`data` and :attr:`name`
        """
        other = self.__class__(start, end)

        if not self.overlaps(other):
            raise NoOverlap

        return self.overlap_with(other, True, 'self')

    def sub_by(self, other: Segment, **kwargs) -> Self:
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

    def sub(self, start: int, end: int, **kwargs) -> Self:
        """
        Subset current segment using provided boundaries.
        Will create a new segment and call :meth:`sub_by`.

        :param start: new start.
        :param end: new end.
        :param kwargs: passed to :meth:`overlap_with`
        """
        return self.sub_by(Segment(start, end), **kwargs)


def _translate_idx(idx: T, offset: int) -> T:
    if isinstance(idx, slice):
        start = idx.start
        stop = idx.stop
        if start is not None:
            start = max(start - offset, 0)
        if stop is not None:
            stop = stop - offset + 1
        # Mypy fails at type narrowing here.
        # See https://github.com/python/mypy/issues/14045
        return t.cast(T, slice(start, stop, idx.step))
    if isinstance(idx, int):
        return t.cast(T, idx - offset)
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
) -> abc.Generator[Segment, None, None]:
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
            sets: abc.Iterable | list = powerset(cc)
            if verbose:
                sets = tqdm(sets, desc=f'Resolving cc {i} with size: {len(cc)}')
            if max_it is not None and max_it > 0:
                sets = take(max_it, sets)
            overlapping_subsets = filterfalse(do_overlap, sets)
            yield from max(overlapping_subsets, key=lambda xs: sum(map(value_fn, xs)))


def map_segment_numbering(
    segments_from: t.Sequence[Segment], segments_to: t.Sequence[Segment]
) -> abc.Iterator[tuple[int, int | None]]:
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
