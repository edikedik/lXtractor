"""
The module defines the :class:`ChainList` - a list of `Chain*`-type objects that
behaves like a regular list but has additional bells and whistles tailored
towards `Chain*` data structures.
"""
from __future__ import annotations

import operator as op
import typing as t
from collections import abc
from functools import partial
from itertools import chain, zip_longest, tee

import pandas as pd
from more_itertools import nth, peekable

import lXtractor.core.segment as lxs
from lXtractor.core.base import Ord, ApplyT
from lXtractor.core.chain.base import is_chain_type_iterable, is_chain_type
from lXtractor.core.config import MetaNames
from lXtractor.core.exceptions import MissingData
from lXtractor.util import apply

if t.TYPE_CHECKING:
    from lXtractor.core.chain import ChainSequence, ChainStructure, Chain

    CT = t.TypeVar("CT", ChainStructure, ChainSequence, Chain)
    # CT = t.TypeVar('CT', bound=t.Union[ChainSequence, ChainStructure, Chain])
    CS = t.TypeVar("CS", ChainStructure, ChainSequence)
    CTU: t.TypeAlias = ChainSequence | ChainStructure | Chain
else:
    CT = t.TypeVar("CT")

T = t.TypeVar("T")


def add_category(c: t.Any, cat: str):
    """
    :param c: A Chain*-type object.
    :param cat: Category name.
    :return:
    """

    if hasattr(c, "meta"):
        meta = c.meta
    else:
        raise TypeError(f"Failed to find .meta attr in {c}")

    field = MetaNames.category
    if field not in meta:
        meta[field] = cat
    else:
        existing = meta[field].split(",")
        if cat not in existing:
            meta[field] += f",{cat}"


def _check_chain_types(objs: abc.Sequence[T]):
    if not is_chain_type_iterable(objs):
        raise TypeError("A sequence of objects is not a Chain*-type sequence")


class ChainList(abc.MutableSequence[CT]):
    # TODO: consider implementing pattern-based search over whole sequence
    # or a sequence region.

    # For the above, consider filtering to hits.
    # It may be beneficial to implement this functionality for ChainSequence.
    """
    A mutable single-type collection holding either :class:`Chain`'s,
    or :class:`ChainSequence`'s, or :class:`ChainStructure`'s.

    Object's funtionality relies on this type purity.
    Adding of / contatenating with objects of a different type shall
    raise an error.

    It behaves like a regular list with additional functionality.

    >>> from lXtractor.core.chain.sequence import ChainSequence
    >>> s = ChainSequence.from_string('SEQUENCE', name='S')
    >>> x = ChainSequence.from_string('XXX', name='X')
    >>> x.meta['category'] = 'x'
    >>> cl = ChainList([s, s, x])
    >>> cl
    [S|1-8, S|1-8, X|1-3]
    >>> cl[0]
    S|1-8
    >>> cl['S']
    [S|1-8, S|1-8]
    >>> cl[:2]
    [S|1-8, S|1-8]
    >>> cl['1-3']
    [X|1-3]

    Adding/appending/removing objects of a similar type is easy and works
    similar to a regular list.

    >>> cl += [s]
    >>> assert len(cl) == 4
    >>> cl.remove(s)
    >>> assert len(cl) == 3

    Categories can be accessed as attributes or using ``[]`` syntax
    (similar to the `Pandas.DataFrame` columns).

    >>> cl.x
    [X|1-3]
    >>> cl['x']
    [X|1-3]

    While creating a chain list, using a `groups` parameter will assign
    categories to sequences.
    Note that such operations return a new :class:`ChainList` object.

    >>> cl = ChainList([s, x], categories=['S', ['X1', 'X2']])
    >>> cl.S
    [S|1-8]
    >>> cl.X2
    [X|1-3]
    >>> cl['X1']
    [X|1-3]

    """
    __slots__ = ("_chains",)

    def __init__(
        self,
        chains: abc.Iterable[CT],
        categories: abc.Iterable[str | abc.Iterable[str]] | None = None,
    ):
        """
        :param chains: An iterable over ``Chain*``-type objects.
        :param categories: An optional list of categories.
            If provided, they will be assigned to inputs' `meta` attributes.
        """

        if not isinstance(chains, list):
            chains = list(chains)

        _check_chain_types(chains)

        if categories is not None:
            for c, cat in zip(chains, categories, strict=True):
                if isinstance(cat, str):
                    add_category(c, cat)
                else:
                    for _cat in cat:
                        add_category(c, _cat)

        #: Protected container. One should NOT change it directly.
        self._chains: list[CT] = chains

    @property
    def categories(self) -> abc.Set[str]:
        """
        :return: A set of categories inferred from `meta` of encompassed
            objects.
        """
        return set(chain.from_iterable(map(lambda c: c.categories, self)))

    def __len__(self) -> int:
        return len(self._chains)

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, ChainList):
            return False
        if isinstance(other, abc.Sized):
            if len(self) != len(other):
                return False
            return all(o1 == o2 for o1, o2 in zip(self, other))
        return False

    @t.overload
    def __getitem__(self, index: int) -> CT:
        ...

    @t.overload
    def __getitem__(self, index: slice) -> ChainList[CT]:
        ...

    @t.overload
    def __getitem__(self, index: str) -> ChainList[CT]:
        ...

    def __getitem__(self, index: t.SupportsIndex | slice | str) -> CT | ChainList[CT]:
        match index:
            case int():
                return self._chains.__getitem__(index)
            case slice():
                return ChainList(self._chains[index])
            case str():
                if index in self.categories:
                    return self.filter_category(index)
                return self.filter(lambda x: index in x.id)  # type: ignore
            case _:
                raise TypeError(f"Incorrect index type {type(index)}")

    def __getattr__(self, name: str):
        """
        See the example in pandas:

        https://github.com/pandas-dev/pandas/blob/
        61e0db25f5982063ba7bab062074d55d5e549586/pandas/core/generic.py#L5811
        """

        if name == "__setstate__":
            raise AttributeError(name)

        if name.startswith("__"):
            object.__getattribute__(self, name)

        if name in self.categories:
            return self.filter(lambda c: any(cat == name for cat in c.categories))

        raise AttributeError

    @t.overload
    def __setitem__(self, index: int, value: CT):
        ...

    @t.overload
    def __setitem__(self, index: slice, value: abc.Iterable[CT]):
        ...

    def __setitem__(self, index: t.SupportsIndex | slice, value: CT | abc.Iterable[CT]):
        if len(self) == 0:
            raise MissingData("Not possible to use __setitem__ when ChainList is empty")
        self_type = type(self._chains[0])
        if is_chain_type(value):
            other_type = type(value)
        else:
            if is_chain_type_iterable(value):
                # Doesn't accept unions for some reason
                value = peekable(value)  # type: ignore
                other_type = type(value.peek())
            else:
                raise TypeError("Incompatible value type")

        if self_type is other_type or id(self_type) == id(other_type):
            self._chains[index] = value  # type: ignore  # overloading failure
        else:
            raise TypeError(
                f"Value type {other_type} conflicts with existing "
                f"items type {self_type}"
            )

    def __delitem__(self, index: t.SupportsIndex | int | slice):
        self._chains.__delitem__(index)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            for c in self:
                if c.id == item:
                    return True
            return False
        return item in self._chains

    def __add__(self, other: ChainList | abc.Iterable):
        match other:
            case ChainList():
                if len(self._chains) > 0:
                    _check_chain_types([self._chains[0], *other])
                return ChainList(self._chains + other._chains)
            case abc.Iterable():
                if len(self._chains) > 0:
                    other = list(other)
                    _check_chain_types([self._chains[0], *other])
                return ChainList(self._chains + list(other))
            case _:
                raise TypeError(f"Unsupported type {type(other)}")

    def __repr__(self) -> str:
        return self._chains.__repr__()

    def __iter__(self) -> abc.Iterator[CT]:
        return iter(self._chains)

    def index(self, value: CT, start: int = 0, stop: int | None = None) -> int:
        stop = stop or len(self)
        return self._chains.index(value, start, stop)

    def insert(self, index: int, value: CT):
        if len(self) > 0:
            _check_chain_types([self[0], value])
        self._chains.insert(index, value)

    def iter_children(self) -> abc.Generator[ChainList[CT], None, None]:
        """
        Simultaneously iterate over topological levels of children.

        >>> from lXtractor.core.chain.sequence import ChainSequence
        >>> s = ChainSequence.from_string('ABCDE', name='A')
        >>> child1 = s.spawn_child(1, 4)
        >>> child2 = child1.spawn_child(2, 3)
        >>> x = ChainSequence.from_string('XXXX', name='X')
        >>> child3 = x.spawn_child(1, 3)
        >>> cl = ChainList([s, x])
        >>> list(cl.iter_children())
        [[A|1-4<-(A|1-5), X|1-3<-(X|1-4)], [A|2-3<-(A|1-4<-(A|1-5))]]

        :return: An iterator over chain lists of children levels.
        """
        # Mypy thinks zip_longest produces tuples of `object` types
        # probably due to "*"
        yield from map(
            lambda xs: ChainList(chain.from_iterable(xs)),  # type: ignore
            zip_longest(*map(lambda c: c.iter_children(), self._chains), fillvalue=[]),
        )

    def get_level(self, n: int) -> ChainList[CT]:
        """
        Get a specific level of a hierarchical tree starting from this list::

            l0: this list
            l1: children of each child of each object in l0
            l2: children of each child of each object in l1
            ...

        :param n: The level index (0 indicates this list).
            Other levels are obtained via :meth:`iter_children`.
        :return: A chain list of object corresponding to a specific topological
            level of a child tree.
        """
        if n == 0:
            return self
        return nth(self.iter_children(), n - 1, default=ChainList([]))

    def collapse_children(self) -> ChainList[CT]:
        """
        Collapse all children of each object in this list into a single
        chain list.

        >>> from lXtractor.core.chain.sequence import ChainSequence
        >>> s = ChainSequence.from_string('ABCDE', name='A')
        >>> child1 = s.spawn_child(1, 4)
        >>> child2 = child1.spawn_child(2, 3)
        >>> cl = ChainList([s]).collapse_children()
        >>> assert isinstance(cl, ChainList)
        >>> cl
        [A|1-4<-(A|1-5), A|2-3<-(A|1-4<-(A|1-5))]

        :return: A chain list of all children.

        """
        return ChainList(chain.from_iterable(self.iter_children()))

    def collapse(self) -> ChainList[CT]:
        """
        Collapse all objects and their children within this list into a new
        chain list. This is a shortcut for
        ``chain_list + chain_list.collapse_children()``.

        :return: Collapsed list.
        """
        return self + self.collapse_children()

    def iter_sequences(self) -> abc.Generator[ChainSequence, None, None]:
        """
        :return: An iterator over :class:`ChainSequence`'s.
        """
        # mypy doesn't know the type is known at runtime
        from lXtractor.core import chain as lxc

        if len(self) > 0:
            x = self[0]
            if isinstance(x, (lxc.chain.Chain, lxc.structure.ChainStructure)):
                yield from (c.seq for c in self._chains)
            else:
                yield from iter(self._chains)
        else:
            yield from iter([])

    def iter_structures(self) -> abc.Generator[ChainStructure, None, None]:
        """
        :return: An generator over :class:`ChainStructure`'s.
        """
        # mypy doesn't know the type is known at runtime
        from lXtractor.core import chain as lxc

        if len(self) > 0:
            x = self[0]
            if isinstance(x, lxc.Chain):
                yield from chain.from_iterable(c.structures for c in self._chains)
            elif isinstance(x, lxc.ChainStructure):
                yield from iter(self._chains)
            else:
                yield from iter([])
        else:
            yield from iter([])

    def iter_structure_sequences(self) -> abc.Generator[ChainSequence, None, None]:
        """
        :return: Iterate over :attr:`ChainStructure.seq` attributes.
        """
        yield from (s.seq for s in self.iter_structures())

    @property
    def sequences(self) -> ChainList[ChainSequence]:
        """
        :return: Get all :attr:`lXtractor.core.chain.Chain.seq` or
            `lXtractor.core.chain.sequence.ChainSequence` objects within this
            chain list.
        """
        return ChainList(self.iter_sequences())

    @property
    def structures(self) -> ChainList[ChainStructure]:
        return ChainList(self.iter_structures())

    @property
    def structure_sequences(self) -> ChainList[ChainSequence]:
        return ChainList(self.iter_structure_sequences())

    @staticmethod
    def _get_seg_matcher(
        s: str,
    ) -> abc.Callable[[ChainSequence, lxs.Segment, t.Optional[str]], bool]:
        def matcher(
            seq: ChainSequence, seg: lxs.Segment, map_name: t.Optional[str] = None
        ) -> bool:
            if map_name is not None:
                # Get elements in the seq whose mapped sequence matches
                # seg boundaries
                start_item = seq.get_closest(map_name, seg.start)
                end_item = seq.get_closest(map_name, seg.end, reverse=True)
                if start_item is None or end_item is None:
                    return False

                start = start_item._asdict()[map_name]
                end = end_item._asdict()[map_name]
                # If not such elements -> no match

                # Create a new temporary segment using the mapped boundaries
                _seq: lxs.Segment | ChainSequence = lxs.Segment(start, end)
            else:
                _seq = seq
            match s:
                case "overlap":
                    return _seq.overlaps(seg)
                case "bounded":
                    return _seq.bounded_by(seg)
                case "bounding":
                    return _seq.bounds(seg)
                case _:
                    raise ValueError(f"Invalid matching mode {s}")

        return matcher

    @staticmethod
    def _get_pos_matcher(
        ps: abc.Iterable[Ord],
    ) -> abc.Callable[[ChainSequence, t.Optional[str]], bool]:
        def matcher(seq: ChainSequence, map_name: t.Optional[str] = None) -> bool:
            obj: abc.Sequence | ChainSequence = seq
            if map_name:
                obj = seq[map_name]
            return all(p in obj for p in ps)

        return matcher

    def _filter_seqs(
        self,
        seqs: abc.Iterable[ChainSequence],
        match_type: str,
        s: lxs.Segment | abc.Iterable[Ord],
        map_name: t.Optional[str],
    ) -> abc.Iterator[bool]:
        if isinstance(s, lxs.Segment):
            match_fn = partial(
                self._get_seg_matcher(match_type), seg=s, map_name=map_name
            )
        else:
            match_fn = partial(self._get_pos_matcher(s), map_name=map_name)

        return map(match_fn, seqs)

    def _filter_str(
        self,
        structures: abc.Iterable[ChainStructure],
        match_type: str,
        s: lxs.Segment | abc.Collection[Ord],
        map_name: t.Optional[str],
    ) -> abc.Iterator[bool]:
        return self._filter_seqs(
            map(lambda x: x.seq, structures), match_type, s, map_name
        )

    def filter_pos(
        self,
        s: lxs.Segment | abc.Collection[Ord],
        *,
        match_type: str = "overlap",
        map_name: str | None = None,
    ) -> ChainList[CS]:
        """
        Filter to objects encompassing certain consecutive position regions
        or arbitrary positions' collections.

        For :class:`Chain` and :class:`ChainStructure`, the filtering is over
        `seq` attributes.

        :param s: What to search for:

            #. ``s=Segment(start, end)`` to find all objects encompassing
                certain region.
            #. ``[pos1, posX, posN]`` to find all objects encompassing the
                specified positions.

        :param match_type: If `s` is `Segment`, this value determines the
            acceptable relationships between `s` and each
            :class:`ChainSequence`:

                #. "overlap" -- it's enough to overlap with `s`.
                #. "bounding" -- object is accepted if it bounds `s`.
                #. "bounded" -- object is accepted if it's bounded by `s`.

        :param map_name: Use this map within to map positions of `s`.
            For instance, to each for all elements encompassing region 1-5 of
            a canonical sequence, one would use

                .. code-block:: python

                    chain_list.filter_pos(
                        s=Segment(1, 5), match_type="bounding",
                        map_name="map_canonical"
                    )

        :return: A list of hits of the same type.
        """

        from lXtractor.core import chain as lxc

        if len(self) > 0:
            x = self[0]
            if isinstance(x, lxc.Chain):
                objs, fn = self.iter_sequences(), self._filter_seqs
            elif isinstance(x, lxc.ChainSequence):
                objs, fn = iter(self), self._filter_seqs
            else:
                objs, fn = iter(self), self._filter_str
        else:
            return ChainList([])

        objs1, objs2 = tee(objs)
        mask = fn(objs1, match_type, s, map_name)

        return ChainList(
            map(op.itemgetter(1), filter(lambda x: x[0], zip(mask, objs2)))
        )

    def filter(self, pred: abc.Callable[[CT], bool]) -> ChainList[CT]:
        """
        >>> from lXtractor.core.chain.sequence import ChainSequence
        >>> cl = ChainList(
        ...     [ChainSequence.from_string('AAAX', name='A'),
        ...      ChainSequence.from_string('XXX', name='X')]
        ... )
        >>> cl.filter(lambda c: c.seq1[0] == 'A')
        [A|1-4]

        :param pred: Predicate callable for filtering.
        :return: A filtered chain list (new object).
        """
        return ChainList(filter(pred, self))

    def filter_category(self, name: str) -> ChainList:
        """
        :param name: Category name.
        :return: Filtered objects having this category within their
            ``meta["category"]``.
        """
        return self.filter(lambda c: any(cat == name for cat in c.categories))

    def apply(
        self,
        fn: ApplyT,
        verbose: bool = False,
        desc: str = "Applying to objects",
        num_proc: int = 1,
    ) -> ChainList[CT]:
        """
        Apply a function to each object and return a new chain list of results.

        :param fn: A callable to apply.
        :param verbose: Display progress bar.
        :param desc: Progress bar description.
        :param num_proc: The number of CPUs to use. ``num_proc <= 1`` indicates
            sequential processing.
        :return: A new chain list with application results.
        """
        return ChainList(apply(fn, self._chains, verbose, desc, num_proc))

    def summary(self, **kwargs) -> pd.DataFrame:
        return pd.concat([c.summary(**kwargs) for c in self])


def _wrap_children(children: abc.Iterable[CT] | None) -> ChainList[CT]:
    if children:
        if not isinstance(children, ChainList):
            assert is_chain_type_iterable(children)
            return ChainList(children)
        return children
    return ChainList([])


if __name__ == "__main__":
    raise RuntimeError
