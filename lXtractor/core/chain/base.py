from __future__ import annotations
import typing as t

from collections import abc
from itertools import chain, tee

from lXtractor.util.typing import is_iterable_of, is_type

if t.TYPE_CHECKING:
    from lXtractor.core.chain import ChainSequence, ChainStructure, Chain

    # CT = t.TypeVar('CT', bound=t.Union[ChainSequence, ChainStructure, Chain])
    CT = t.TypeVar('CT', ChainSequence, ChainStructure, Chain)
    CTU: t.TypeAlias = ChainSequence | ChainStructure | Chain
else:
    CT = t.TypeVar('CT')

T = t.TypeVar('T')


__all__ = ('topo_iter', 'is_chain_type', 'is_chain_type_iterable')


def topo_iter(
    start_obj: T, iterator: abc.Callable[[T], abc.Iterable[T]]
) -> abc.Generator[list[T], None, None]:
    """
    Iterate over sequences in topological order.

    >>> n = 1
    >>> it = topo_iter(n, lambda x: (x + 1 for n in range(x)))
    >>> next(it)
    [2]
    >>> next(it)
    [3, 3]

    :param start_obj: Starting object.
    :param iterator: A callable accepting a single argument of the same type as
        the `start_obj` and returning an iterator over objects with the same
        type, representing the next level.
    :return: A generator yielding lists of objects obtained using `iterator`
        and representing topological levels with the root in `start_obj`.
    """

    def get_level(objs: abc.Iterable[T]) -> abc.Iterable[T]:
        return chain.from_iterable(map(iterator, objs))

    curr_level = list(iterator(start_obj))

    while True:
        yield curr_level
        curr_level = list(get_level(curr_level))
        if not curr_level:
            return


def is_chain_type_iterable(
    s: t.Any,
) -> t.TypeGuard[
    abc.Iterable[Chain] | abc.Iterable[ChainSequence] | abc.Iterable[ChainStructure]
]:
    from lXtractor.core import chain as lxc

    if not isinstance(s, abc.Iterable):
        return False

    ss = tee(s, 3)

    return any(
        is_iterable_of(_s, _t)
        for _t, _s in zip([lxc.ChainSequence, lxc.ChainStructure, lxc.Chain], ss)
    )


def is_chain_type(s: t.Any) -> t.TypeGuard[CTU]:
    from lXtractor.core import chain as lxc

    return any(
        is_type(s, _t) for _t in [lxc.ChainSequence, lxc.ChainStructure, lxc.Chain]
    )


if __name__ == '__main__':
    raise ValueError
