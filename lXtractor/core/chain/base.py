from __future__ import annotations
import typing as t

from collections import abc
from itertools import chain

if t.TYPE_CHECKING:
    from lXtractor.core.chain.sequence import ChainSequence
    from lXtractor.core.chain.structure import ChainStructure
    from lXtractor.core.chain.chain import Chain

    CT = t.TypeVar('CT', ChainStructure, ChainSequence, Chain)

T = t.TypeVar('T')


def topo_iter(
    start_obj: T, iterator: abc.Callable[[T], abc.Iterator[T]]
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

    def get_level(objs: abc.Iterable[T]) -> abc.Iterator[T]:
        return chain.from_iterable(map(iterator, objs))

    curr_level = list(iterator(start_obj))

    while True:
        yield curr_level
        curr_level = list(get_level(curr_level))
        if not curr_level:
            return


if __name__ == '__main__':
    raise ValueError
