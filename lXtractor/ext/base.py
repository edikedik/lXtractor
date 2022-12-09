import typing as t
from collections import abc

from lXtractor.core.chain import CT


@t.runtime_checkable
class SupportsAnnotate(t.Protocol[CT]):
    """
    A class that serves as basis for annotators -- callables accepting a `Chain*`-type
    object and returning a single or multiple objects derived from an initial `Chain*`,
    e.g., via :meth:`spawn_child <lXtractor.core.chain.Chain.spawn_child`.
    """

    def annotate(self, c: CT, *args, keep: bool = True, **kwargs) -> CT | abc.Iterable[CT]: ...


if __name__ == '__main__':
    raise RuntimeError
