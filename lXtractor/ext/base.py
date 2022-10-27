import typing as t
from collections import abc

from lXtractor.core.chain import CT


@t.runtime_checkable
class SupportsAnnotate(t.Protocol[CT]):

    def annotate(self, c: CT, *args, keep: bool = True, **kwargs) -> CT | abc.Iterable[CT]: ...


if __name__ == '__main__':
    raise RuntimeError
