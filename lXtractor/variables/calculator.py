"""
Module defining variable calculators managing the exact calculation process
of variables on objects.
"""
from __future__ import annotations

import typing as t
from collections import abc
from concurrent.futures import ProcessPoolExecutor
from itertools import starmap, repeat

from more_itertools import peekable

from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.base import (
    AbstractCalculator,
    ERT,
    OT,
    VT,
    MappingT,
    AbstractVariable,
)

M: t.TypeAlias = MappingT | abc.Iterable[MappingT | None] | None


def _try_calculate(
    o: OT,
    v: VT,
    m: MappingT | None,
    valid_exceptions: abc.Sequence[t.Type[Exception]] | None,
) -> ERT:
    if valid_exceptions is not None:
        try:
            return True, v.calculate(o, m)
        except tuple(valid_exceptions) as e:
            return False, str(e)
    return True, v.calculate(o, m)


def _calc_on_object(
    o: OT,
    vs: abc.Iterable[VT],
    m: MappingT | None,
    valid_exceptions: abc.Sequence[t.Type[Exception]] | None,
) -> abc.Iterator[ERT]:
    return (_try_calculate(o, v, m, valid_exceptions) for v in vs)


def calculate(
    o: abc.Iterable[OT],
    v: abc.Iterable[VT] | abc.Iterable[abc.Iterable[VT]],
    m: MappingT | abc.Iterable[MappingT | None] | None,
    valid_exceptions: abc.Sequence[t.Type[Exception]] | None,
    num_proc: int | None,
) -> abc.Generator[abc.Iterator[ERT], None, None]:
    # Unpack o to know the size
    o = list(o)

    # Check variables type
    v = peekable(v)
    fst = v.peek(None)
    if isinstance(fst, AbstractVariable):
        v = repeat(list(v), len(o))

    if m is None:
        m = repeat(None, len(o))
    elif isinstance(m, abc.Mapping):
        m = repeat(m, len(o))
    if num_proc is None:
        yield from (
            _calc_on_object(*xs, valid_exceptions) for xs in zip(o, v, m, strict=True)
        )
    else:
        with ProcessPoolExecutor(num_proc) as executor:
            results = executor.map(_calc_on_object, o, v, m, repeat(valid_exceptions))
            yield from results


class GenericCalculator(AbstractCalculator):
    """
    Parallel calculator, calculating variables in parallel. Duh.
    """

    __slots__ = ('num_proc', 'valid_exceptions')

    def __init__(
        self,
        num_proc: int | None = None,
        valid_exceptions: abc.Sequence[t.Type[Exception]] = (FailedCalculation,),
    ):
        self.num_proc = num_proc
        self.valid_exceptions = valid_exceptions

    @t.overload
    def __call__(self, o: OT, v: VT, m: MappingT | None) -> ERT:
        ...

    @t.overload
    def __call__(
        self,
        o: abc.Iterable[OT],
        v: abc.Iterable[VT] | abc.Iterable[abc.Iterable[VT]],
        m: abc.Iterable[MappingT | None] | None,
    ) -> abc.Iterable[abc.Iterable[ERT]]:
        ...

    def __call__(
        self,
        o: OT | abc.Iterable[OT],
        v: VT | abc.Iterable[VT] | abc.Iterable[abc.Iterable[VT]],
        m: MappingT | abc.Iterable[MappingT | None] | None,
    ) -> ERT | abc.Iterable[abc.Iterable[ERT]]:
        if isinstance(v, abc.Iterable):
            return calculate(o, v, m, self.valid_exceptions, self.num_proc)
        assert (
            isinstance(m, abc.Mapping) or m is None
        ), "Check if the mapping is a single obj"
        return _try_calculate(o, v, m, self.valid_exceptions)

    def map(
        self, o: OT, v: abc.Iterable[VT], m: MappingT | None
    ) -> abc.Generator[ERT, None, None]:
        if self.num_proc is None:
            yield from _calc_on_object(o, v, m, self.valid_exceptions)
        else:
            with ProcessPoolExecutor(self.num_proc) as executor:
                yield from executor.map(
                    _try_calculate,
                    repeat(o),
                    v,
                    repeat(m),
                    repeat(self.valid_exceptions),
                )

    def vmap(
        self,
        o: abc.Iterable[OT],
        v: VT,
        m: abc.Iterable[MappingT | None] | MappingT | None,
    ) -> abc.Generator[ERT, None, None]:
        if m is None or isinstance(m, abc.Mapping):
            m = repeat(m)
        if self.num_proc is None:
            yield from starmap(
                lambda _o, _m: _try_calculate(_o, v, _m, self.valid_exceptions),
                zip(o, m),
            )
        else:
            with ProcessPoolExecutor(self.num_proc) as executor:
                results = executor.map(
                    _try_calculate, o, repeat(v), m, repeat(self.valid_exceptions)
                )
                yield from results


if __name__ == '__main__':
    raise RuntimeError
