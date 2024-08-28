"""
Module defining variable calculators managing the exact calculation process
of variables on objects.
"""
from __future__ import annotations

import typing as t
from collections import abc
from itertools import repeat

from more_itertools import peekable
from toolz import curry

import lXtractor.util as util
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
METHODS = ("joblib", "multiprocess")


# TODO: docs


@curry
def _try_calculate(
    inp: tuple[OT, VT, MappingT | None],
    valid_exceptions: abc.Sequence[t.Type[Exception]] | None,
) -> ERT:
    o, v, m = inp
    if valid_exceptions is not None:
        try:
            return True, v.calculate(o, m)
        except tuple(valid_exceptions) as e:
            return False, str(e)
    return True, v.calculate(o, m)


@curry
def _calc_on_object(
    inp: tuple[OT, abc.Iterable[VT], MappingT | None],
    valid_exceptions: abc.Sequence[t.Type[Exception]] | None,
) -> list[ERT]:
    o, vs, m = inp
    return [_try_calculate((o, v, m), valid_exceptions) for v in vs]


def calculate(
    o: abc.Iterable[OT],
    v: abc.Iterable[VT] | abc.Iterable[abc.Iterable[VT]],
    m: MappingT | abc.Iterable[MappingT | None] | None,
    valid_exceptions: abc.Sequence[t.Type[Exception]] | None,
    num_proc: int,
    verbose: bool = False,
    **kwargs,
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

    inputs = zip(o, v, m, strict=True)
    fn = _calc_on_object(valid_exceptions=valid_exceptions)
    yield from util.apply(fn, inputs, verbose, "Calculating variables", num_proc, **kwargs)


class GenericCalculator(AbstractCalculator):
    """
    Parallel calculator, calculating variables in parallel. Duh.
    """

    __slots__ = ("num_proc", "valid_exceptions", "apply_kwargs", "verbose")

    def __init__(
        self,
        num_proc: int = 1,
        valid_exceptions: abc.Sequence[t.Type[Exception]] = (FailedCalculation,),
        apply_kwargs: dict | None = None,
        verbose: bool = False,
    ):
        self.num_proc = num_proc
        self.valid_exceptions = valid_exceptions
        self.apply_kwargs = apply_kwargs or {}
        self.verbose = verbose

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
            return calculate(
                o,
                v,
                m,
                self.valid_exceptions,
                self.num_proc,
                self.verbose,
                **self.apply_kwargs,
            )
        assert (
            isinstance(m, abc.Mapping) or m is None
        ), "Mapping must be a single object"

        return _try_calculate((o, v, m), self.valid_exceptions)

    def map(
        self, o: OT, v: abc.Iterable[VT], m: MappingT | None
    ) -> abc.Generator[ERT, None, None]:
        v = list(v)
        inputs = zip(
            repeat(o, len(v)),
            v,
            repeat(m, len(v)),
            strict=True,
        )
        fn = _try_calculate(valid_exceptions=self.valid_exceptions)
        yield from util.apply(
            fn, inputs, self.verbose, "Calculating variables", self.num_proc
        )

    def vmap(
        self,
        o: abc.Iterable[OT],
        v: VT,
        m: abc.Iterable[MappingT | None] | MappingT | None,
    ) -> abc.Generator[ERT, None, None]:
        o = list(o)
        if m is None or isinstance(m, abc.Mapping):
            m = repeat(m, len(o))
        inputs = zip(o, repeat(v, len(o)), m)
        fn = _try_calculate(valid_exceptions=self.valid_exceptions)
        yield from util.apply(
            fn, inputs, self.verbose, "Calculating variables", self.num_proc
        )


if __name__ == "__main__":
    raise RuntimeError
