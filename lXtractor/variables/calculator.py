"""
Module defining variable calculators managing the exact calculation process
of variables on objects.
"""
from __future__ import annotations

import typing as t
from collections import abc
from concurrent.futures import ProcessPoolExecutor
from itertools import starmap, repeat

from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.base import AbstractCalculator, OT, RT, VT, MappingT

ERT: t.TypeAlias = tuple[bool, t.Union[RT, str]]  # extended return type


def _try_calculate(
    o: OT, v: VT, m: MappingT | None, valid_exceptions: abc.Sequence[t.Type[Exception]]
) -> ERT:
    try:
        return True, v.calculate(o, m)
    except valid_exceptions as e:
        return False, str(e)


def _calc_on_object(
    o: OT,
    vs: abc.Iterable[VT],
    m: MappingT | None,
    valid_exceptions: abc.Sequence[t.Type[Exception]],
):
    return [_try_calculate(o, v, m, valid_exceptions) for v in vs]


class SimpleCalculator(AbstractCalculator[OT, VT, RT]):
    """
    Uses straightforward calls of :meth:`calculate` method, i.e.,
    `variable.calculate(obj, mapping)`.
    Returns the extended calculation result `(is_calculated, res)` where
    `is_calculated` is `True` when the calculation completed successfully,
    and `False` otherwise. In turn, `res` holds the calculation result,
    which constitutes an error message if calculation failed.

    The calculation has gracefully failed iff class:`lXtractor.core.exceptions.
    FailedCalculation` is raised.
    """

    __slots__ = ('valid_exceptions',)

    def __init__(
        self, valid_exceptions: abc.Sequence[t.Type[Exception]] = (FailedCalculation,)
    ):
        self.valid_exceptions = valid_exceptions

    def __call__(self, o: OT, v: VT, m: MappingT | None) -> ERT:
        return _try_calculate(o, v, m, self.valid_exceptions)

    def map(self, o: OT, v: abc.Iterable[VT], m: MappingT | None) -> abc.Iterator[ERT]:
        return map(lambda _v: _try_calculate(o, _v, m, self.valid_exceptions), v)

    def vmap(
        self, o: abc.Iterable[OT], v: VT, m: abc.Iterable[MappingT | None] | None
    ) -> abc.Iterator[ERT]:
        if not isinstance(m, abc.Iterable):
            m = repeat(m)
        return starmap(
            lambda _o, _m: _try_calculate(_o, v, _m, self.valid_exceptions), zip(o, m)
        )


class ParallelCalculator(AbstractCalculator):
    """
    Parallel calculator, calculating variables in parallel. Duh.
    """

    __slots__ = ('num_proc', 'verbose', 'valid_exceptions')

    def __init__(
        self,
        num_proc: int | None,
        verbose: bool = True,
        valid_exceptions: abc.Sequence[t.Type[Exception]] = (FailedCalculation,),
    ):
        self.num_proc = num_proc
        self.verbose = verbose
        self.valid_exceptions = valid_exceptions

    def __call__(
        self,
        o: abc.Iterable[OT],
        v: abc.Iterable[abc.Iterable[VT]],
        m: abc.Iterable[MappingT | None] | None,
    ) -> abc.Iterator[list[ERT]]:
        if isinstance(m, dict) or m is None:
            m = repeat(m)
        with ProcessPoolExecutor(self.num_proc) as executor:
            results = executor.map(
                _calc_on_object, o, v, m, repeat(self.valid_exceptions)
            )
            yield from results

    def map(self, o: OT, v: abc.Iterable[VT], m: MappingT | None) -> abc.Iterator[RT]:
        with ProcessPoolExecutor(self.num_proc) as executor:
            results = executor.map(
                _try_calculate, repeat(o), v, repeat(m), repeat(self.valid_exceptions)
            )
            yield from results

    def vmap(
        self, o: abc.Iterable[OT], v: VT, m: abc.Iterable[MappingT | None]
    ) -> abc.Iterator[RT]:
        if not isinstance(m, abc.Iterable):
            m = repeat(m)
        with ProcessPoolExecutor(self.num_proc) as executor:
            results = executor.map(
                _try_calculate, o, repeat(v), m, repeat(self.valid_exceptions)
            )
            yield from results


if __name__ == '__main__':
    raise RuntimeError
