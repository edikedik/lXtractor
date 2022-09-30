from __future__ import annotations

import typing as t
from collections import abc
from itertools import starmap

from more_itertools import zip_equal

from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.base import AbstractCalculator, OT, RT, VT, MappingT


ERT: t.TypeAlias = tuple[bool, RT | str]


def _try_calculate(
        o: OT, v: VT, m: MappingT | None,
        valid_exceptions: abc.Sequence[Exception] = (FailedCalculation, ),
) -> ERT:
    try:
        return True, v.calculate(o, m)
    except valid_exceptions as e:
        return False, str(e)


class SimpleCalculator(AbstractCalculator):
    """
    Uses straightforward calls of :meth:`calculate` method, i.e., `variable.calculate(obj, mapping)`.
    Returns the extended calculation result `(is_calculated, res)` where `is_calculated` is `True` when
    the calculation completed successfully, and `False` otherwise.
    In turn, `res` holds the calculation result, which constitutes an error message if calculation failed.

    The calculation has gracefully failed iff class:`lXtractor.core.exceptions.FailedCalculation` is raised.
    """

    __slots__ = ()

    def __call__(self, o: OT, v: VT, m: MappingT | None) -> ERT:
        return _try_calculate(o, v, m)

    def map(self, o: OT, v: abc.Iterable[VT], m: MappingT | None) -> abc.Iterator[ERT]:
        return map(lambda _v: _try_calculate(o, _v, m), v)

    def vmap(
            self, o: abc.Iterable[OT], v: VT, m: abc.Iterable[MappingT | None]
    ) -> abc.Iterator[ERT]:
        return starmap(lambda _o, _m: _try_calculate(_o, v, _m), zip_equal(o, m))


if __name__ == '__main__':
    raise RuntimeError
