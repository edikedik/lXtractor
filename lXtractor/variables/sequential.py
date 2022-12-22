"""
Module defines variables calculated on sequences
"""
from __future__ import annotations

import typing as t
from abc import abstractmethod
from collections import abc

from more_itertools import islice_extended

from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.base import SequenceVariable, MappingT, ProtFP, _try_map

T = t.TypeVar('T')
V = t.TypeVar('V')
K = t.TypeVar('K')

_ProtFP = ProtFP()


class SeqEl(SequenceVariable):
    """
    A sequence element variable. It doesn't encompass any calculation.
    Rather, it simply accesses sequence at certain position.

    >>> v1, v2 = SeqEl(1), SeqEl(1, 'X')
    >>> s1, s2 = 'XYZ', [1, 2, 3]
    >>> v1.calculate(s1)
    'X'
    >>> v2.calculate(s2)
    1

    """

    __slots__ = ('p', 'seq_name')

    def __init__(self, p: int, seq_name: str = SeqNames.seq1):
        """
        :param p: Position, starting from 1.
        :param seq_name: The name of the sequence used to distinguish variables
            pointing to the same position.
        """
        #: Position, starting from 1.
        self.p = p
        #: Sequence name for which the element is accessed
        self.seq_name = seq_name

    @property
    def rtype(self) -> t.Type[str]:
        return str

    def calculate(
        self, obj: abc.Sequence[T], mapping: t.Optional[MappingT] = None
    ) -> T:
        p = _try_map(self.p, mapping)
        try:
            return obj[p - 1]
        except IndexError:
            raise FailedCalculation(f'Missing index {p - 1} in sequence')


class PFP(SequenceVariable):
    """
    A ProtFP embedding variable.

    .. seealso::

        :class:`lXtractor.variables.base.ProtFP`
    """

    __slots__ = ('p', 'i')

    def __init__(self, p: int, i: int):
        """

        :param p: Position, starting from 1.
        :param i: A PCA component index starting from 1.
        """
        #: Position, starting from 1
        self.p = p
        #: A PCA component index starting from 1.
        self.i = i

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
        self, obj: abc.Sequence[str], mapping: t.Optional[MappingT] = None
    ) -> float:
        p = _try_map(self.p, mapping)
        try:
            return _ProtFP[(obj[p - 1], self.i)]
        except (KeyError, IndexError) as e:
            raise FailedCalculation(f'Failed to map {p - 1} with ProtFP') from e


class SliceTransformReduce(SequenceVariable, t.Generic[T, V, K]):
    """
    A composite variable with three sequential operations:

        1. Slice -- subset the sequence (optional).
        2. Transform -- transform the sequence (optional).
        3. Reduce -- reduce to a final variable.

    **This is an abstract class.** It requires to define at least two methods:
        1. :meth:`transform`.
        2. :meth:`rtype` property.

    .. seealso::
        :func:`make_str` -- a factory function to quickly make child classes.

    """

    __slots__ = ('start', 'stop', 'step', 'seq_name')

    def __init__(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        seq_name: str = SeqNames.seq1,
    ):
        """
        .. note::
            `start` and `stop` have inclusive boundaries.

        :param start: Start position
        :param stop: Stop position.
        :param step: Slicing step.
        :param seq_name: Sequence name. Please use it in case a resulting
            variable will be applied to seqs other than the primary sequence.
        """
        #: Start position.
        self.start = start
        #: End position.
        self.stop = stop
        #: Slicing step.
        self.step = step
        #: Sequence name.
        self.seq_name = seq_name

    @staticmethod
    @abstractmethod
    def reduce(seq: abc.Iterable[T]) -> V:
        """
        Reduce the input iterable into the variable result.

        :param seq: Some sort of iterable -- the results of the transform
            (or slicing, if no transformation is used)
        :return: An aggregated value  (e.g., float, string, etc.).
        """
        raise NotImplementedError

    @staticmethod
    def transform(seq: abc.Iterator[K]) -> abc.Iterable[T]:
        """
        Optionally transform the slicing result.
        If not used, it is the identity operation.

        :param seq: The result of slicing operation. If no slicing is used,
            it is just an ``iter(input_seq)``.
        :return: Iterable over transformed elements (can have another type than
            the input ones).
        """
        return seq

    def calculate(
        self, obj: abc.Iterable[K], mapping: t.Optional[MappingT] = None
    ) -> V:
        start, stop, step = map(
            lambda x: None if x is None else _try_map(x, mapping),
            [self.start, self.stop, self.step],
        )

        if start is not None:
            start -= 1

        return self.reduce(self.transform(islice_extended(obj, start, stop, step)))


# TODO: isn't compatible with parallel computation because ABC aren't serializable
# monitor https://github.com/uqfoundation/dill/issues/332 the solution in the next
# versions of dill
def make_str(
    reduce: abc.Callable[[abc.Iterable[T]], V],
    rtype: t.Type,
    transform: abc.Callable[[abc.Iterator[K]], abc.Iterable[T]] | None = None,
    reduce_name: str | None = None,
    transform_name: str | None = None,
) -> t.Type[SliceTransformReduce]:
    """
    Makes a non-abstract subclass of :class:`SliceTransformReduce`
    with specific transform and reduce operations.

    To make things clearer, transform and reduce operations will have certain
    names that will be incoroporated into a created class name.

    **Example 1: no transformation:**

    >>> v_type = make_str(sum, float)
    >>> v_type.__name__
    'SliceSum'

    To instanciate it, we provide additional slicing parameters

    >>> v = v_type(1, 2, seq_name='X')
    >>> v.id
    "SliceSum(start=1,stop=2,step=None,seq_name='X')"

    >>> v.calculate([1, 2, 3, 4, 5])
    3

    **Example 2: with transformation:**

    Note that the first operatoiin -- slicing -- inevitably produces an
    iterator over the input sequence. Hence, even if we aren't slicing,
    i.e., provide ``None`` for all :meth:`SliceTransformReduce.__init__`
    arguments, we still obtain an iterator over characters. Therefore,
    we convert it to string and then apply the necessary operation.
    Note that this feature makes transform ``map``-friendly.

    >>> count_x = lambda x: sum(1 for c in x if c == 'X')
    >>> upper = lambda x: "".join(x).upper()
    >>> v = make_str(count_x, int, transform=upper, transform_name='upper',
    ...              reduce_name='countX')()
    >>> v.calculate('XoXoxo')
    3
    >>> v.id
    "SliceUpperCountx(start=None,stop=None,step=None,seq_name='seq1')"

    .. seealso::
        :class:`SliceTransformReduce` -- a base abstract class from which this
        function generates variables.

    :param reduce: Reduce operation peferably producing a single output.
    :param rtype: Return type of the reduce operation and, since this is the
        last operatoin, of a variable itself.
    :param transform: Optional transformation operation. It accepts an iterator
        over (optionally) sliced input elements and returns an iterable over
        elements of potentially another type,
        as long as they are supported by the `reduce`.
    :param reduce_name: The name of the reduce operation.
        Please provide it in case using ``lambda``.
    :param transform_name: The name of the transform operation.
        Please provide it in case using ``lambda``.
    :return: An uninitialized subclass of :class:`SliceTransformReduce`
        encapsulating the provided operations within the
        :meth:`SliceTransformReduce.calculate`.
    """
    d = {'reduce': staticmethod(reduce), 'rtype': property(lambda _: rtype)}

    if transform is None:
        transform_name = ''
    else:
        transform_name = transform_name or transform.__name__
        d['transform'] = staticmethod(transform)

    reduce_name = reduce_name or reduce.__name__

    transform_name, reduce_name = map(
        lambda x: x.capitalize(), [transform_name, reduce_name]
    )

    cls_name = f'Slice{transform_name}{reduce_name}'
    obj = type(cls_name, (SliceTransformReduce,), d)

    return obj


if __name__ == '__main__':
    raise RuntimeError
