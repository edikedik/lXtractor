"""
Base classes, common types and functions for the `variables` module.
"""
from __future__ import annotations

import inspect
import logging
import typing as t
from abc import abstractmethod, ABCMeta
from collections import UserDict, abc
from pathlib import Path

import biotite.structure as bst
import numpy as np
import numpy.typing as npt
import pandas as pd
from toolz import curry
from typing_extensions import reveal_type

from lXtractor.core.exceptions import FailedCalculation
from lXtractor.ext import resources
from lXtractor.util.io import read_n_col_table


class AggFn(t.Protocol):
    def __call__(self, a: npt.ArrayLike, **kwargs) -> np.ndarray | float:
        ...


AggFns: dict[str, AggFn] = dict(
    min=np.min, max=np.max, mean=np.mean, median=np.median  # type: ignore
)

LOGGER = logging.getLogger(__name__)
T = t.TypeVar('T')
V = t.TypeVar('V')

MappingT: t.TypeAlias = abc.Mapping[int, t.Optional[int]]
OT = t.TypeVar('OT', bst.AtomArray, abc.Sequence)  # object type
RT = t.TypeVar('RT')  # return type
ERT: t.TypeAlias = tuple[bool, RT]  # extended return type


class AbstractVariable(t.Generic[OT, RT], metaclass=ABCMeta):
    """
    Abstract base class for variables.
    """

    __slots__ = ()

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return not isinstance(other, type(self)) or self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def id(self) -> str:
        """
        Variable identifier such that eval(x.id) produces another instance.
        """

        def parse_value(v):
            if isinstance(v, str):
                return f"\'{v}\'"
            return v

        # Complains about accessing init of an instance since it can change
        # which is exactly the purpose of this method!
        init_params = inspect.signature(self.__init__).parameters  # type: ignore
        args = ','.join(
            map(lambda x: f'{x}={parse_value(getattr(self, x))}', init_params)
        )
        return f'{self.__class__.__name__}({args})'

    @property
    @abstractmethod
    def rtype(self) -> t.Type[RT]:
        """
        Variable's return type, such that `rtype("result")` converts to
        the relevant type.
        """

    @abstractmethod
    def calculate(self, obj: OT, mapping: t.Optional[MappingT] = None) -> RT:
        """
        Calculate variable. Each variable defines its own calculation strategy.

        :param obj: An object used for variable's calculation.
        :param mapping: Mapping from generalizable positions of
            MSA/reference/etc. to the `obj`'s positions.
        :return: Calculation result.
        :raises: :class:`FailedCalculation` if the calculation fails.
        """


class StructureVariable(AbstractVariable[bst.AtomArray, RT], t.Generic[RT]):
    """
    A type of variable whose :meth:`calculate` method requires protein
    structure.
    """

    __slots__ = ()

    @abstractmethod
    def calculate(self, obj: bst.AtomArray, mapping: t.Optional[MappingT] = None) -> RT:
        """
        :param obj: Some atom array.
        :param mapping: Optional mapping between structure and some reference
            object numbering schemes.
        :return: A calculation result of some sensible non-sequence type, such
            as string, float, int, etc.
        """


class SequenceVariable(AbstractVariable[abc.Sequence[T], RT], t.Generic[T, RT]):
    """
    A type of variable whose :meth:`calculate` method requires protein
    sequence.
    """

    __slots__ = ()

    @abstractmethod
    def calculate(
        self, obj: abc.Sequence[T], mapping: t.Optional[MappingT] = None
    ) -> RT:
        """
        :param obj: Some sequence.
        :param mapping: Optional mapping between sequence and some reference
            object numbering schemes.
        :return: A calculation result of some sensible non-sequence type, such
            as string, float, int, etc.
        """


VT = t.TypeVar(
    'VT', bound=t.Union[StructureVariable, SequenceVariable]
)  # variable type


class Variables(UserDict):
    # TODO: Proper generic type?
    """
    A subclass of :class:`dict` holding variables (:class:`AbstractVariable`
    subclasses).

    The keys are the :class:`AbstractVariable` subclasses' instances
    (hashed by :meth::class:`id <AbstractVariable.id>`), and values are
    calculation results.
    """

    def __getitem__(self, item: str | AbstractVariable):
        if isinstance(item, str):
            return super().__getitem__(hash(item))
        return super().__getitem__(item)

    @property
    def structure(self) -> Variables:
        """
        :return: values that are :class:`StructureVariable` instances.
        """
        return Variables(
            {k: v for k, v in self.items() if isinstance(k, StructureVariable)}
        )

    @property
    def sequence(self) -> Variables:
        """
        :return: values that are :class:`SequenceVariable` instances.
        """
        return Variables(
            {k: v for k, v in self.items() if isinstance(k, SequenceVariable)}
        )

    @classmethod
    def read(cls, path: Path) -> Variables:
        # TODO: does it still need the dynamic imports?
        """
        Read and initialize variables.

        :param path: Path to a two-column .tsv file holding pairs
            (var_id, var_value). Will use `var_id` to initialize variable,
            importing dynamically a relevant class from :mod:`variables`.
        :return: A dict mapping variable object to its value.
        """

        try:
            vs = read_n_col_table(path, 2) or pd.DataFrame()
        except pd.errors.EmptyDataError:
            vs = pd.DataFrame()
        variables = cls()

        for v_id, v_val in vs.itertuples(index=False):
            v_name = v_id.split('(')[0]

            import_statement = f'from lXtractor.variables import {v_name}'
            try:
                exec(import_statement)
            except ImportError:
                LOGGER.exception(
                    f'Failed to exec {import_statement} for variable {v_name} '
                    f'causing variable\'s init to fail'
                )
                continue

            try:
                v = eval(v_id)
            except Exception as e:
                LOGGER.exception(f'Failed to eval variable {v_id} due to {e}')
                continue
            try:
                v_val = eval(v_val)
            except Exception as e:
                LOGGER.debug(f'Failed to eval {v_val} for variable {v_name} due to {e}')
            variables[v] = v_val

        return variables

    def write(self, path: Path) -> None:
        """
        :param path: Path to a file.
        :param skip_if_contains: Skip if a variable ID contains any of the
            provided strings.
        """
        items = (f'{v.id}\t{r}' for v, r in self.items())
        path.write_text('\n'.join(items))

    def as_df(self) -> pd.DataFrame:
        """
        :return: A table with two columns: VariableID and VariableResult.
        """
        if len(self) == 0:
            return pd.DataFrame()
        return pd.DataFrame(
            {'VariableID': [k.id for k in self], 'VariableResult': list(self.values())}
        )


class AbstractCalculator(t.Generic[OT], metaclass=ABCMeta):
    """
    Class defining variables' calculation strategy.
    """

    __slots__ = ()

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

    @abstractmethod
    def __call__(
        self,
        o: OT | abc.Iterable[OT],
        v: VT | abc.Iterable[VT] | abc.Iterable[abc.Iterable[VT]],
        m: MappingT | abc.Iterable[MappingT | None] | None,
    ) -> ERT | abc.Iterable[abc.Iterable[ERT]]:
        """
        :param o: Object to calculate on.
        :param v: Some variable whose `calculate` method accepts `o`-type
            instances.
        :param m: Optional mapping between object and some reference object
            numbering schemes.
        :return: Calculation result.
        """

    @abstractmethod
    def map(self, o: OT, v: abc.Iterable[VT], m: MappingT | None) -> abc.Iterable[ERT]:
        """
        Map variables to a single object.

        :param o: Object to calculate on.
        :param v: An iterable over variables whose `calculate` method accepts
            `o`-type instances.
        :param m: Optional mapping between object and some reference object
            numbering schemes.
        :return: An iterator (generator) over calculation result.
        """

    @abstractmethod
    def vmap(
        self, o: abc.Iterable[OT], v: VT, m: abc.Iterable[MappingT | None]
    ) -> abc.Iterable[ERT]:
        """
        Map objects to a single variable.

        :param o: An iterable over objects to calculate on.
        :param v: Some variable whose `calculate` method accepts `o`-type
            instances.
        :param m:  Optional mapping between object and some reference object
            numbering schemes.
        :return: An iterator (generator) over calculation result.
        """


# class CalculatorProtocol(t.Protocol[OT, VT, RT]):
#     """
#     An interface of a calculator definition for typing.
#     """
#
#     @t.overload
#     def __call__(self, o: OT, v: VT, m: MappingT | None, *args, **kwargs) -> RT:
#         ...
#
#     @t.overload
#     def __call__(
#         self,
#         o: abc.Iterable[OT],
#         v: abc.Iterable[abc.Iterable[VT]],
#         m: abc.Iterable[MappingT | None] | None,
#         *args,
#         **kwargs,
#     ) -> abc.Iterable[abc.Iterable[RT]]:
#         ...
#
#     def __call__(
#         self,
#         o: OT | abc.Iterable[OT],
#         v: VT | abc.Iterable[abc.Iterable[VT]],
#         m: MappingT | abc.Iterable[MappingT | None] | None,
#         *args,
#         **kwargs,
#     ) -> RT | abc.Iterable[abc.Iterable[RT]]:
#         ...


class ProtFP:
    """
    ProtFP embeddings for amino acid residues.

    ProtFP is a coding scheme derived from
    the PCA analysis of the AAIndex database :cite:`pfp1,pfp2`.

    >>> pfp = ProtFP()
    >>> pfp[('G', 1)]
    -5.7
    >>> list(pfp['G'])
    [-5.7, -8.72, 4.18, -1.35, -0.31]
    >>> comp1 = pfp[1]
    >>> assert len(comp1) == 20
    >>> comp1[0]
    -5.7
    >>> comp1.index[0]
    'G'

    .. bibliography::

    """

    def __init__(self, path: Path = Path(resources.__file__).parent / 'PFP.csv'):
        self._df = pd.read_csv(path).set_index('AA')

    @t.overload
    def __getitem__(self, item: tuple[str, int]) -> float:
        ...

    @t.overload
    def __getitem__(self, item: str) -> np.ndarray:
        ...

    @t.overload
    def __getitem__(self, item: int) -> pd.Series:
        ...

    def __getitem__(
        self, item: tuple[str, int] | str | int
    ) -> float | np.ndarray | pd.Series:
        match item:
            case [c, i]:
                # 1. Fails to infer types of c and i
                # 2. Fails to infer return type (thinks it's a data frame)
                return self._df.loc[c, str(i - 1)]  # type: ignore
            case str():
                # Pandas always puts "ExtensionArray | ndarray" as a return
                # type, although only ndarray is actually returned here
                return self._df.loc[item].values  # type: ignore
            case int():
                return self._df[str(item - 1)]
            case _:
                raise TypeError(f'Invalid index type {item}')


@curry
def _try_map(p: T, m: abc.Mapping[T, V] | None) -> V | T:
    try:
        if m is not None:
            return m[p]
        return p
    except KeyError as e:
        raise FailedCalculation(f'Missing {p} in mapping') from e


if __name__ == '__main__':
    raise RuntimeError
