from __future__ import annotations

import inspect
import logging
import typing as t
from abc import abstractmethod, ABCMeta
from collections import UserDict, abc
from pathlib import Path

import biotite.structure as bst
import numpy as np
import pandas as pd

from lXtractor.util.io import read_n_col_table

AggFns = {'min': np.min, 'max': np.max, 'mean': np.mean, 'median': np.median}
LOGGER = logging.getLogger(__name__)
MappingT: t.TypeAlias = abc.Mapping[int, t.Optional[int]]
RT = t.TypeVar('RT')  # return type
OT = t.TypeVar('OT', str, bst.AtomArray)  # object type
T = t.TypeVar('T')


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
        return (not isinstance(other, type(self)) or
                self.id == other.id)

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

        init_params = inspect.signature(self.__init__).parameters
        args = ','.join(f'{k}={parse_value(v)}'
                        for k, v in vars(self).items() if k in init_params)
        return f'{self.__class__.__name__}({args})'

    @property
    @abstractmethod
    def rtype(self) -> t.Type[RT]:
        """
        Variable's return type, such that `rtype("result")` converts to the relevant type.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate(
            self, obj: OT, mapping: t.Optional[MappingT] = None
    ) -> RT:
        """
        Calculate variable. Each variable defines its own calculation strategy.

        :param obj: An object used for variable's calculation.
        :param mapping: Mapping from generalizable positions of MSA/reference/etc.
            to the `obj`'s positions.
        :return: Calculation result.
        :raises: :class:`FailedCalculation` if the calculation fails.
        """
        raise NotImplementedError


class StructureVariable(AbstractVariable[bst.AtomArray, RT]):
    """
    A type of variable whose :meth:`calculate` method requires protein structure.
    """

    @abstractmethod
    def calculate(
            self, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
    ) -> RT:
        raise NotImplementedError


class SequenceVariable(AbstractVariable[str, RT]):
    """
    A type of variable whose :meth:`calculate` method requires protein sequence.
    """

    @abstractmethod
    def calculate(self, seq: str, mapping: t.Optional[MappingT] = None) -> RT:
        raise NotImplementedError


VT = t.TypeVar('VT', bound=StructureVariable | SequenceVariable)  # variable type


class Variables(UserDict):
    # TODO: Proper generic type?
    """
    A subclass of :class:`dict` holding variables (:class:`AbstractVariable` subclasses).

    The keys are the :class:`AbstractVariable` subclasses' instances (since these are hashable objects),
    and values are calculation results.
    """

    @property
    def structure(self) -> Variables:
        """
        :return: values that are :class:`StructureVariable` instances.
        """
        return Variables({k: v for k, v in self.items() if isinstance(k, StructureVariable)})

    @property
    def sequence(self) -> Variables:
        """
        :return: values that are :class:`SequenceVariable` instances.
        """
        return Variables({k: v for k, v in self.items() if isinstance(k, SequenceVariable)})

    @classmethod
    def read(cls, path: Path) -> Variables:
        # TODO: read pdist
        # TODO: does it still need the dynamic imports?
        """
        Read and initialize variables.

        :param path: Path to a two-column .tsv file holding pairs (var_id, var_value).
            Will use `var_id` to initialize variable, importing dynamically a relevant
            class from :mod:`variables`.
        :return: A dict mapping variable object to its value.
        """

        try:
            vs = read_n_col_table(path, 2)
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
                    f'causing variable\'s init to fail')
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

    def write(
            self, path: Path,
            skip_if_contains: t.Collection[str] = ('ALL',)
    ) -> None:
        """
        :param path: Path to a file.
        :param skip_if_contains: Skip if variable ID contains any the provided strings.
            By defaults, skips all `ALL`-containing variables as these are expected to
            be pairwise distance matrices.
        """
        items = (f'{v.id}\t{r}' for v, r in self.items()
                 if all(x not in v.id for x in skip_if_contains))
        path.write_text('\n'.join(items))


class AbstractCalculator(t.Generic[OT, VT, RT], metaclass=ABCMeta):
    """
    Class defining variables' calculation strategy.
    """

    @abstractmethod
    def __call__(self, o: OT, v: VT, m: MappingT | None) -> RT: ...

    @abstractmethod
    def map(
            self, o: OT, v: abc.Iterable[VT], m: MappingT | None
    ) -> abc.Iterator[RT]: ...

    @abstractmethod
    def vmap(
            self, o: abc.Iterable[OT], v: VT, m: abc.Iterable[MappingT | None]
    ) -> abc.Iterator[RT]: ...


if __name__ == '__main__':
    raise RuntimeError
