"""
``Manager`` handles variable calculations, such as:

#. Variable manipulations (assignment, deletions, and resetting).
#. Calculation of variables. Simply manages the calculation process, whereas
    calculators (:class:`lXtractor.variables.calculator.GenericCalculator`
    for instance) do the heavy lifting.
#. Aggregation of the calculation results, either
    :meth:`from_chains <Manager.aggregate_from_chains>` or
    :meth:`from_iterable <Manager.aggregate_from_it>`.

"""
import logging
import typing as t
from collections import abc
from itertools import chain, repeat, tee

import numpy as np
import pandas as pd
from more_itertools import unzip, peekable, split_when, chunked
from toolz import curry
from tqdm.auto import tqdm

import lXtractor.chain as lxc
from lXtractor.core import Ligand
from lXtractor.core.config import DefaultConfig
from lXtractor.core.exceptions import MissingData
from lXtractor.core.structure import GenericStructure
from lXtractor.variables.base import (
    SequenceVariable,
    StructureVariable,
    Variables,
    AbstractCalculator,
    LigandVariable,
)

# TODO: get proper target by using `seq_name` from a seq var definition

T = t.TypeVar("T")
LigInp: t.TypeAlias = tuple[lxc.ChainStructure, Ligand]
Inp: t.TypeAlias = lxc.ChainSequence | lxc.ChainStructure | LigInp
InpT = t.TypeVar("InpT", lxc.ChainStructure, lxc.ChainSequence, LigInp)
InpV: t.TypeAlias = SequenceVariable | StructureVariable | LigandVariable
CalcRes: t.TypeAlias = tuple[Inp, InpV, bool, t.Any]
StagedSeq: t.TypeAlias = tuple[
    lxc.ChainSequence,
    abc.Sequence[t.Any],
    abc.Sequence[SequenceVariable],
    abc.Mapping[int, int] | None,
]
StagedStr: t.TypeAlias = tuple[
    lxc.ChainStructure,
    GenericStructure,
    abc.Sequence[StructureVariable],
    abc.Mapping[int, int] | None,
]
StagedLig: t.TypeAlias = tuple[
    tuple[lxc.ChainStructure, Ligand],
    Ligand,
    abc.Sequence[LigandVariable],
    abc.Mapping[int, int] | None,
]

LOGGER = logging.getLogger(__name__)


def _update_variables(vs: Variables, upd: abc.Iterable[InpV]) -> Variables:
    for v in upd:
        vs[v] = None
    return vs


def get_mapping(obj: t.Any, map_name: str | None, map_to: str | None) -> dict | None:
    """
    Obtain mapping from a Chain*-type object.

    >>> s = lxc.ChainSequence.from_string('ABCD', name='_seq')
    >>> s.add_seq('some_map', [5, 6, 7, 8])
    >>> s.add_seq('another_map', ['D', 'B', 'C', 'A'])
    >>> get_mapping(s, 'some_map', None)
    {5: 1, 6: 2, 7: 3, 8: 4}
    >>> get_mapping(s, 'another_map', 'some_map')
    {'D': 5, 'B': 6, 'C': 7, 'A': 8}

    :param obj: Chain*-type object. If not a Chain*-type object,
        raises `AttributeError`.
    :param map_name: The name of a map to create the mapping from.
        If ``None``, the resulting mapping is ``None``.
    :param map_to: The name of a map to create a mapping to.
        If ``None``, will default to the real sequence indices (1-based) for a
        :class:`ChainSequence <lXtractor.core.chain.ChainSequence>` object
        and to the structure actual numbering for the
        :class:`ChainStructure <lXtractor.core.chain.ChainStructure>`.
    :return: A dictionary mapping from the `map_name` sequence to `map_to`
        sequence.
    """
    if map_name is None:
        return None

    if not isinstance(obj, lxc.ChainSequence):
        try:
            seq = obj.seq
        except AttributeError as e:
            raise MissingData(f"Object {obj} is missing `seq` attribute") from e
    else:
        seq = obj

    fr = seq[map_name]

    if map_to is None:
        if isinstance(obj, lxc.ChainStructure):
            to = seq[DefaultConfig["mapnames"]["enum"]]
        else:
            to = range(1, len(fr) + 1)
    else:
        to = seq[map_to]

    return dict(filter(lambda x: x[0] is not None, zip(fr, to, strict=True)))


@t.overload
def _get_vs(obj: lxc.ChainStructure, missing) -> list[StructureVariable]:
    ...


@t.overload
def _get_vs(obj: lxc.ChainSequence, missing) -> list[SequenceVariable]:
    ...


def _get_vs(
    obj: Inp, missing: bool
) -> list[SequenceVariable] | list[StructureVariable]:
    if missing:
        return [v for v, r in obj.variables.items() if r is None]
    return list(obj.variables)


def _filter_type(xs: abc.Iterable[t.Any], _t: t.Type[T]) -> abc.Iterator[T]:
    return filter(lambda x: isinstance(x, _t), xs)


def _split_objects(
    objs: abc.Iterable[Inp],
) -> tuple[list[lxc.ChainSequence], list[lxc.ChainStructure], list[Ligand]]:
    types = [lxc.ChainSequence, lxc.ChainStructure, LigandVariable]
    seqs, strs, ligs = (
        list(_filter_type(xs, _t)) for xs, _t in zip(tee(objs, len(types)), types)
    )
    return seqs, strs, ligs


def _split_variables(
    vs: abc.Sequence[InpV],
) -> tuple[list[SequenceVariable], list[StructureVariable], list[LigandVariable]]:
    seq_vs, str_vs, lig_vs = (
        list(_filter_type(vs, _t))
        for _t in [SequenceVariable, StructureVariable, LigandVariable]
    )
    return seq_vs, str_vs, lig_vs


@t.overload
def stage(
    obj: lxc.ChainStructure, vs, *, missing, seq_name, map_name, map_to
) -> StagedStr:
    ...


@t.overload
def stage(
    obj: lxc.ChainSequence, vs, *, missing, seq_name, map_name, map_to
) -> StagedSeq:
    ...


@t.overload
def stage(
    obj: lxc.ChainSequence, vs, *, missing, seq_name, map_name, map_to
) -> StagedSeq:
    ...


@t.overload
def stage(obj: LigInp, vs, *, missing, seq_name, map_name, map_to) -> StagedLig:
    ...


def stage(
    obj: InpT,
    vs: abc.Sequence[InpV] | None,
    *,
    missing: bool = True,
    seq_name: str = DefaultConfig["mapnames"]["seq1"],
    map_name: str | None = None,
    map_to: str | None = None,
) -> StagedStr | StagedSeq | StagedLig:
    """
    Stage object for calculation. If it's a chain sequence, will stage some
    sequence/mapping within it. If it's a chain structure, will stage the
    atom array.

    :param obj: A chain sequence or structure or structure-ligand pair to
        calculate the variables on.
    :param vs: A sequence of variables to calculate.
    :param missing: If ``True``, calculate only those assigned variables that
        are missing.
    :param seq_name: If `obj` is the chain sequence, the sequence name is used
        to obtain an actual sequence (``obj[seq_name]``).
    :param map_name: The mapping name to obtain the mapping keys.
        If ``None``, the resulting mapping will be ``None``.
    :param map_to: The mapping name to obtain the mapping values.
        See :func:`get_mapping` for details.
    :return: A tuple with four elements:
        1. Original object.
        2. Staged target passed to a variable for calculation.
        3. A sequence of sequence or structural variables.
        4. An optional mapping.
    """
    target: lxc.ChainStructure | abc.Sequence | None

    def stage_vs_and_mapping(cs: lxc.ChainStructure | lxc.ChainSequence):
        return (
            *_split_variables(vs or _get_vs(cs, missing)),
            get_mapping(cs, map_name, map_to),
        )

    match obj:
        case lxc.ChainStructure():
            target = find_structure(obj)
            _, _vs, _, m = stage_vs_and_mapping(obj)
        case lxc.ChainSequence():
            target = obj[seq_name]
            _vs, _, _, m = stage_vs_and_mapping(obj)
        case (lxc.ChainStructure(), Ligand()):
            target = obj[1]
            _, _, _vs, m = stage_vs_and_mapping(obj[0])
        case _:
            raise TypeError(f"Invalid object type {type(obj)}")

    return obj, target, _vs, m


def find_structure(s: lxc.ChainStructure) -> GenericStructure | None:
    """
    Recursively search for structure up the ancestral tree.

    :param s: An arbitrary chain structure.
    :return: The first non-empty atom array up the parent chain.
    """
    structure = s.structure
    parent = s.parent
    while structure is None and parent is not None:
        structure = parent.structure
        parent = parent.parent
    return None or structure


class Manager:
    """
    Manager of variable calculations, handling assignment, aggregation, and,
    of course, the calculations themselves.
    """

    __slots__ = ("verbose",)

    def __init__(self, verbose: bool = False):
        """
        :param verbose: Display progress bar.
        """
        self.verbose = verbose

    def assign(self, vs: abc.Sequence[InpV], chains: abc.Iterable[Inp]):
        """
        Assign variables to chains sequences/structures.

        :param vs: A sequence of variables.
        :param chains: An iterable over chain sequences/structures.
        :return: No return. Will store assigned variables within the
            `variables` attribute.
        """

        seq_vs, str_vs, lig_vs = _split_variables(vs)
        seqs, strs, ligs = _split_objects(chains)

        staged_objs: abc.Iterable[tuple[Inp, abc.Sequence[InpV]]] = chain(
            zip(seqs, repeat(seq_vs)),
            zip(strs, repeat(str_vs)),
            zip(ligs, repeat(lig_vs)),
        )

        if self.verbose:
            staged_objs = tqdm(staged_objs, desc="Assigning variables")

        for o, _vs in staged_objs:
            o.variables.update({v: None for v in _vs})

    def remove(self, chains: abc.Iterable[Inp], vs: abc.Sequence[InpV] | None = None):
        """
        Remove variables from the `variables` container.

        :param chains: An iterable over chain sequences/structures.
        :param vs: A sequence of variables to remove. If not provided, will
            remove all variables.
        :return: No return.
        """

        def _take_key(v):
            return vs is not None and v in vs or vs is None

        if self.verbose:
            chains = tqdm(chains, desc="Removing variables")

        for c in chains:
            keys = list(filter(_take_key, c.variables))
            for k in keys:
                c.variables.pop(k)

    def reset(self, chains: abc.Iterable[Inp], vs: abc.Sequence[InpV] | None = None):
        """
        Similar to :meth:`remove`, but instead of deleting, resets variable
        calculation results.

        :param chains: An iterable over chain sequences/structures.
        :param vs: A sequence of variables to reset. If not provided, will
            reset all variables.
        :return: No return.
        """

        def _take_key(v):
            return vs is not None and v in vs or vs is None

        if self.verbose:
            chains = tqdm(chains, desc="Resetting variable results")

        for c in chains:
            keys = list(filter(_take_key, c.variables))
            for k in keys:
                c.variables[k] = None

    def aggregate_from_chains(self, chains: abc.Iterable[Inp]) -> pd.DataFrame:
        """
        Aggregate calculation results from the `variables` container of the
        provided chains.

        >>> from lXtractor.variables.sequential import SeqEl
        >>> s = lxc.ChainSequence.from_string('abcd', name='_seq')
        >>> manager = Manager()
        >>> manager.assign([SeqEl(1)], [s])
        >>> df = manager.aggregate_from_chains([s])
        >>> len(df) == 1
        True
        >>> list(df.columns)
        ['VariableID', 'VariableResult', 'ObjectID', 'ObjectType']

        :param chains: An iterable over chain sequences/structures.
        :return: A dataframe with `ObjectID`, `ObjectType`, and calculation
            results.
        """

        def get_vs(obj: Inp) -> pd.DataFrame:
            vs_df = obj.variables.as_df()
            vs_df["ObjectID"] = obj.id
            vs_df["ObjectType"] = obj.__class__.__name__
            return vs_df

        vs: abc.Iterable[pd.DataFrame] = filter(
            lambda x: len(x) > 0, map(get_vs, chains)
        )
        vs = peekable(vs)
        if vs.peek(None) is None:
            return pd.DataFrame(
                columns=["VariableID", "VariableResult", "ObjectID", "ObjectType"]
            )

        if self.verbose:
            vs = tqdm(vs, desc="Aggregating variables")

        return pd.concat(vs, ignore_index=True)

    def aggregate_from_it(
        self,
        results: abc.Iterable[CalcRes],
        vs_to_cols: bool = True,
        replace_errors: bool = True,
        replace_errors_with: t.Any = np.NaN,
        num_vs: int | None = None,
    ) -> pd.DataFrame | dict[str, list]:
        """
        Aggregate calculation results directly from :meth:`calculate` output.

        :param results: An iterable over calculation results.
        :param vs_to_cols: If ``True``, will attempt to use the wide format for
            the final results with variables as columns. Otherwise, will use
            the long format with fixed columns: "ObjectID", "VariableID",
            "VariableCalculated", and "VariableResult". Note that for the wide
            format to work, all objects and their variables must have
            unique IDs.
        :param replace_errors: When calculation failed, replace the calculation
            results with certain value.
        :param replace_errors_with: Use this value to replace erroneous
            calculation results.
        :param num_vs: The number of variables per object. Providing this will
            significantly increase the aggregation speed.
        :return: A table with results in long or short format.
        """

        def substitute_error(res):
            if res[2]:
                return res
            return res[0], res[1], res[2], replace_errors_with

        def substitute_ids(res):
            if isinstance(res[0], tuple):
                obj_id = res[0][1].id
            else:
                obj_id = res[0].id
            return obj_id, res[1].id, res[2], res[3]

        def wrap_into_series(res_chunk):
            idx = chain(["ObjectID"], (res[1] for res in res_chunk))
            vs = chain([res_chunk[0][0]], (res[-1] for res in res_chunk))
            return pd.Series(vs, idx)

        if self.verbose:
            results = tqdm(results, "Accumulating calculations")

        if replace_errors:
            results = map(substitute_error, results)

        if vs_to_cols:
            results = map(substitute_ids, results)
            if num_vs is not None:
                chunks = chunked(results, num_vs)
            else:
                chunks = split_when(results, lambda x, y: x[0] != y[0])
            wrapped = map(wrap_into_series, chunks)
            df = pd.DataFrame(wrapped)
        else:
            colnames = ["Object", "Variable", "VariableCalculated", "VariableResult"]
            df = pd.DataFrame(dict(zip(colnames, map(list, unzip(results)))))

        return df

    def stage(
        self, chains: abc.Iterable[Inp], vs: abc.Sequence[InpV] | None, **kwargs
    ) -> abc.Generator[StagedSeq | StagedStr, None, None]:
        """
        Stage objects for calculations (e.g., using :meth:`calculate`).
        It's a useful method if using a different calculation method and/or
        parallelization strategy within a `Calculator` class.

        .. seealso::
            :func:`stage`
            :meth:`calculate`

        >>> from lXtractor.variables.sequential import SeqEl
        >>> s = lxc.ChainSequence.from_string('ABCD', name='_seq')
        >>> m = Manager()
        >>> staged = list(m.stage([s], [SeqEl(1)]))
        >>> len(staged) == 1
        True
        >>> staged[0]
        (_seq|1-4, 'ABCD', [SeqEl(p=1,_rtype='str',seq_name='seq1')], None)

        :param chains: An iterable over chain sequences/structures.
        :param vs: A sequence of variables. If not provided, will use assigned
            variables (see :meth:`assign`).
        :param kwargs: Passed to :func:`stage`.
        :return: An iterable over tuples holding data for variables'
            calculation.
        """
        yield from map(curry(stage)(vs=vs, **kwargs), chains)

    def calculate(
        self,
        objs: abc.Iterable[Inp],
        vs: abc.Sequence[InpV] | None,
        calculator: AbstractCalculator,
        *,
        save: bool = False,
        **kwargs,
    ) -> abc.Generator[CalcRes, None, None]:
        """
        Handles variable calculations:

            1. Stage calculations (see :meth:`stage`).
            2. Calculate variables using the provided calculator.
            3. (Optional) save the calculation results to variables container.
            4. Output (stream) calculation results.

        Note that 3 and 4 are done lazily as calculation results from the
        calculator become available.

        >>> from lXtractor.variables.calculator import GenericCalculator
        >>> from lXtractor.variables.sequential import SeqEl
        >>> s = lxc.ChainSequence.from_string('ABCD', name='_seq')
        >>> m = Manager()
        >>> c = GenericCalculator()
        >>> list(m.calculate([s],[SeqEl(1)],c))
        [(_seq|1-4, SeqEl(p=1,_rtype='str',seq_name='seq1'), True, 'A')]
        >>> list(m.calculate([s],[SeqEl(5)],c))[0][-2:]
        (False, 'Missing index 4 in sequence')

        :param objs: An iterable over chain sequences/structures.
        :param vs: A sequence of variables. If not provided, will use assigned
            variables (see :meth:`assign`).
        :param calculator: A calculator object -- some callable with the right
            signature handling the calculations.
        :param save: Save calculation results to variables. Will overwrite any
            existing matching variables.
        :param kwargs: Passed to :meth:`stage`.
        :return: A generator over tuples:
            1. Original object.
            2. Variable.
            3. Flag indicated whether the calculation was successful.
            4. The calculation result (or the error message).
        """

        objs, targets, variables, mappings = unzip(self.stage(objs, vs, **kwargs))
        variables1, variables2 = tee(variables)
        calculated = calculator(targets, variables1, mappings)

        for obj, _vs, results in zip(objs, variables2, calculated, strict=True):
            for v, (is_calculated, res) in zip(_vs, results, strict=True):
                if save:
                    if is_calculated:
                        obj.variables[v] = res
                    else:
                        obj.variables[v] = None
                yield obj, v, is_calculated, res


if __name__ == "__main__":
    raise RuntimeError
