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
from collections import abc, defaultdict
from itertools import chain, groupby, repeat, tee

import numpy as np
import pandas as pd
from more_itertools import unzip, peekable
from toolz import curry
from tqdm.auto import tqdm

import lXtractor.core.chain as lxc
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData
from lXtractor.core.structure import GenericStructure
from lXtractor.variables.base import (
    VT,
    SequenceVariable,
    StructureVariable,
    Variables,
    AbstractCalculator,
)

SoS: t.TypeAlias = lxc.ChainSequence | lxc.ChainStructure
SoSv: t.TypeAlias = SequenceVariable | StructureVariable
CalcRes: t.TypeAlias = tuple[SoS, SoSv, bool, t.Any]
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

LOGGER = logging.getLogger(__name__)


def _update_variables(vs: Variables, upd: abc.Iterable[SoSv]) -> Variables:
    for v in upd:
        vs[v] = None
    return vs


def get_mapping(obj: t.Any, map_name: str | None, map_to: str | None) -> dict | None:
    """
    Obtain mapping from a Chain*-type object.

    >>> s = lxc.ChainSequence.from_string('ABCD', name='seq')
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
            raise MissingData(f'Object {obj} is missing `seq` attribute') from e
    else:
        seq = obj

    fr = seq[map_name]

    if map_to is None:
        if isinstance(obj, lxc.ChainStructure):
            to = seq[SeqNames.enum]
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
    obj: SoS, missing: bool
) -> list[SequenceVariable] | list[StructureVariable]:
    if missing:
        return [v for v, r in obj.variables.items() if r is None]
    return list(obj.variables)


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


def stage(
    obj: lxc.ChainStructure | lxc.ChainSequence,
    vs: abc.Sequence[VT] | None,
    *,
    missing: bool = True,
    seq_name: str = SeqNames.seq1,
    map_name: str | None = None,
    map_to: str | None = None,
) -> StagedStr | StagedSeq:
    """
    Stage object for calculation. If it's a chain sequence, will stage some
    sequence/mapping within it. If it's a chain structure, will stage the
    atom array.

    :param obj: Sequence/structure to calculate variables on.
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
        2. Staged sequence or atom array.
        3. A sequence of sequence or structural variables.
        4. An optional mapping.
    """
    target: lxc.ChainStructure | abc.Sequence | None

    seq_vs, str_vs = _split_variables(vs or _get_vs(obj, missing))
    mapping = get_mapping(obj, map_name, map_to)

    if isinstance(obj, lxc.ChainStructure):
        # TODO: searching always gets the original `obj`'s structure
        # since the chain structure must have an atom array.
        # to fix this, should I allow empty atom array?
        target = find_structure(obj)
        if isinstance(target, GenericStructure):
            return obj, target, str_vs, mapping
        raise MissingData(f'Failed to find structure for calculation on {obj}')
    if isinstance(obj, lxc.ChainSequence):
        target = obj[seq_name]
        return obj, target, seq_vs, mapping
    raise TypeError(f'Invalid object type {type(obj)}')


def find_structure(s: lxc.ChainStructure) -> GenericStructure | None:
    """
    Recursively search for structure up the ancestral tree.

    :param s: An arbitrary chain structure.
    :return: The first non-empty atom array up the parent chain.
    """
    structure = s.pdb.structure
    parent = s.parent
    while structure is None and parent is not None:
        structure = parent.pdb.structure
        parent = parent.parent
    return None or structure


def _split_objects(
    objs: abc.Iterable[SoS],
) -> tuple[list[lxc.ChainSequence], list[lxc.ChainStructure]]:
    obs1, obs2 = tee(objs)
    seqs = [x for x in obs1 if isinstance(x, lxc.ChainSequence)]
    strs = [x for x in obs2 if isinstance(x, lxc.ChainStructure)]
    return seqs, strs


def _split_variables(
    vs: abc.Iterable[SoSv],
) -> tuple[list[SequenceVariable], list[StructureVariable]]:
    vs1, vs2 = tee(vs)
    seq_vs = [x for x in vs1 if isinstance(x, SequenceVariable)]
    str_vs = [x for x in vs2 if isinstance(x, StructureVariable)]
    return seq_vs, str_vs


class Manager:
    """
    Manager of variable calculations, handling assignment, aggregation, and,
    of course, the calculations themselves.
    """

    __slots__ = ('verbose',)

    def __init__(self, verbose: bool = False):
        """
        :param verbose: Display progress bar.
        """
        self.verbose = verbose

    def assign(self, vs: abc.Sequence[SoSv], chains: abc.Iterable[SoS]):
        """
        Assign variables to chains sequences/structures.

        :param vs: A sequence of variables.
        :param chains: An iterable over chain sequences/structures.
        :return: No return. Will store assigned variables within the
            `variables` attribute.
        """

        seq_vs, str_vs = _split_variables(vs)
        seqs, strs = _split_objects(chains)

        staged_objs: abc.Iterable[tuple[SoS, abc.Sequence[SoSv]]] = chain(
            zip(seqs, repeat(seq_vs)), zip(strs, repeat(str_vs))
        )

        if self.verbose:
            staged_objs = tqdm(staged_objs, desc='Assigning variables')

        for o, _vs in staged_objs:
            o.variables.update({v: None for v in _vs})

    def remove(self, chains: abc.Iterable[SoS], vs: abc.Sequence[SoSv] | None = None):
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
            chains = tqdm(chains, desc='Removing variables')

        for c in chains:
            keys = list(filter(_take_key, c.variables))
            for k in keys:
                c.variables.pop(k)

    def reset(self, chains: abc.Iterable[SoS], vs: abc.Sequence[SoSv] | None = None):
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
            chains = tqdm(chains, desc='Resetting variable results')

        for c in chains:
            keys = list(filter(_take_key, c.variables))
            for k in keys:
                c.variables[k] = None

    def aggregate_from_chains(self, chains: abc.Iterable[SoS]) -> pd.DataFrame:
        """
        Aggregate calculation results from the `variables` container of the
        provided chains.

        >>> from lXtractor.variables.sequential import SeqEl
        >>> s = lxc.ChainSequence.from_string('abcd', name='seq')
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

        def get_vs(obj: SoS) -> pd.DataFrame:
            vs_df = obj.variables.as_df()
            vs_df['ObjectID'] = obj.id
            vs_df['ObjectType'] = obj.__class__.__name__
            return vs_df

        vs: abc.Iterable[pd.DataFrame] = filter(
            lambda x: len(x) > 0, map(get_vs, chains)
        )
        vs = peekable(vs)
        if vs.peek(None) is None:
            return pd.DataFrame(
                columns=['VariableID', 'VariableResult', 'ObjectID', 'ObjectType']
            )

        if self.verbose:
            vs = tqdm(vs, desc='Aggregating variables')

        return pd.concat(vs, ignore_index=True)

    def aggregate_from_it(
        self,
        results: abc.Iterable[CalcRes],
        vs_to_cols: bool = True,
        replace_errors: bool = True,
        replace_errors_with: t.Any = np.NaN,
    ) -> pd.DataFrame | dict[str, list]:
        """
        Aggregate calculation results directly from :meth:`calculate` output.

        :param results: An iterable over calculation results.
        :param vs_to_cols: If ``True``, will attempt to use the wide format for
            the final results with variables are columns. Otherwise, will use
            the long format with fixed columns: "ObjectID", "VariableID",
            "VariableCalculated", and "VariableResult". Note that for the wide
            format to work, all objects variables were calculated on must have
            unique IDs.
        :param replace_errors: When calculation failed, replace the calculation
            results with certain value.
        :param replace_errors_with: Use this value to replace erroneous
            calculation results.
        :return: A table with results in long or short format. If the
            conversion to the DataFrame fails, will output a default dictionary
            holding aggregated results. It's a good idea to inspect this
            dict to find values (lists) with unexpected lengths diverging from
            the rest.
        """

        d: t.DefaultDict[str, list] = defaultdict(list)

        if self.verbose:
            results = tqdm(results, 'Aggregating variables')

        if vs_to_cols:
            # Note that these should already be sorted by ID
            for obj_id, group in groupby(results, lambda x: x[0].id):
                d['ObjectID'].append(obj_id)
                for _, v, is_calculated, calc_res in group:
                    if is_calculated:
                        d[v.id].append(calc_res)
                    else:
                        if replace_errors:
                            d[v.id].append(replace_errors_with)
                        else:
                            d[v.id].append(calc_res)
        else:
            for obj, v, is_calculated, calc_res in results:
                d['ObjectID'].append(obj.id)
                d['VariableID'].append(v.id)
                d['VariableCalculated'].append(is_calculated)
                d['VariableResult'].append(
                    replace_errors_with
                    if replace_errors and not is_calculated
                    else calc_res
                )
        try:
            return pd.DataFrame(d)
        except ValueError as e:
            LOGGER.error('Failed to convert to a DataFrame (stacktrace below)')
            LOGGER.exception(e)
            return d

    def stage(
        self, chains: abc.Iterable[SoS], vs: abc.Sequence[SoSv] | None, **kwargs
    ) -> abc.Generator[StagedSeq | StagedStr, None, None]:
        """
        Stage objects for calculations (e.g., using :meth:`calculate`).
        It's a useful method if using a different calculation method and/or
        parallelization strategy within a `Calculator` class.

        .. seealso::
            :func:`stage`
            :meth:`calculate`

        >>> from lXtractor.variables.sequential import SeqEl
        >>> s = lxc.ChainSequence.from_string('ABCD', name='seq')
        >>> m = Manager()
        >>> staged = list(m.stage([s], [SeqEl(1)]))
        >>> len(staged) == 1
        True
        >>> staged[0]
        (seq|1-4, 'ABCD', [SeqEl(p=1,_rtype='str',seq_name='seq1')], None)

        :param chains: An iterable over chain sequences/structures.
        :param vs: A sequence of variables. If not provided, will use assigned
            variables (see :meth:`assign`).
        :param kwargs: Passed to :func:`stage`.
        :return: An iterable over tuples holding data for variables'
            calculation.
        """

        if self.verbose:
            chains = tqdm(chains, desc='Staging calculations')

        yield from map(curry(stage)(vs=vs, **kwargs), chains)

    def calculate(
        self,
        chains: abc.Iterable[SoS],
        vs: abc.Sequence[SoSv] | None,
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
        >>> s = lxc.ChainSequence.from_string('ABCD', name='seq')
        >>> m = Manager()
        >>> c = GenericCalculator()
        >>> list(m.calculate([s], [SeqEl(1)], c))
        [(seq|1-4, SeqEl(p=1,_rtype='str',seq_name='seq1'), True, 'A')]
        >>> list(m.calculate([s], [SeqEl(5)], c))[0][-2:]
        (False, 'Missing index 4 in sequence')

        :param chains: An iterable over chain sequences/structures.
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

        objs, targets, variables, mappings = unzip(self.stage(chains, vs, **kwargs))
        variables1, variables2 = tee(variables)
        calculated = calculator(targets, variables1, mappings)

        if self.verbose:
            calculated = tqdm(calculated, desc='Calculating variables')

        for obj, _vs, results in zip(objs, variables2, calculated, strict=True):
            for v, (is_calculated, res) in zip(_vs, results, strict=True):
                if save:
                    if is_calculated:
                        obj.variables[v] = res
                    else:
                        obj.variables[v] = None
                yield obj, v, is_calculated, res


if __name__ == '__main__':
    raise RuntimeError
