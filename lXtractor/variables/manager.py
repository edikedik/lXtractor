import logging
import typing as t
from collections import abc, defaultdict
from itertools import chain, groupby, repeat, tee

import biotite.structure as bst
import numpy as np
import pandas as pd
from more_itertools import unzip
from toolz import curry
from tqdm.auto import tqdm

from lXtractor.core.chain import ChainSequence, ChainStructure
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData
from lXtractor.variables.base import (
    VT,
    SequenceVariable,
    StructureVariable,
    Variables,
    AbstractCalculator,
)

SoS: t.TypeAlias = ChainSequence | ChainStructure
SoSv: t.TypeAlias = SequenceVariable | StructureVariable
CalcRes: t.TypeAlias = tuple[SoS, SoSv, bool, t.Any]
StagedSeq: t.TypeAlias = tuple[
    ChainSequence,
    abc.Sequence[t.Any],
    abc.Sequence[SequenceVariable],
    abc.Mapping[int, int] | None,
]
StagedStr: t.TypeAlias = tuple[
    ChainStructure,
    bst.AtomArray,
    abc.Sequence[StructureVariable],
    abc.Mapping[int, int] | None,
]

LOGGER = logging.getLogger(__name__)


def _update_variables(vs: Variables, upd: abc.Iterable[SoSv]) -> Variables:
    for v in upd:
        vs[v] = None
    return vs


def get_mapping(
    obj: t.Any, map_name: str | None, map_to: str | None
) -> t.Optional[dict[int, int]]:
    if map_name is None:
        return None

    if not isinstance(obj, ChainSequence):
        try:
            seq = obj.seq
        except AttributeError as e:
            raise MissingData(f'Object {obj} is missing `seq` attribute') from e
    else:
        seq = obj

    fr = seq[map_name]

    if map_to is None:
        if isinstance(obj, ChainStructure):
            to = seq[SeqNames.enum]
        else:
            to = range(1, len(fr) + 1)
    else:
        to = seq[map_to]

    return dict(filter(lambda x: x[0] is not None, zip(fr, to, strict=True)))


@t.overload
def get_vs(obj: ChainStructure, missing) -> list[StructureVariable]:
    ...


@t.overload
def get_vs(obj: ChainSequence, missing) -> list[SequenceVariable]:
    ...


def get_vs(obj: SoS, missing: bool) -> list[SequenceVariable] | list[StructureVariable]:
    if missing:
        return [v for v, r in obj.variables.items() if r is None]
    return list(obj.variables)


@t.overload
def stage(obj: ChainStructure, vs, *, missing, seq_name, map_name, map_to) -> StagedStr:
    ...


@t.overload
def stage(obj: ChainSequence, vs, *, missing, seq_name, map_name, map_to) -> StagedSeq:
    ...


def stage(
    obj: ChainStructure | ChainSequence,
    vs: abc.Sequence[VT] | None,
    *,
    missing: bool = True,
    seq_name: str = SeqNames.seq1,
    map_name: str | None = None,
    map_to: str | None = None,
) -> StagedStr | StagedSeq:
    target: ChainStructure | abc.Sequence | None

    seq_vs, str_vs = _split_variables(vs or get_vs(obj, missing))
    mapping = get_mapping(obj, map_name, map_to)

    if isinstance(obj, ChainStructure):
        target = find_structure(obj)
        if isinstance(target, bst.AtomArray):
            return obj, target, str_vs, mapping
        raise MissingData(f'Failed to find structure for calculation on {obj}')
    elif isinstance(obj, ChainSequence):
        target = obj[seq_name]
        return obj, target, seq_vs, mapping
    else:
        raise TypeError(f'Invalid object type {type(obj)}')


def find_structure(s: ChainStructure) -> bst.AtomArray | None:
    structure = s.pdb.structure
    parent = s.parent
    while structure is None and parent is not None:
        structure = parent.pdb.structure
        parent = parent.parent
    return None or structure.array


def _split_objects(
    objs: abc.Iterable[SoS],
) -> tuple[list[ChainSequence], list[ChainStructure]]:
    obs1, obs2 = tee(objs)
    seqs = [x for x in obs1 if isinstance(x, ChainSequence)]
    strs = [x for x in obs2 if isinstance(x, ChainStructure)]
    return seqs, strs


def _split_variables(
    vs: abc.Iterable[SoSv],
) -> tuple[list[SequenceVariable], list[StructureVariable]]:
    vs1, vs2 = tee(vs)
    seq_vs = [x for x in vs1 if isinstance(x, SequenceVariable)]
    str_vs = [x for x in vs2 if isinstance(x, StructureVariable)]
    return seq_vs, str_vs


class Manager:
    __slots__ = ('verbose',)

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def assign(self, vs: abc.Sequence[SoSv], chains: abc.Iterable[SoS]):

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
        def _take_key(v):
            return vs is not None and v in vs or vs is None

        if self.verbose:
            chains = tqdm(chains, desc='Removing variables')

        for c in chains:
            keys = list(filter(_take_key, c.variables))
            for k in keys:
                c.variables.pop(k)

    def reset(self, chains: abc.Iterable[SoS], vs: abc.Sequence[SoSv] | None = None):
        def _take_key(v):
            return vs is not None and v in vs or vs is None

        if self.verbose:
            chains = tqdm(chains, desc='Resetting variable results')

        for c in chains:
            keys = list(filter(_take_key, c.variables))
            for k in keys:
                c.variables[k] = None

    def aggregate_from_chains(self, chains: abc.Iterable[SoS]) -> pd.DataFrame:
        def _get_vs(obj: SoS) -> pd.DataFrame:
            vs_df = obj.variables.as_df()
            vs_df['ObjectID'] = obj.id
            vs_df['ObjectType'] = obj.__class__.__name__
            return vs_df

        vs: abc.Iterable[pd.DataFrame] = filter(
            lambda x: len(x) > 0, map(_get_vs, chains)
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
            LOGGER.error(f'Failed to convert to a DataFrame (stacktrace below)')
            LOGGER.exception(e)
            return d

    def stage(
        self,
        chains: abc.Iterable[SoS],
        vs: abc.Sequence[SoSv] | None,
        *,
        missing: bool = True,
        seq_name: str = SeqNames.seq1,
        map_name: str | None = None,
        map_to: str | None = None,
    ) -> abc.Generator[StagedSeq | StagedStr, None, None]:

        _stage = curry(stage)(
            vs=vs,
            missing=missing,
            seq_name=seq_name,
            map_name=map_name,
            map_to=map_to,
        )

        if self.verbose:
            chains = tqdm(chains, desc='Staging calculations')

        yield from map(_stage, chains)

    def calculate(
        self,
        chains: abc.Iterable[SoS],
        vs: abc.Sequence[SoSv] | None,
        calculator: AbstractCalculator,
        *,
        save: bool = False,
        **kwargs,
    ) -> abc.Generator[CalcRes, None, None]:

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
