import logging
import typing as t
from collections import abc, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, groupby, repeat, starmap, tee

import biotite.structure as bst
import numpy as np
import pandas as pd
from more_itertools import partition, unzip
from tqdm.auto import tqdm

from lXtractor.core.chain import ChainList, SS, ChainSequence, ChainStructure, CT
from lXtractor.core.config import SeqNames
from lXtractor.variables.base import VT, SequenceVariable, StructureVariable, Variables, CalculatorProtocol, RT, OT
from lXtractor.variables.calculator import SimpleCalculator, ParallelCalculator

T = t.TypeVar('T')
StagedSeq: t.TypeAlias = tuple[
    ChainSequence, abc.Sequence[T], list[SequenceVariable], abc.Mapping[int, int] | None]
StagedStr: t.TypeAlias = tuple[
    ChainStructure, bst.AtomArray, list[StructureVariable], abc.Mapping[int, int] | None]
C = t.Union[ChainSequence, ChainStructure]

LOGGER = logging.getLogger(__name__)


def _update_variables(vs: Variables, upd: abc.Iterable[VT]) -> Variables:
    for v in upd:
        vs[v] = None
    return vs


class Manager:
    __slots__ = ('num_proc', 'verbose')

    def __init__(self, num_proc: int | None = None, verbose: bool = False):
        self.num_proc = num_proc
        self.verbose = verbose

    @staticmethod
    def _split_variables(
            vs: abc.Iterable[VT]
    ) -> tuple[list[SequenceVariable], list[StructureVariable]]:
        seq_vs, str_vs = map(list, partition(
            lambda x: isinstance(x, StructureVariable), vs
        ))
        return seq_vs, str_vs

    @staticmethod
    def _split_objects(
            objs: abc.Iterable[SS]
    ) -> tuple[abc.Iterator[ChainSequence], abc.Iterator[ChainStructure]]:
        seqs, strs = partition(lambda x: isinstance(x, ChainStructure), objs)
        return seqs, strs

    @staticmethod
    def _find_structure(s: ChainStructure) -> t.Optional[bst.AtomArray]:
        structure = s.pdb.structure
        parent = s.parent
        while structure is None and parent is not None:
            structure = parent.pdb.structure
            parent = parent.parent
        return None or structure.array

    def assign(
            self, vs: abc.Sequence[VT], chains: abc.Iterable[SS]
    ) -> t.NoReturn:

        seq_vs, str_vs = self._split_variables(vs)
        seqs, strs = self._split_objects(chains)

        staged = ((obj, (obj.variables, upd_vs)) for obj, upd_vs in
                  chain(zip(seqs, repeat(seq_vs)), zip(strs, repeat(str_vs))))
        objs, updates = unzip(staged)

        if self.num_proc is None:
            updates = starmap(_update_variables, updates)

            if self.verbose:
                updates = tqdm(updates, desc='Assigning variables')

            for obj, updated_vs in zip(objs, updates):
                obj.variables = updated_vs

        else:
            with ProcessPoolExecutor(self.num_proc) as executor:

                old, new = unzip(updates)
                updates = executor.map(_update_variables, old, new)

                if self.verbose:
                    updates = tqdm(updates, desc='Assigning variables')

                for obj, updated_vs in zip(objs, updates):
                    obj.variables = updated_vs

    def remove(
            self, chains: abc.Iterable[SS], vs: abc.Sequence[VT] | None = None
    ) -> t.NoReturn:
        def _take_key(v):
            return vs is not None and v in vs or vs is None

        if self.verbose:
            chains = tqdm(chains, desc='Removing variables')

        for c in chains:
            keys = list(filter(_take_key, c.variables))
            for k in keys:
                c.variables.pop(k)

    def reset(
            self, chains: abc.Iterable[SS], vs: abc.Sequence[VT] | None = None
    ) -> t.NoReturn:
        def _take_key(v):
            return vs is not None and v in vs or vs is None

        if self.verbose:
            chains = tqdm(chains, desc='Resetting variable results')

        for c in chains:
            keys = list(filter(_take_key, c.variables))
            for k in keys:
                c.variables[k] = None

    def aggregate_from_chains(self, chains: abc.Iterable[SS]) -> pd.DataFrame:
        def _get_vs(obj: SS):
            vs_df = obj.variables.as_df()
            vs_df['ObjectID'] = obj.id
            vs_df['ObjectType'] = obj.__class__.__name__
            return vs_df

        vs = filter(lambda x: len(x) > 0, map(_get_vs, chains))

        if self.verbose:
            vs = tqdm(vs, desc='Aggregating variables')

        return pd.concat(vs, ignore_index=True)

    def aggregate_from_it(
            self, results: abc.Iterable[tuple[
                ChainStructure | ChainSequence, SequenceVariable | StructureVariable, bool, t.Any]],
            vs_to_cols: bool = True, replace_errors: bool = True,
            replace_errors_with: t.Any = np.NaN,
    ) -> pd.DataFrame | dict:
        d = defaultdict(list)
        if self.verbose:
            results = tqdm(results, 'Aggregating variables')

        if vs_to_cols:
            # Note that these should be already sorted by IDs
            results = groupby(results, lambda x: x[0].id)
            for obj_id, group in results:
                d['ObjectID'].append(obj_id)
                for _, v, is_calculated, calc_res in group:
                    if is_calculated:
                        d[v.id].append(calc_res)
                    else:
                        if replace_errors:
                            d[v.id].append(None)
                        else:
                            d[v.id].append(calc_res)
        else:
            for obj, v, is_calculated, calc_res in results:
                d['ObjectID'].append(obj.id)
                d['VariableID'].append(v.id)
                d['VariableCalculated'].append(is_calculated)
                d['VariableResult'].append(
                    replace_errors_with if replace_errors and not is_calculated else calc_res)
        try:
            return pd.DataFrame(d)
        except ValueError as e:
            LOGGER.error(f'Failed to convert to a DataFrame (stacktrace below)')
            LOGGER.exception(e)
            return d

    def stage_calculations(
            self, chains: CT | ChainList, vs: abc.Sequence[VT] | None, *, missing: bool = True,
            seq_name: str = SeqNames.seq1, map_name: t.Optional[str] = None, map_to: str | None = None
    ) -> tuple[abc.Iterator[StagedSeq], abc.Iterator[StagedStr]]:
        def get_mapping(obj: SS) -> t.Optional[abc.Mapping[int, int]]:
            if map_name is None:
                return None

            seq = obj.seq if isinstance(obj, ChainStructure) else obj

            fr = seq[map_name]

            if map_to is None:
                if isinstance(obj, ChainStructure):
                    to = seq[SeqNames.enum]
                else:
                    to = range(1, len(fr) + 1)
            else:
                to = seq[map_to]

            return dict(filter(lambda x: x[0] is not None, zip(fr, to, strict=True)))

        def get_vs(obj: SS) -> list[VT]:
            if missing:
                return [v for v, r in obj.variables.items() if r is None]
            return list(obj.variables)

        @t.overload
        def stage(obj: ChainStructure) -> t.Optional[StagedStr]:
            ...

        @t.overload
        def stage(obj: ChainSequence) -> t.Optional[StagedSeq]:
            ...

        def stage(obj: ChainStructure | ChainSequence) -> t.Optional[StagedStr | StagedSeq]:
            _vs = vs or get_vs(obj)

            target = self._find_structure(obj) if isinstance(obj, ChainStructure) else obj[seq_name]
            mapping = get_mapping(obj)

            return obj, target, _vs, mapping

        seqs, strs = self._split_objects(chains)

        staged_seqs, staged_strs = map(
            lambda xs: map(stage, xs), [seqs, strs])

        return staged_seqs, staged_strs

    def calculate_sequentially(
            self, chains: C | abc.Iterable[C], vs: abc.Sequence[VT] | None,
            calculator: CalculatorProtocol[OT, VT, RT], *, save: bool = False, missing: bool = True,
            seq_name: str = SeqNames.seq1, map_name: t.Optional[str] = None, map_to: str | None = None
    ) -> abc.Generator[tuple[C, VT, bool, t.Any]]:

        staged_seqs, staged_strs = self.stage_calculations(
            chains, vs, missing=missing, seq_name=seq_name, map_name=map_name, map_to=map_to)

        # element = (object, variable, is_calculated, result)
        calculated = chain.from_iterable(
            ((obj, v, *calculator(target, v, mapping)) for v in vs)
            for obj, target, vs, mapping in chain(staged_seqs, staged_strs))

        if self.verbose:
            calculated = tqdm(calculated, desc='Calculating variables')

        for obj, v, is_calculated, res in calculated:
            if save:
                if is_calculated:
                    obj.variables[v] = res
                else:
                    obj.variables[v] = None
            yield obj, v, is_calculated, res

    def calculate_parallel(
            self, chains: C | abc.Iterable[C], vs: abc.Sequence[VT] | None,
            calculator: CalculatorProtocol[OT, VT, RT], *, save: bool = False,
            missing: bool = True, seq_name: str = SeqNames.seq1,
            map_name: str | None = None, map_to: str | None = None
    ) -> abc.Generator[tuple[C, VT, bool, RT]]:
        staged_seqs, staged_strs = self.stage_calculations(
            chains, vs, missing=missing, seq_name=seq_name,
            map_name=map_name, map_to=map_to)

        objs, targets, variables, mappings = unzip(chain(staged_seqs, staged_strs))
        variables1, variables2 = tee(variables)
        calculated = calculator(targets, variables1, mappings)

        if self.verbose:
            calculated = tqdm(calculated, desc='Calculating variables')

        for obj, vs, results in zip(objs, variables2, calculated, strict=True):
            try:
                for v, (is_calculated, res) in zip(vs, results, strict=True):
                    if save:
                        if is_calculated:
                            obj.variables[v] = res
                        else:
                            obj.variables[v] = None
                    yield obj, v, is_calculated, res
            except Exception as e:
                print(f'Failed to assign variables due to {e}')
                print(len(vs), len(results))

    def calculate(
            self, chains: T | abc.Iterable[T], vs: abc.Sequence[VT] | None, *,
            missing: bool = True, seq_name: str = SeqNames.seq1,
            map_name: str | None = None, map_to: str | None = None
    ) -> abc.Generator[tuple[C, VT, bool, t.Any]]:
        if self.num_proc is None:
            yield from self.calculate_sequentially(
                chains, vs, SimpleCalculator(), missing=missing,
                seq_name=seq_name, map_name=map_name, map_to=map_to)
        else:
            yield from self.calculate_parallel(
                chains, vs, ParallelCalculator(self.num_proc), missing=missing,
                seq_name=seq_name, map_name=map_name, map_to=map_to)


if __name__ == '__main__':
    raise RuntimeError
