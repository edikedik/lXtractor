import typing as t
from collections import abc
from itertools import chain

import biotite.structure as bst
import pandas as pd
from more_itertools import nth, partition

from lXtractor.core.chain import ChainList, SS, ChainSequence, ChainStructure, CT
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData
from lXtractor.variables.base import AbstractManager, CalcT, VT, SequenceVariable, StructureVariable

T = t.TypeVar('T')
StagedSeq: t.TypeAlias = tuple[
    ChainSequence, list[SequenceVariable], abc.Iterable[T], abc.Mapping[int, int] | None]
StagedStr: t.TypeAlias = tuple[
    ChainStructure, list[StructureVariable], bst.AtomArray, abc.Mapping[int, int] | None]


class Manager(AbstractManager):

    @property
    def valid_types(self) -> set[str]:
        return {'seq', 'str', 'str_seq'}

    def _validate_obj_type(self, obj_type: str | abc.Sequence[str] | None):
        match obj_type:
            case str():
                is_valid = obj_type[:3].lower() in self.valid_types
            case abc.Sequence():
                is_valid = set(obj_type).issubset(self.valid_types)
            case _:
                is_valid = False
        if not is_valid:
            raise ValueError(f'Invalid object type {obj_type}')

    def _filter_objs(
            self, chains: CT | ChainList, level: int | None,
            id_contains: str | None, obj_type: str | abc.Sequence[str] | None,
    ) -> abc.Iterator[SS]:

        match obj_type:
            case str():
                obj_type = [obj_type]
            case None:
                obj_type = list(self.valid_types)
        self._validate_obj_type(obj_type)

        if not isinstance(chains, ChainList):
            if not isinstance(chains, abc.Iterable):
                chains = [chains]
            else:
                chains = list(chains)
            chains = ChainList(chains)
        if level is None:
            chains = chains + chains.collapse_children()
        elif level == 0:
            chains = chains
        else:
            chains = nth(chains.iter_children(), level - 1, default=None)
            if chains is None:
                raise MissingData(f'Level {level} is absent')

        objs = iter([])
        if 'seq' in obj_type:
            objs = chain(objs, chains.iter_sequences())
        if 'str' in obj_type:
            objs = chain(objs, chains.iter_structures())
        if 'str_seq' in obj_type:
            objs = chain(objs, (s.seq for s in chains.iter_structures()))
        if id_contains:
            objs = filter(lambda o: id_contains in o.id, objs)
        return objs

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
            self, vs: abc.Sequence[VT], chains: CT | ChainList, *,
            level: int | None = None, id_contains: str | None = None,
            obj_type: abc.Sequence[str] | str | None = None
    ) -> t.NoReturn:
        def _assign(c: SS, _vs: abc.Sequence[VT]) -> t.NoReturn:
            for v in _vs:
                if c not in c.variables:
                    c.variables[v] = None

        objs = self._filter_objs(chains, level, id_contains, obj_type)
        seq_vs, str_vs = self._split_variables(vs)
        for obj in objs:
            if isinstance(obj, ChainSequence):
                _assign(obj, seq_vs)
            else:
                _assign(obj, str_vs)

    def remove(
            self, chains: CT | ChainList, vs: abc.Sequence[VT] | None = None, *,
            level: int | None = None, id_contains: str | None = None,
            obj_type: abc.Sequence[str] | str | None = None
    ) -> t.NoReturn:
        def _take_key(v):
            return vs is not None and v in vs or vs is None

        objs = self._filter_objs(chains, level, id_contains, obj_type)
        for obj in objs:
            keys = list(filter(_take_key, obj.variables))
            for k in keys:
                obj.variables.pop(k)

    def reset(
            self, chains: CT | ChainList, vs: abc.Sequence[VT] | None = None, *,
            level: int | None = None, id_contains: str | None = None,
            obj_type: abc.Sequence[str] | str | None = None
    ) -> t.NoReturn:
        def _take_key(v):
            return vs is not None and v in vs or vs is None

        objs = self._filter_objs(chains, level, id_contains, obj_type)
        for obj in objs:
            keys = list(filter(_take_key, obj.variables))
            for k in keys:
                obj.variables[k] = None

    def aggregate(
            self, chains: CT | ChainList, *, level: int | None = None,
            id_contains: str | None = None, obj_type: abc.Sequence[str] | str | None = None,
    ) -> pd.DataFrame:
        def _get_vs(obj: SS):
            vs_df = obj.variables.as_df()
            vs_df['ObjectID'] = obj.id
            vs_df['ObjectType'] = obj.__class__.__name__
            return vs_df

        objs = self._filter_objs(chains, level, id_contains, obj_type)
        vs = filter(lambda x: len(x) > 0, map(_get_vs, objs))
        return pd.concat(vs, ignore_index=True)

    def stage_calculations(
            self, chains: CT | ChainList, *, missing: bool = True,
            seq_name: str = SeqNames.seq1, map_name: t.Optional[str] = None, level: int | None = None,
            id_contains: str | None = None, obj_type: abc.Sequence[str] | str | None = None
    ) -> tuple[abc.Iterator[StagedSeq], abc.Iterator[StagedStr]]:
        def get_mapping(obj: SS) -> t.Optional[abc.Mapping[int, int]]:
            if map_name is None:
                return None
            if isinstance(obj, ChainStructure):
                obj = obj.seq
            map_from, map_to = obj[map_name], obj[SeqNames.enum]
            return dict(filter(lambda x: x[0] is not None, zip(map_from, map_to)))

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
            vs = get_vs(obj)
            target = self._find_structure(obj) if isinstance(obj, ChainStructure) else obj[seq_name]
            mapping = get_mapping(obj)
            if not vs or obj is None:
                return None
            return obj, vs, target, mapping

        objs = self._filter_objs(chains, level, id_contains, obj_type)
        seqs, strs = self._split_objects(objs)
        staged_seqs, staged_strs = map(
            lambda xs: filter(bool, map(stage, xs)), [seqs, strs])
        return staged_seqs, staged_strs

    def calculate(
            self, chains: CT | ChainList, calculator: CalcT, *, missing: bool = True,
            seq_name: str = SeqNames.seq1, map_name: t.Optional[str] = None, level: int | None = None,
            id_contains: str | None = None, obj_type: abc.Sequence[str] | str | None = None
    ) -> tuple[list[tuple[CT, VT, t.Any]], list[tuple[CT, VT, str]]]:
        staged_seqs, staged_strs = self.stage_calculations(
            chains, missing=missing, seq_name=seq_name, map_name=map_name,
            level=level, id_contains=id_contains, obj_type=obj_type)
        # element = (object, variable, is_calculated, result)
        calculated = chain.from_iterable(
            ((obj, v, *calculator(target, v, mapping)) for v in vs)
            for obj, vs, target, mapping in chain(staged_seqs, staged_strs))

        failures, successes = map(lambda xs: [(x[0], x[1], x[3]) for x in xs],
                                  partition(lambda x: x[2], calculated))
        for obj, v, res in successes:
            obj.variables[v] = res

        return successes, failures


if __name__ == '__main__':
    raise RuntimeError
