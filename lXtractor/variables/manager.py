import typing as t
from collections import abc
from itertools import chain

import biotite.structure as bst
import pandas as pd
from more_itertools import nth, partition

from lXtractor.core.chain import ChainList, Chain, SS, ChainSequence, ChainStructure
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData
from lXtractor.variables.base import AbstractManager, CalcT, VT, SequenceVariable, StructureVariable

T = t.TypeVar('T')
StagedSeq: t.TypeAlias = tuple[
    ChainSequence, abc.Iterator[SequenceVariable],
    abc.Iterable[T], abc.Mapping[int, int] | None]
StagedStr: t.TypeAlias = tuple[
    ChainStructure, abc.Iterator[StructureVariable],
    bst.AtomArray, abc.Mapping[int, int] | None]


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
            self, chains: Chain | ChainList, level: int | None,
            id_contains: str | None, obj_type: str | abc.Sequence[str] | None,
    ) -> abc.Iterator[SS]:
        self._validate_obj_type(obj_type)
        match obj_type:
            case str():
                obj_type = [obj_type]
            case None:
                obj_type = list(self.valid_types)
        if isinstance(chains, Chain):
            chains = ChainList([chains])
        if level is None:
            chains = chains + chains.collapse_children()
        elif level == 0:
            chains = chains
        else:
            chains = nth(chains.iter_children(), level, default=None)
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
        return structure

    def assign(
            self, vs: abc.Sequence[VT], chains: Chain | ChainList, *,
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
            self, chains: Chain | ChainList, vs: abc.Sequence[VT] | None = None, *,
            level: int | None = None, id_contains: str | None = None,
            obj_type: abc.Sequence[str] | str | None = None
    ) -> t.NoReturn:
        objs = self._filter_objs(chains, level, id_contains, obj_type)
        for obj in objs:
            for v in obj.variables:
                if vs is not None:
                    if v in vs:
                        del obj.variables[v]
                else:
                    del obj.variables[v]

    def reset(
            self, chains: Chain | ChainList, vs: abc.Sequence[VT] | None, *, level: int | None,
            id_contains: str | None, obj_type: abc.Sequence[str] | str | None
    ) -> t.NoReturn:
        objs = self._filter_objs(chains, level, id_contains, obj_type)
        for obj in objs:
            for v in obj.variables:
                if vs is not None:
                    if v in vs:
                        obj.variables[v] = None
                else:
                    obj.variables[v] = None

    def aggregate(
            self, chains: Chain | ChainList, *, level: int | None,
            id_contains: str | None, obj_type: abc.Sequence[str] | str | None
    ) -> pd.DataFrame:
        def _get_vs(obj: SS):
            return ((obj.id, str(type(obj)), k.id, str(type(k)), v, k.rtype)
                    for k, v in obj.variables)

        objs = self._filter_objs(chains, level, id_contains, obj_type)

        return pd.DataFrame(
            chain.from_iterable(map(_get_vs, objs)),
            columns=['ObjectID', 'ObjectType', 'VarID', 'VarType', 'VarResult', 'VarRType'])

    def stage_calculations(
            self, chains: Chain | ChainList, *, missing: bool = True,
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

        def get_vs(obj: SS) -> t.Optional[abc.Iterator[VT]]:
            if missing:
                return (v for v, r in obj.variables.items() if r is None)
            return iter(obj.variables)

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
            if vs is None or obj is None:
                return None
            return obj, vs, target, mapping

        objs = self._filter_objs(chains, level, id_contains, obj_type)
        seqs, strs = self._split_objects(objs)
        staged_seqs, staged_strs = map(
            lambda xs: filter(bool, map(stage, xs)), [seqs, strs])
        return staged_seqs, staged_strs

    def calculate(
            self, chains: Chain | ChainList, calculator: CalcT, *, missing: bool = True,
            seq_name: str = SeqNames.seq1, map_name: t.Optional[str] = None, level: int | None = None,
            id_contains: str | None = None, obj_type: abc.Sequence[str] | str | None = None
    ) -> t.NoReturn:
        pass


if __name__ == '__main__':
    raise RuntimeError
