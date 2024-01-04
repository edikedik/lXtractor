import logging
import os
import sqlite3
import typing as t
from collections import abc
from itertools import chain, tee
from os import PathLike
from pathlib import Path

import pandas as pd
from more_itertools import spy

import lXtractor.chain as lxc
from lXtractor.util.typing import is_sequence_of
from lXtractor.variables.base import SequenceVariable, StructureVariable, LigandVariable
from lXtractor.variables.manager import CalcRes

LOGGER = logging.getLogger(__name__)
_T = t.TypeVar("_T")
_CT = t.TypeVar("_CT", lxc.ChainSequence, lxc.ChainStructure, lxc.Chain)
_CT_MAP = {lxc.ChainSequence: 1, lxc.ChainStructure: 2, lxc.Chain: 3}
_VT_MAP = {SequenceVariable: 1, StructureVariable: 2, LigandVariable: 3}


def _verify_chain_types(objs: abc.Sequence[t.Any], ct: t.Type[_CT]) -> None:
    if not is_sequence_of(objs, ct):
        raise TypeError(f"All chains must have the same type {ct}")


class Collection(t.Generic[_CT]):
    def __init__(self, loc: str | PathLike = ":memory:"):
        self._loc = loc if loc == ":memory:" else Path(loc)
        self._db = self._connect()
        self._setup()
        self._chains: tuple[_CT, ...] = ()

    @property
    def loaded(self) -> lxc.ChainList[_CT]:
        return lxc.ChainList(self._chains)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._loc)

    def _close(self) -> None:
        self._db.close()

    def _column_names_for(self, table_name) -> list[str]:
        res = self._execute(f"SELECT * FROM {table_name}")
        return [x[0] for x in res.description]

    def _num_rows_for(self, table_name: str) -> int:
        res = self._execute(f"SELECT COUNT(*) from {table_name}")
        return len(res.fetchall())

    def _execute(
        self,
        statement: str,
        params: abc.Iterable | dict[str, t.Any] = (),
        many: bool = False,
        script: bool = False,
    ) -> sqlite3.Cursor:
        if script and many:
            raise ValueError("`script` and `many` cannot both be True")
        try:
            with self._db as cur:
                if script:
                    return cur.executescript(statement)
                elif many:
                    return cur.executemany(statement, params)
                else:
                    return cur.execute(statement, params)
        except Exception:
            LOGGER.error(f"Failed to execute statement {statement}")
            raise

    @staticmethod
    def _peek_data_size(
        data: abc.Iterable[abc.Sequence[_T]],
    ) -> tuple[int, abc.Iterable[abc.Sequence[_T]]]:
        if isinstance(data, abc.Sequence):
            if not data:
                return -1, data
            return len(data[0]), data
        head, it = spy(data)
        if not head:
            return -1, it
        return len(head[0]), it

    def _insert(
        self,
        table_name: str,
        data: abc.Iterable[abc.Sequence[_T]],
        columns: abc.Sequence[str] = None,
        omit_first_id: bool = False,
        execute: bool = True,
    ) -> tuple[str | None, abc.Iterable[abc.Sequence[_T]]]:
        data_size, data = self._peek_data_size(data)
        if data_size == -1:
            LOGGER.warning(f"Attempting to insert empty `data` for {table_name}")
            return None, data
        placeholders = ", ".join("?" for _ in range(data_size))
        if columns is None:
            columns = self._column_names_for(table_name)
        if omit_first_id:
            columns = columns[1:]
        columns = ", ".join(columns)
        statement = f"INSERT INTO {table_name}({columns}) VALUES({placeholders})"
        if execute:
            self._execute(statement, data, many=True)
        return statement, data

    def _setup(self):
        make_chain_types = """ CREATE TABLE IF NOT EXISTS chain_types (
            type_id integer NOT NULL PRIMARY KEY AUTOINCREMENT,
            type_name TEXT NOT NULL
        ); """
        make_var_types = """ CREATE TABLE IF NOT EXISTS var_types (
            var_type_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            var_type_name TEXT NOT NULL UNIQUE
        ); """

        make_chains = """ CREATE TABLE IF NOT EXISTS chains (
            id TEXT PRIMARY KEY,
            chain_type INTEGER NOT NULL,
            level INTEGER NOT NULL,
            FOREIGN KEY (chain_type) REFERENCES chain_types(type_id)
        ); """
        make_parents = """ CREATE TABLE IF NOT EXISTS parents (
            chain_id_parent TEXT NOT NULL,
            chain_id_child TEXT NOT NULL,
            FOREIGN KEY (chain_id_parent) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
            FOREIGN KEY (chain_id_child) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE 
        ); """
        make_variables = """ CREATE TABLE IF NOT EXISTS variables (
            chain_id TEXT NOT NULL,
            variable_id TEXT NOT NULL,
            variable_calculated BOOL NOT NULL,
            variable_value TEXT,
            variable_type_id INTEGER NOT NULL,
            variable_rtype TEXT NOT NULL,
            FOREIGN KEY (chain_id) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
            FOREIGN KEY (variable_type_id) REFERENCES var_types (var_type_id)
        ); """
        make_paths = """ CREATE TABLE IF NOT EXISTS paths (
            chain_id TEXT NOT NULL PRIMARY KEY,
            chain_path TEXT NOT NULL,
            FOREIGN KEY (chain_id) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        ); """

        self._execute("PRAGMA foreign_keys = 1")

        for statement in [
            make_chain_types,
            make_var_types,
            make_chains,
            make_parents,
            make_variables,
            make_paths,
        ]:
            self._execute(statement)

        inserts = [
            ("chain_types", ((x.__name__,) for x in _CT_MAP)),
            ("var_types", ((x.__name__,) for x in _VT_MAP)),
        ]
        for table_name, data in inserts:
            if self._num_rows_for(table_name) == 1:
                self._insert(table_name, data, omit_first_id=True)

    def get_table(
        self, name: str, as_df: bool = False, **kwargs
    ) -> list[str] | pd.DataFrame | abc.Iterator[pd.DataFrame]:
        statement = f"SELECT * FROM {name}"
        if as_df:
            return pd.read_sql(statement, self._db, **kwargs)
        return self._execute(statement).fetchall()

    def get_ids(
        self, level: int | None = None, chain_type: str | int | None = None
    ) -> list[str]:
        statement = "SELECT id FROM chains"
        if level is not None:
            statement += f" WHERE level = {level}"
        if chain_type is not None:
            statement += f" AND chain_type = {chain_type}"
        res = self._execute(statement)
        return [x[0] for x in res.fetchall()]

    def _verify_chain_types(self, chains: t.Any) -> None:
        pass

    def _filter_existing_chains(self, chains: abc.Iterable[_CT]) -> abc.Iterator[_CT]:
        existing_ids = self.get_ids()
        return filter(lambda x: x.id in existing_ids, chains)

    def _filter_absent_chains(self, chains: abc.Iterable[_CT]) -> abc.Iterator[_CT]:
        existing_ids = self.get_ids()
        return filter(lambda x: x.id not in existing_ids, chains)

    def _filter_existing_ids(self, ids: abc.Iterable[str]) -> abc.Iterator[str]:
        existing_ids = self.get_ids()
        return filter(lambda x: x in existing_ids, ids)

    def _insert_chains_data(
        self, chains: lxc.ChainList[_CT], chain_type: int, level: int
    ):
        data = [(c.id, chain_type, level) for c in chains]
        self._insert("chains", data)

    def _add_chains_data(self, chains: lxc.ChainList[_CT], chain_type: int) -> None:
        for i, _chains in enumerate((chains, *chains.iter_children()), start=0):
            if _chains:
                self._insert_chains_data(_chains, chain_type, i)

    def _add_parents_data(self, chains: lxc.ChainList[_CT]) -> None:
        data = [(child.parent.id, child.id) for child in chains.collapse_children()]
        self._insert("parents", data)

    def add(self, chains: abc.Sequence[_CT], load: bool = False):
        if not chains:
            return
        self._verify_chain_types(chains)
        chain_type = _CT_MAP[chains[0].__class__]
        chains = lxc.ChainList[_CT](self._filter_absent_chains(chains))
        self._add_chains_data(chains, chain_type)
        self._add_parents_data(chains)
        if load:
            self._chains += tuple(chains)

    def unload(self, chains: abc.Sequence[_CT] | abc.Sequence[str]) -> None:
        if not chains:
            return
        ids = chains if isinstance(chains[0], str) else lxc.ChainList(chains).ids
        self._chains = tuple(self.loaded.filter(lambda x: x.id not in ids))

    def remove(self, chains: abc.Sequence[_CT] | abc.Sequence[str]) -> None:
        # TODO: removing from other tables here?
        if not chains:
            return
        if isinstance(chains[0], (lxc.Chain, lxc.ChainSequence, lxc.ChainStructure)):
            cl = lxc.ChainList(chains)
            ids = cl.ids + cl.collapse_children().ids
            if isinstance(cl[0], lxc.Chain):
                ids += cl.structures.ids
                ids += cl.collapse_children().structures.ids
        else:
            ids = chains
        ids = [(x,) for x in ids]
        self._execute("DELETE FROM chains WHERE id=?", ids, many=True)
        self.unload(chains)

    @staticmethod
    def _get_var_type(v: t.Any):
        if isinstance(v, SequenceVariable):
            return 1
        if isinstance(v, StructureVariable):
            return 2
        if isinstance(v, LigandVariable):
            return 3
        raise TypeError(f"Unsupported variable type {v.type}")

    def vs_add(
        self,
        vs: abc.Iterable[CalcRes],
        miscalculated: bool = False,
    ) -> None:
        existing = self.get_ids()
        vs = filter(lambda x: x[0].id in existing, vs)
        if not miscalculated:
            vs = filter(lambda x: x[2], vs)
        data = (
            (
                x[0].id,  # Chain ID
                x[1].id,  # Variable ID
                x[2],  # Is calculated?
                x[3],  # Calculation result
                self._get_var_type(x[1]),  # Variable type
                str(x[1].rtype.__name__),  # Return type
            )
            for x in vs
        )
        data = list(data)
        self._insert("variables", data)

    def _expand_structures(self, chain_path: Path) -> abc.Iterable[Path]:
        """The method is only relevant for :class:`ChainCollection`, where
        associated structure paths must be included as well"""
        yield from iter([])

    def _expand_children(self, chain_path: Path) -> abc.Iterator[Path]:
        for root, dirs, _ in os.walk(chain_path):
            root = Path(root)
            if root.name == "segments":
                yield from (root / x for x in dirs)

    def link(
        self,
        paths: abc.Iterable[Path | str | PathLike],
    ) -> None:
        existing_ids = self.get_ids()

        paths = filter(lambda x: x.exists() and x.is_dir(), map(Path, paths))
        p1, p2, p3 = tee(paths, 3)
        paths = chain(
            p1,
            chain.from_iterable(map(self._expand_children, p2)),
            chain.from_iterable(map(self._expand_structures, p3)),
        )
        paths = filter(lambda x: x.name in existing_ids, paths)

        data = ((x.name, str(x)) for x in paths)
        statement, data = self._insert("paths", data, execute=False)
        if statement is None:
            return
        statement += (
            " ON CONFLICT(chain_id) DO UPDATE SET chain_path=excluded.chain_path"
        )
        self._execute(statement, data, many=True)


class SequenceCollection(Collection[lxc.ChainSequence]):
    def _verify_chain_types(self, chains: abc.Sequence[t.Any]) -> None:
        _verify_chain_types(chains, lxc.ChainSequence)


class StructureCollection(Collection[lxc.ChainStructure]):
    def _verify_chain_types(self, chains: abc.Sequence[t.Any]) -> None:
        _verify_chain_types(chains, lxc.ChainStructure)


class ChainCollection(Collection[lxc.Chain]):
    def __init__(self, loc: str | PathLike = ":memory:"):
        super().__init__(loc)
        self._make_structures_table()

    def _make_structures_table(self):
        make_structures = """ CREATE TABLE IF NOT EXISTS structures(
            chain_id TEXT NOT NULL,
            structure_id TEXT NOT NULL,
            FOREIGN KEY(chain_id) REFERENCES chains(id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
            FOREIGN KEY(structure_id) REFERENCES chains(id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        ); """
        self._execute(make_structures)

    def _verify_chain_types(self, chains: abc.Sequence[t.Any]) -> None:
        _verify_chain_types(chains, lxc.Chain)

    def _insert_chains_data(
        self, chains: lxc.ChainList[_CT], chain_type: int, level: int
    ) -> None:
        data_chains = ((c.id, chain_type, level) for c in chains)
        data_structures = chain.from_iterable(
            ((s.id, 2, level) for s in c.structures) for c in chains
        )
        self._insert("chains", list(chain(data_chains, data_structures)))
        # Add chain--structure relationships
        data = list(
            chain.from_iterable(((c.id, s.id) for s in c.structures) for c in chains)
        )
        self._insert("structures", data)

    def _expand_structures(self, chain_path: Path) -> abc.Iterator[Path]:
        yield from chain_path.glob("structures/*")


if __name__ == "__main__":
    raise RuntimeError
