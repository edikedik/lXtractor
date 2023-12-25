import logging
import sqlite3
import typing as t
from collections import abc
from itertools import chain
from os import PathLike
from pathlib import Path

import pandas as pd

import lXtractor.chain as lxc
from lXtractor.core.exceptions import FormatError
from lXtractor.util.typing import is_sequence_of

LOGGER = logging.getLogger(__name__)
_CT = t.TypeVar("_CT", lxc.ChainSequence, lxc.ChainStructure, lxc.Chain)
_CT_MAP = {lxc.ChainSequence: 1, lxc.ChainStructure: 2, lxc.Chain: 3}


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
        params: abc.Sequence | dict[str, t.Any] = (),
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

    def _insert(
        self,
        table_name: str,
        data: abc.Sequence[abc.Sequence[t.Any]],
        columns: abc.Sequence[str] = None,
        omit_first_id: bool = False,
    ):
        if len(data) == 0:
            return
        data_sizes = set(map(len, data))
        if len(data_sizes) > 1:
            raise FormatError(
                f"Attempting to insert rows with unequal number of "
                f"elements ({data_sizes})."
            )
        data_size = data_sizes.pop()
        placeholders = ", ".join("?" for _ in range(data_size))
        if columns is None:
            columns = self._column_names_for(table_name)
        if omit_first_id:
            columns = columns[1:]
        columns = ", ".join(columns)
        statement = f"INSERT INTO {table_name}({columns}) VALUES({placeholders})"
        self._execute(statement, data, many=True)

    def _setup(self):
        make_chain_types = """ CREATE TABLE IF NOT EXISTS chain_types (
            type_id integer NOT NULL PRIMARY KEY AUTOINCREMENT,
            type_name TEXT NOT NULL
        ); """
        make_var_types = """ CREATE TABLE IF NOT EXISTS var_types (
            var_type_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            var_type_name TEXT NOT NULL UNIQUE
        ); """
        make_var_return_types = """ CREATE TABLE IF NOT EXISTS var_rtypes (
            rtype_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            rtype_name TEXT NOT NULL UNIQUE
        ) ; """

        make_chains = """ CREATE TABLE IF NOT EXISTS chains (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            chain_id TEXT NOT NULL UNIQUE,
            chain_type INTEGER NOT NULL,
            level INTEGER NOT NULL,
            FOREIGN KEY(chain_type) REFERENCES chain_types (type_id)
        ); """
        make_parents = """ CREATE TABLE IF NOT EXISTS parents (
            chain_id_parent TEXT NOT NULL,
            chain_id_child TEXT NOT NULL,
            FOREIGN KEY(chain_id_parent) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
            FOREIGN KEY(chain_id_child) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE 
        ); """
        make_variables = """ CREATE TABLE IF NOT EXISTS variables (
            chain_id INTEGER NOT NULL,
            variable_name TEXT NOT NULL,
            variable_value TEXT,
            variable_type_id INTEGER NOT NULL,
            variable_rtype_id INTEGER NOT NULL,
            FOREIGN KEY(chain_id) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
            FOREIGN KEY(variable_type_id) REFERENCES var_types (var_type_id)
            FOREIGN KEY(variable_rtype_ID) REFERENCES var_rtypes (rtype_id)
        ); """
        make_paths = """ CREATE TABLE IF NOT EXISTS paths (
            chain_id INTEGER NOT NULL PRIMARY KEY,
            chain_path TEXT NOT NULL,
            FOREIGN KEY(chain_id) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        ); """

        for statement in [
            make_chain_types,
            make_var_types,
            make_var_return_types,
            make_chains,
            make_parents,
            make_variables,
            make_paths,
        ]:
            self._execute(statement)

        inserts = [
            (
                "chain_types",
                (
                    ("ChainSequence",),
                    ("ChainStructure",),
                    ("Chain",),
                ),
            ),
            ("var_types", (("SequenceVariable",), ("StructureVariable",))),
            (
                "var_rtypes",
                (
                    ("str",),
                    ("float",),
                    ("int",),
                ),
            ),
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

    def _verify_chain_types(self, chains: t.Any) -> None:
        pass

    def _filter_existing_chains(self, chains: abc.Iterable[_CT]) -> abc.Iterator[_CT]:
        existing_ids = self.get_chain_ids()
        return filter(lambda x: x.id in existing_ids, chains)

    def _filter_absent_chains(self, chains: abc.Iterable[_CT]) -> abc.Iterator[_CT]:
        existing_ids = self.get_chain_ids()
        return filter(lambda x: x.id not in existing_ids, chains)

    def _filter_existing_ids(self, ids: abc.Iterable[str]) -> abc.Iterator[str]:
        existing_ids = self.get_chain_ids()
        return filter(lambda x: x in existing_ids, ids)

    def get_chain_ids(
        self, level: int | None = None, chain_type: str | int | None = None
    ) -> list[str]:
        statement = "SELECT chain_id FROM chains"
        if level is not None:
            statement += f" WHERE level = {level}"
        if chain_type is not None:
            statement += f" AND chain_type = {chain_type}"
        res = self._execute(statement)
        return [x[0] for x in res.fetchall()]

    def _add_chains_data(
        self, chains: lxc.ChainList[_CT], chain_type: int, level: int
    ) -> None:
        data = [(c.id, chain_type, level) for c in chains]
        self._insert("chains", data, omit_first_id=True)
        for i, children in enumerate(chains.iter_children(), start=1):
            if children:
                self._add_chains_data(children, chain_type, i)

    def _add_parents_data(self, chains: lxc.ChainList[_CT]) -> None:
        data = [(child.parent.id, child.id) for child in chains.collapse_children()]
        self._insert("parents", data)

    def add_chains(self, chains: abc.Sequence[_CT], load: bool = False):
        if not chains:
            return
        self._verify_chain_types(chains)
        chain_type = _CT_MAP[chains[0].__class__]
        chains = lxc.ChainList[_CT](self._filter_absent_chains(chains))
        self._add_chains_data(chains, chain_type, 0)
        self._add_parents_data(chains)
        if load:
            self._chains += tuple(chains)


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

    def _add_chains_data(
        self, chains: lxc.ChainList[_CT], chain_type: int, level: int
    ) -> None:
        # Add chain IDs
        data = [(c.id, chain_type, level) for c in chains]
        self._insert("chains", data, omit_first_id=True)
        # Add chain structure IDs
        data = list(
            chain.from_iterable(
                ((s.id, 2, level) for s in c.structures) for c in chains
            )
        )
        self._insert("chains", data, omit_first_id=True)
        # Add chain--structure relationships
        data = list(
            chain.from_iterable(((c.id, s.id) for s in c.structures) for c in chains)
        )
        self._insert("structures", data)

        for i, children in enumerate(chains.iter_children(), start=1):
            if children:
                self._add_chains_data(children, chain_type, i)


if __name__ == "__main__":
    raise RuntimeError
