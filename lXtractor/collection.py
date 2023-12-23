import logging
import sqlite3
import typing as t
from collections import abc
from os import PathLike
from pathlib import Path

from lXtractor.core.exceptions import FormatError

LOGGER = logging.getLogger(__name__)


class Collection:
    def __init__(self, loc: str | PathLike):
        self._loc = loc if loc == ":memory:" else Path(loc)
        self._db = self._connect()
        self._setup()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._loc)

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

    def _column_names_for(self, table_name) -> list[str]:
        res = self._execute(f"SELECT * FROM {table_name}")
        return [x[0] for x in res.description]

    def _insert(
        self,
        table_name: str,
        data: abc.Sequence[abc.Sequence[t.Any]],
        columns: abc.Sequence[str] = None,
        omit_first_id: bool = False
    ):
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
            FOREIGN KEY(chain_type) REFERENCES chain_types (type_id)
        ); """
        make_parents = """ CREATE TABLE IF NOT EXISTS parents (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
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
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
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

        for statement in [
            make_chain_types,
            make_var_types,
            make_var_return_types,
            make_chains,
            make_parents,
            make_variables,
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
            self._insert(table_name, data, omit_first_id=True)


if __name__ == "__main__":
    raise RuntimeError
