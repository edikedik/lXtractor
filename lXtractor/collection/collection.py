import os
import pickle
import sqlite3
import typing as t
from collections import abc
from itertools import chain, tee, groupby
from os import PathLike
from pathlib import Path, PosixPath

import pandas as pd
from loguru import logger
from more_itertools import spy, unzip
from toolz import curry

import lXtractor.chain as lxc
from lXtractor.chain import make_str_tree
from lXtractor.util.typing import is_sequence_of
from lXtractor.variables.base import SequenceVariable, StructureVariable, LigandVariable
from lXtractor.variables.manager import CalcRes

_T = t.TypeVar("_T")
_CT = t.TypeVar("_CT", lxc.ChainSequence, lxc.ChainStructure, lxc.Chain)
_CT_MAP = {lxc.ChainSequence: 1, lxc.ChainStructure: 2, lxc.Chain: 3}
_VT_MAP = {SequenceVariable: 1, StructureVariable: 2, LigandVariable: 3}


def _verify_chain_types(objs: abc.Sequence[t.Any], ct: t.Type[_CT]) -> None:
    if not is_sequence_of(objs, ct):
        raise TypeError(f"All chains must have the same type {ct}")


def _create_chain_converters():
    dump_pickle = curry(pickle.dumps, protocol=5)
    for c in _CT_MAP:
        sqlite3.register_adapter(c, dump_pickle)
    sqlite3.register_converter("chain_data", pickle.loads)

    sqlite3.register_adapter(PosixPath, str)
    sqlite3.register_converter("path", lambda x: Path(x.decode("utf-8")))


def _get_var_type(v: t.Any):
    if isinstance(v, SequenceVariable):
        return 1
    if isinstance(v, StructureVariable):
        return 2
    if isinstance(v, LigandVariable):
        return 3
    raise TypeError(f"Unsupported variable type {v.type}")


def _make_placeholders(n: int) -> str:
    return ", ".join(["?"] * n)


class Collection(t.Generic[_CT]):
    def __init__(self, loc: str | PathLike = ":memory:", overwrite: bool = False):
        """
        :param loc: Location of the data collection. By default, will use RAM
            to store the data.
        :param overwrite: If `loc` is Path to an existing file, overwrite it.
        """
        self._loc = loc if loc == ":memory:" else Path(loc)
        if isinstance(self._loc, Path) and self._loc.exists() and overwrite:
            self._loc.unlink()
        _create_chain_converters()
        self._db = self._connect()
        self._setup()
        self._chains: lxc.ChainList[_CT] = lxc.ChainList([])

        logger.info(f"Initialized collection in {loc}.")

    @property
    def loaded(self) -> lxc.ChainList[_CT]:
        """
        :return: A chain list of currently loaded objects.
        """
        return self._chains

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._loc, detect_types=sqlite3.PARSE_DECLTYPES)

    def _close(self) -> None:
        self._db.close()

    def _column_names_for(self, table_name) -> list[str]:
        res = self._execute(f"SELECT * FROM {table_name}")
        return [x[0] for x in res.description]

    def _num_rows_for(self, table_name: str) -> int:
        res = self._execute(f"SELECT COUNT(*) from {table_name}")
        return res.fetchone()[0]

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
            logger.error(f"Failed to execute statement {statement}")
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
        on_conflict: str | None = None,
        conflict_action: str | None = None,
    ) -> tuple[str | None, abc.Iterable[abc.Sequence[_T]]]:
        data_size, data = self._peek_data_size(data)
        if data_size == -1:
            logger.warning(f"Attempting to insert empty `data` for {table_name}")
            return None, data
        placeholders = _make_placeholders(data_size)
        if columns is None:
            columns = self._column_names_for(table_name)
        if omit_first_id:
            columns = columns[1:]
        columns_str = ", ".join(columns)
        statement = f"INSERT INTO {table_name}({columns_str}) VALUES({placeholders})"

        # Handling ON CONFLICT clause
        if on_conflict and conflict_action:
            statement += f" ON CONFLICT({on_conflict}) DO {conflict_action}"

        if execute:
            init_rows = self._num_rows_for(table_name)
            self._execute(statement, data, many=True)
            res_rows = self._num_rows_for(table_name)
            logger.debug(f"Added {res_rows - init_rows} new rows to {table_name}.")
        return statement, data

    def _setup(self):
        make_chain_types = """ CREATE TABLE IF NOT EXISTS chain_types (
            type_id integer NOT NULL PRIMARY KEY AUTOINCREMENT,
            type_name TEXT NOT NULL UNIQUE
        ); """
        make_var_types = """ CREATE TABLE IF NOT EXISTS var_types (
            var_type_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            var_type_name TEXT NOT NULL UNIQUE
        ); """

        make_chains = """ CREATE TABLE IF NOT EXISTS chains (
            id TEXT PRIMARY KEY,
            chain_type INTEGER NOT NULL,
            level INTEGER NOT NULL,
            data chain_data,
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
            UNIQUE(chain_id_parent, chain_id_child)
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
            chain_path path NOT NULL,
            FOREIGN KEY (chain_id) REFERENCES chains (id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
            UNIQUE (chain_id, chain_path)
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
            if self._num_rows_for(table_name) == 0:
                self._insert(table_name, data, omit_first_id=True)

    def list_tables(self) -> list[str]:
        """
        :return: An unsorted list of table names in this collection.
        """
        res = self._execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [x[0] for x in res.fetchall()]

    @staticmethod
    def list_chain_type_codes() -> list[tuple[t.Type, int]]:
        """
        :return: A list of tuples (chain_type, chain_code) where chain_code is
            an integer associated with a chain_type in lXtractor data
            collections.
        """
        return list(_CT_MAP.items())

    def get_table(
        self, name: str, as_df: bool = False, where: str | None = None, **kwargs
    ) -> list[str] | pd.DataFrame | abc.Iterator[pd.DataFrame]:
        """
        Get a table from the data collection.

        >>> col = Collection()
        >>> col.get_table("chains")
        []
        >>> c = lxc.ChainSequence.from_string("AAA", name="X")
        >>> col.add([c])
        >>> df = col.get_table("chains", as_df=True)
        >>> df.id.iloc[0] == c.id
        True

        Note that the chain object is automatically loaded:

        >>> c_in_db = df.data.iloc[0]
        >>> c == c_in_db
        True

        :param name: Table name. See :meth:`list_tables` for available tables.
        :param as_df: Convert the table to a pandas dataframe.
        :param where: Specify conditions to filter the data. Do not add
            "WHERE" keyword.
        :param kwargs: Passed to pandas :func:`read_sql`. Used only if `as_df`
            is ``True``.
        :return: The requested table.
        """
        statement = f"SELECT * FROM {name}"
        if where:
            statement += f" WHERE {where}"
        if as_df:
            return pd.read_sql(statement, self._db, **kwargs)
        return self._execute(statement).fetchall()

    def get_ids(
        self, level: int | None = None, chain_type: str | int | None = None
    ) -> list[str]:
        """
        Get a list of chain identifiers currently stored in this collection.

        :param level: The optional topological level in the ancestral tree.
            ``0`` corresponds to "root" objects that don't have any parents,
            ``1`` corresponds to "child" objects that have a parent with the
            level ``0``, and so on.
        :param chain_type: The chain type encoded as integer.
            See :meth:`list_chain_type_codes` for related correspondence.
        :return: A list of chain identifiers.
        """
        statement = "SELECT id FROM chains"
        if level is not None:
            statement += f" WHERE level = {level}"
        if chain_type is not None:
            statement += f" AND chain_type = {chain_type}"
        res = self._execute(statement)
        return [x[0] for x in res.fetchall()]

    def get_children_of(
        self, ids: abc.Sequence[str]
    ) -> abc.Generator[list[str], None, None]:
        """
        Get a generator over lists of children corresponding to the given ids.

        >>> col = Collection()
        >>> c = lxc.ChainSequence.from_string("")

        :param ids: A list of chain identifiers.
        :return: A generator over lists of children, where each list corresponds
            to an identifier in `ids`. If an identifier is missing or doesn't
            have associated children, yields empty list.
        """
        placeholders = _make_placeholders(len(ids))
        statement = f"SELECT * from parents WHERE chain_id_parent IN ({placeholders})"
        res = self._execute(statement, ids).fetchall()
        for query in ids:
            yield [child_id for parent_id, child_id in res if parent_id == query]

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
        # Temporarily cleanup any existing children
        id2children = {c.id: c.children for c in chains}
        for c in chains:
            c.children = lxc.ChainList([])
        # Insert the data
        data = ((c.id, chain_type, level, c) for c in chains)
        self._insert("chains", data, on_conflict="id", conflict_action="NOTHING")
        # Restore the children
        for c in chains:
            c.children = id2children[c.id]

    def _add_chains_data(self, chains: lxc.ChainList[_CT], chain_type: int) -> None:
        for i, _chains in enumerate((chains, *chains.iter_children()), start=0):
            if _chains:
                self._insert_chains_data(_chains, chain_type, i)

    def _add_parents_data(self, chains: lxc.ChainList[_CT]) -> None:
        data = ((child.parent.id, child.id) for child in chains.collapse_children())
        self._insert(
            "parents",
            data,
            on_conflict="chain_id_parent, chain_id_child",
            conflict_action="NOTHING",
        )

    def _add_chain_objects(self, chains: lxc.ChainList[_CT]) -> None:
        all_chains = chains + chains.collapse_children()
        id2children = {c.id: c.children for c in all_chains}
        for c in all_chains:
            c.children = lxc.ChainList([])
        statement = "UPDATE chains SET data=? WHERE id=?"
        data = ((c, c.id) for c in all_chains)
        self._execute(statement, data, many=True)
        for c in all_chains:
            c.children = id2children[c.id]
        logger.debug(f"Added {len(chains)} serialized chain objects.")

    def add(self, chains: abc.Sequence[_CT], load: bool = False) -> None:
        """
        Add chains to this collection.

        :param chains: A list of chains to add.
        :param load: Add chains to a list of currently loaded chains accessible
            via :meth:`loaded`.
        """
        if not chains:
            return
        if not isinstance(chains, lxc.ChainList):
            chains = lxc.ChainList(chains)

        self._verify_chain_types(chains)
        chain_type = _CT_MAP[chains[0].__class__]
        # chains = lxc.ChainList[_CT](self._filter_absent_chains(chains))
        logger.debug(f"Adding {len(chains)} chains of type {chain_type}.")

        self._add_chains_data(chains, chain_type)
        self._add_parents_data(chains)
        # self._add_chain_objects(chains)
        if load:
            self._chains += chains
            logger.debug(
                f"Loaded {len(chains)} more chains. "
                f"Total loaded: {len(self._chains)}."
            )

    def _recover_structures(self, chains: lxc.ChainList[_CT]) -> None:
        # Valid only for MappingCollection
        pass

    def _get_all_children(self, chains: lxc.ChainList[_CT]) -> abc.Sequence[str]:
        max_level = self._execute("SELECT MAX(level) from chains").fetchone()[0]
        children = []
        ids = chains.ids
        for _ in range(max_level):
            tree_level = list(chain.from_iterable(self.get_children_of(ids)))
            if len(tree_level) == 0:
                break
            children += tree_level
            ids = tree_level
        return list(set(children))

    def _recover_children(self, chains: lxc.ChainList[_CT]) -> None:
        if not chains:
            return None

        chain_type = _CT_MAP[chains[0].__class__]
        child_ids = self._get_all_children(chains)
        _chains = chains.filter(lambda x: x.id in child_ids or x.id in chains.ids)
        absent = [x for x in child_ids if x not in _chains.ids]
        if absent:
            _chains += self.load(
                chain_type,
                ids=absent,
                keep=False,
                recover_tree=False,
                load_structures=False,
            )
        make_str_tree(_chains, connect=True)

    def clean_loaded(self):
        """
        Clean currently loaded chain objects from :meth:`loaded`.
        """
        self._chains = lxc.ChainList([])

    def load(
        self,
        chain_type: int,
        level: int = 0,
        ids: abc.Sequence[str] | None = None,
        keep: bool = False,
        recover_tree: bool = True,
        load_structures: bool = True,
    ) -> lxc.ChainList:
        # TODO: add "recover" call if `recover_tree=False` and update docs

        """
        Load chains into RAM and return.

        :param chain_type: An integer-encoded chain type. See
            :meth:`list_chain_type_codes` for related correspondence.
        :param level: An optional topological level in the ancestral tree.
        :param ids: A list of target IDs, which chains should be loaded.
        :param keep: Keep loaded chains in :meth:`loaded`. Note that for this
            to work, chains loaded here must be of the same type.
        :param recover_tree: Recover parent-child relationships for loaded
            chains. This will populate ``parent`` and ``children`` attributes
            and also load any children into memory, too. This should be set
            to ``True``; otherwise, loaded chains that initially had some
            parent will have different IDs.
        :param load_structures: Load structures associated with a chain
            sequence. Valid only for :class:`MappingCollection`.
        :return: A chain list of loaded chain objects.
        """
        params = (chain_type,)
        statement = "SELECT data FROM chains WHERE chain_type=?"
        if level:
            params += (level,)
            statement += " AND level=?"
        if ids is not None:
            if len(ids) == 0:
                return lxc.ChainList([])
            params += tuple(ids)
            placeholders = _make_placeholders(len(ids))
            statement += f" AND id IN ({placeholders})"
        res = self._execute(statement, params)
        chains = lxc.ChainList(x[0] for x in res)
        if recover_tree:
            self._recover_children(chains)
        if load_structures:
            self._recover_structures(chains)
        if keep:
            self._chains += chains
        logger.debug(f"Loaded {len(chains)} chains from the database.")
        return chains

    def unload(self, chains: abc.Sequence[_CT] | abc.Sequence[str]) -> None:
        """
        Remove some chains from currently :meth:`loaded`.

        :param chains: A sequence of chain objects or their identifiers.
        """
        if not chains:
            return
        ids = chains if isinstance(chains[0], str) else lxc.ChainList(chains).ids
        self._chains = self.loaded.filter(lambda x: x.id not in ids)
        logger.debug(f"Unloaded {len(chains)} chains from the database.")

    def remove(
        self,
        targets: abc.Sequence[_CT] | abc.Sequence[str],
        table: str = "chains",
        column: str = "id",
    ) -> None:
        """
        A general-purpose interface for removing data. It will remove rows from
        a table in this collection if they contain `targets` in a specified
        `column`. By default, removes chains from the collection. Removing a
        chain from the ``chains`` table will also remove any associated data
        from other tables and :meth:`unload` it, but not vice versa.

        :param targets: A sequence of chain objects or values to remove.
        :param table: Table name to remove the data from. See :meth:`list_tables`
            for available tables.
        :param column: A column name to remove the data from.
        """
        if not targets:
            return
        if isinstance(targets[0], (lxc.Chain, lxc.ChainSequence, lxc.ChainStructure)):
            cl = lxc.ChainList(targets)
            ids = cl.ids + cl.collapse_children().ids
            if isinstance(cl[0], lxc.Chain):
                ids += cl.structures.ids + cl.collapse_children().structures.ids
        else:
            ids = targets
        ids = [(x,) for x in ids]
        self._execute(f"DELETE FROM {table} WHERE {column}=?", ids, many=True)
        self.unload(targets)
        logger.debug(f"Removed data for {len(ids)} chains.")

    def add_vs(
        self,
        vs: abc.Iterable[CalcRes],
        miscalculated: bool = False,
    ) -> None:
        """
        Add variables' calculation results.

        :param vs: An iterable of over tuples produced by
            :meth:`lXtractor.variables.manager.Manager.calculate`.
        :param miscalculated: If ``True``, adds only successfully calculated
            variables.
        """
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
                _get_var_type(x[1]),  # Variable type
                str(x[1].rtype.__name__),  # Return type
            )
            for x in vs
        )
        data = list(data)
        self._insert("variables", data)
        logger.debug(f"Inserted {len(data)} variables.")

    def _expand_structures(self, chain_path: Path) -> abc.Iterable[Path]:
        """The method is only relevant for :class:`MappingCollection`, where
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
        """
        Link chains to existing paths in the filesystem. This method will
        automatically find and link any embedded child segments and structures.

        :param paths: An iterable over paths to stored chain objects.
        """
        existing_ids = self.get_ids()

        paths = list(filter(lambda x: x.exists() and x.is_dir(), map(Path, paths)))

        paths_children = chain.from_iterable(map(self._expand_children, paths))
        paths_structures = chain.from_iterable(
            map(self._expand_structures, chain(paths, paths_children))
        )
        paths_to_add = filter(
            lambda x: x.name in existing_ids,
            chain(paths, paths_children, paths_structures),
        )

        data = ((x.name, x) for x in paths_to_add)
        self._insert(
            "paths",
            data,
            on_conflict="chain_id",
            conflict_action="UPDATE SET chain_path=excluded.chain_path",
        )

    def _update(
        self,
        table: str,
        target_cols: abc.Iterable[str],
        target_values: abc.Iterable[tuple],
        condition_cols: abc.Iterable[str],
        condition_values: abc.Iterable[tuple],
    ):
        setters = ", ".join(f"{name}=?" for name in target_cols)
        conditions = " AND ".join(f"{name}=?" for name in condition_cols)
        statement = f"UPDATE {table} SET {setters} WHERE {conditions}"
        data = (target + cond for target, cond in zip(target_values, condition_values))
        self._execute(statement, data, many=True)

    def update_parents(self, values: abc.Iterable[tuple[str, str]]) -> None:
        """
        Update parent-child relationships.

        :param values: Iterable of (parent_id, child_id).
        """
        targets, conditions = map(lambda xs: ((x,) for x in xs), unzip(values))
        self._update(
            "parents", ["chain_id_parent"], targets, ["chain_id_child"], conditions
        )

    def update_variables(
        self, values: abc.Iterable[tuple[str, str, str]], set_calculated: bool = True
    ) -> None:
        """
        Update variable calculation results.

        :param values: Iterable of (chain_id, variable_id, variable_value).
        :param set_calculated: Set the flag that a variable is calculated to
            ``True``.
        """
        vs1, vs2 = tee(values)
        conditions = (x[:2] for x in vs1)
        if set_calculated:
            target_cols = ["variable_calculated", "variable_value"]
            targets = ((True, x[-1]) for x in vs2)
        else:
            target_cols = ["variable_value"]
            targets = (x[2:] for x in vs2)
        targets = list(targets)
        self._update(
            "variables",
            target_cols,
            targets,
            ["chain_id", "variable_id"],
            conditions,
        )


class SequenceCollection(Collection[lxc.ChainSequence]):
    def _verify_chain_types(self, chains: abc.Sequence[t.Any]) -> None:
        _verify_chain_types(chains, lxc.ChainSequence)


class StructureCollection(Collection[lxc.ChainStructure]):
    def _verify_chain_types(self, chains: abc.Sequence[t.Any]) -> None:
        _verify_chain_types(chains, lxc.ChainStructure)


class MappingCollection(Collection[lxc.Chain]):
    def __init__(self, loc: str | PathLike = ":memory:", overwrite: bool = False):
        super().__init__(loc, overwrite)
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
            UNIQUE(chain_id, structure_id)
        ); """
        self._execute(make_structures)

    def _verify_chain_types(self, chains: abc.Sequence[t.Any]) -> None:
        _verify_chain_types(chains, lxc.Chain)

    def _insert_chains_data(
        self, chains: lxc.ChainList[_CT], chain_type: int, level: int
    ) -> None:
        id2children = {c.id: c.children for c in chains}
        id2structures = {c.id: c.structures for c in chains}
        structures = chains.structures

        # Temporarily cleanup children and structures to avoid storing the same
        # data multiple times
        for c in chains:
            c.children = lxc.ChainList([])
            c.structures = lxc.ChainList([])

        data = chain(
            ((c.id, chain_type, level, c) for c in chains),
            ((s.id, 2, level, s) for s in structures),
        )

        # Insert new chains
        self._insert("chains", data, on_conflict="id", conflict_action="NOTHING")

        # Populate back children and structures
        for c in chains:
            c.children = id2children[c.id]
            c.structures = id2structures[c.id]

        # Insert chain-structure relationships
        data = chain.from_iterable(((c.id, s.id) for s in c.structures) for c in chains)
        self._insert(
            "structures",
            data,
            on_conflict="chain_id, structure_id",
            conflict_action="NOTHING",
        )

    def _expand_structures(self, chain_path: Path) -> abc.Iterator[Path]:
        for root, dirs, _ in os.walk(chain_path):
            root = Path(root)
            if root.name == "structures":
                yield from (root / x for x in dirs)

    def _recover_structures(self, chains: lxc.ChainList[_CT]) -> None:
        if not chains:
            return None

        id2chain = {c.id: c for c in chains + chains.collapse_children()}
        placeholders = _make_placeholders(len(id2chain))
        statement = f"SELECT * FROM structures where chain_id in ({placeholders})"
        res = self._execute(statement, list(id2chain)).fetchall()
        if not res:
            return None

        for g, gg in groupby(res, lambda x: x[0]):
            structures = self.load(
                2,
                ids=[x[1] for x in gg],
                keep=False,
                recover_tree=False,
                load_structures=False,
            )
            id2chain[g].structures = lxc.ChainList(structures)


if __name__ == "__main__":
    raise RuntimeError
