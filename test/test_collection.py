import tempfile
from pathlib import Path

import pytest

import lXtractor.chain as lxc
from lXtractor.collection import (
    Collection,
    SequenceCollection,
    ChainCollection,
    StructureCollection,
)

GET_TABLE_NAMES = """SELECT name FROM sqlite_master WHERE type='table';"""
TABLE_NAMES = (
    "chain_types",
    "var_types",
    "var_rtypes",
    "chains",
    "parents",
    "variables",
    "paths",
    "sqlite_sequence",
)
COLLECTION_TYPES = (
    Collection,
    SequenceCollection,
    StructureCollection,
    ChainCollection,
)


def get_all_ids(chains: lxc.ChainList):
    if not chains:
        return []
    if isinstance(chains[0], lxc.Chain):
        return (
            chains.ids
            + chains.structures.ids
            + chains.collapse_children().ids
            + chains.collapse_children().structures.ids
        )
    return [c.id for c in chains] + [c.id for c in chains.collapse_children()]


def iter_parent_child_ids(chains: lxc.ChainList):
    for child in chains.collapse_children():
        yield child.parent.id, child.id


@pytest.fixture()
def chain_structures(simple_chain_structure) -> lxc.ChainList[lxc.ChainStructure]:
    c = simple_chain_structure.spawn_child(1, 20)
    c.spawn_child(5, 10)
    return lxc.ChainList([simple_chain_structure])


@pytest.fixture()
def chain_sequences(chain_structures) -> lxc.ChainList[lxc.ChainSequence]:
    c = chain_structures[0].seq
    c.spawn_child(1, 20)
    return lxc.ChainList([c])


@pytest.fixture()
def chains(chain_structures) -> lxc.ChainList[lxc.Chain]:
    c = lxc.Chain(chain_structures[0].seq, structures=chain_structures)
    cc = c.spawn_child(1, 20)
    cc.spawn_child(1, 10)
    return lxc.ChainList([c])


@pytest.mark.parametrize("cls", COLLECTION_TYPES)
@pytest.mark.parametrize("loc", [":memory:", "file"])
def test_setup(cls, loc):
    if loc == ":memory:":
        col = cls(loc)
        handle = None
    else:
        handle = tempfile.NamedTemporaryFile("w")
        col = cls(Path(handle.name))
    table_names = set(TABLE_NAMES)
    if cls is ChainCollection:
        table_names.add("structures")
    with col._db as cur:
        names = cur.execute(GET_TABLE_NAMES)
        assert set(x[0] for x in names) == table_names

    if handle is not None:
        handle.close()


@pytest.mark.parametrize("cls", COLLECTION_TYPES)
def test_add_chains(cls, chain_sequences, chain_structures, chains):
    col = cls()

    if cls is SequenceCollection or cls is Collection:
        cs = chain_sequences
        ct = 1
    elif cls is StructureCollection:
        cs = chain_structures
        ct = 2
    else:
        cs = chains
        ct = 3

    ids = get_all_ids(cs)
    col.add(cs, load=True)

    added_ids = col.get_ids()

    assert set(added_ids) == set(ids)

    ids_level0 = col.get_ids(level=0, chain_type=ct)
    assert ids_level0 == [c.id for c in cs]

    # Test parent information incorporated correctly
    ids_exp = list(iter_parent_child_ids(cs))
    df = col.get_table("parents", as_df=True)
    ids = [(row.chain_id_parent, row.chain_id_child) for _, row in df.iterrows()]
    assert sorted(ids) == sorted(ids_exp)


def test_add_chains_structures(chains):
    # Test if structures are correctly added in chain collection
    col = ChainCollection()
    col.add(chains)
    df = col.get_table("structures", as_df=True)
    init_ids = set(chains.structures.ids + chains.collapse_children().structures.ids)
    assert init_ids == set(df["structure_id"])


def test_rm_chains(chains):
    col = ChainCollection()
    col.add(chains, load=True)
    assert len(col.loaded) == len(chains)
    col.remove(chains)
    assert len(col.loaded) == 0

    for table_name in ["chains", "parents", "variables", "structures"]:
        df = col.get_table(table_name, as_df=True)
        assert len(df) == 0
