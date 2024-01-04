import tempfile
from pathlib import Path

import pytest

import lXtractor.chain as lxc
from lXtractor.chain import ChainIO
from lXtractor.collection import (
    Collection,
    SequenceCollection,
    ChainCollection,
    StructureCollection,
)
from lXtractor.variables import SeqEl

GET_TABLE_NAMES = """SELECT name FROM sqlite_master WHERE type='table';"""
TABLE_NAMES = (
    "chain_types",
    "var_types",
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


def get_all_ids(chains: lxc.ChainList, nested_structures=True):
    if not chains:
        return []
    if isinstance(chains[0], lxc.Chain):
        ids = (
            chains.ids
            + chains.structures.ids
            + chains.collapse_children().ids
        )
        print(ids)
        if nested_structures:
            ids += chains.collapse_children().structures.ids
        return ids
    return chains.ids + chains.collapse_children().ids


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

    # Cascading must clear all tables
    for table_name in ["chains", "parents", "variables", "structures"]:
        df = col.get_table(table_name, as_df=True)
        assert len(df) == 0


def test_add_vs(chain_sequences):
    col = SequenceCollection()
    col.add(chain_sequences)
    c = chain_sequences[0]
    cc = c.spawn_child(1, 2)
    inputs = [
        (c, SeqEl(1), True, "A"),
        (c, SeqEl(2), False, "Squeak"),
        (cc, SeqEl(1), True, "A"),
    ]
    col.vs_add(inputs)
    df = col.get_table("variables", as_df=True)
    assert len(df) == 1
    ids = set(df.chain_id)
    assert ids == {c.id}


@pytest.mark.parametrize("cls", COLLECTION_TYPES)
def test_link(cls, chain_sequences, chain_structures, chains):
    col = cls()

    if cls is SequenceCollection or cls is Collection:
        cs = chain_sequences
    elif cls is StructureCollection:
        cs = chain_structures
    else:
        cs = chains

    # 0. paths exists and is empty
    df = col.get_table("paths", as_df=True)
    assert len(df) == 0

    # 1. Adding nothing does nothing
    col.link(iter([]))
    df = col.get_table("paths", as_df=True)
    assert len(df) == 0

    # 2. Adding random path does nothing
    col.link((Path("tmp.txt"),))
    df = col.get_table("paths", as_df=True)
    assert len(df) == 0

    # 3. Adding existing chains
    io = ChainIO()
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = list(io.write(cs, Path(tmpdir), write_children=True))

        # 3.1 Chains were not added, so adding existing paths does nothing
        col.link(paths)
        df = col.get_table("paths", as_df=True)
        assert len(df) == 0

        # 3.2 Adding valid paths to existing chains works for all of them
        col.add(cs)
        col.link(paths)
        df = col.get_table("paths", as_df=True)
        assert len(df) > 0
        added_ids = set(df.chain_id)
        chain_ids = set(get_all_ids(cs, nested_structures=False))
        assert added_ids == chain_ids
