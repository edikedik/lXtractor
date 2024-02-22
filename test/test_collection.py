import tempfile
from pathlib import Path

import pytest

import lXtractor.chain as lxc
from lXtractor.chain import ChainIO
from lXtractor.collection import SeqCollectionConstructor, StrCollectionConstructor
from lXtractor.collection import (
    SequenceCollection,
    MappingCollection,
    StructureCollection,
)
from lXtractor.collection.collection import Collection
from lXtractor.collection.support import (
    ConstructorConfig,
    BatchData,
    BatchesHistory,
)
from lXtractor.core import Alignment
from lXtractor.core.exceptions import MissingData
from lXtractor.ext import PyHMMer
from lXtractor.variables import SeqEl
from test.common import TestError

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
    SequenceCollection,
    StructureCollection,
    MappingCollection,
)
DATA = Path(__file__).parent / "data"


def get_all_ids(chains: lxc.ChainList, nested_structures=True):
    if not chains:
        return []
    if isinstance(chains[0], lxc.Chain):
        ids = chains.ids + chains.structures.ids + chains.collapse_children().ids
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
    if cls is MappingCollection:
        table_names.add("structures")
    assert table_names == set(col.list_tables())

    if loc == "file":
        assert isinstance(cls(Path(handle.name)), cls)

    if handle is not None:
        handle.close()


@pytest.mark.parametrize("cls", COLLECTION_TYPES)
def test_add_chains(cls, chain_sequences, chain_structures, chains):
    col = cls()

    if cls is SequenceCollection:
        cs = chain_sequences
        ct = 1
    elif cls is StructureCollection:
        cs = chain_structures
        ct = 2
    else:
        cs = chains
        ct = 3

    ids = get_all_ids(cs)
    col.add(cs)

    added_ids = col.get_ids()

    assert set(added_ids) == set(ids)

    ids_level0 = col.get_ids(level=0, chain_type=ct)
    assert ids_level0 == [c.id for c in cs]

    # Test parent information incorporated correctly
    ids_exp = list(iter_parent_child_ids(cs))
    df = col.get_table("parents", as_df=True)
    ids_par = [(row.chain_id_parent, row.chain_id_child) for _, row in df.iterrows()]
    assert sorted(ids_par) == sorted(ids_exp)

    # Adding the same chains second time does nothing
    col.add(cs)
    assert set(col.get_ids()) == set(ids)


def test_add_chains_structures(chains):
    # Test if structures are correctly added in chain collection
    col = MappingCollection()
    col.add(chains)
    df = col.get_table("structures", as_df=True)
    init_ids = set(chains.structures.ids + chains.collapse_children().structures.ids)
    assert init_ids == set(df["structure_id"])


def test_rm_chains(chains):
    col = MappingCollection()
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
    col.add_vs(inputs)
    df = col.get_table("variables", as_df=True)
    assert len(df) == 1
    ids = set(df.chain_id)
    assert ids == {c.id}


@pytest.mark.parametrize("cls", COLLECTION_TYPES)
def test_link(cls, chain_sequences, chain_structures, chains):
    col = cls()

    if cls is SequenceCollection:
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

    # 4. Test for updating behavior
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = list(io.write((cs[0],), Path(tmpdir)))
        col.link(paths)
        df = col.get_table("paths", as_df=True)
        sub = df.loc[df.chain_id == cs[0].id, "chain_path"]
        assert len(sub) == 1
        path = sub[0]
        assert path.exists() and path == paths[0]


def test_load(chains):
    col = MappingCollection()
    col.add(chains)
    df = col.get_table("chains", as_df=True)
    assert not df.data.isna().any()
    assert len(col.loaded) == 0

    # loading without level loads all chains
    loaded = col.load(3)
    assert loaded == chains + chains.collapse_children()
    # loading only the first level should load exactly one chain
    loaded = col.load(3, level=1)
    assert len(loaded) == 1
    assert loaded == next(chains.iter_children())


def test_update_parents(chains):
    col = MappingCollection()
    col.add(chains)
    p, c1, c2 = chains + chains.collapse_children()
    # initial c1 <- p, c2 <- c1
    col.update_parents([(p.id, "X")])
    upd = (p.id, c2.id)
    col.update_parents([upd])
    parents = col.get_table("parents")
    assert upd in parents


@pytest.mark.parametrize("set_calculated", [True, False])
def test_update_variables(chain_sequences, set_calculated):
    col = SequenceCollection()
    col.add(chain_sequences)
    c = chain_sequences[0]
    v1, v2 = SeqEl(1), SeqEl(2)
    vs_inp = [
        (c, v1, True, "A"),
        (c, v2, False, "Squeak"),
    ]
    col.add_vs(vs_inp, miscalculated=True)
    col.update_variables([(c.id, v2.id, "Peek")], set_calculated)
    df = col.get_table("variables", as_df=True)
    assert len(df) == 2
    row = df[df.variable_id == v2.id].iloc[0]
    assert row.variable_value == "Peek"
    assert row.variable_calculated == set_calculated


def test_constructor_config():
    cfg = ConstructorConfig(source="SIFTS")
    assert "references" in cfg.list_fields()
    assert "references" in cfg.list_missing_fields()
    assert cfg["source"] == "SIFTS"

    with pytest.raises(MissingData):
        cfg.validate()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "config.json"
        cfg.save(cfg_path)

        _cfg = ConstructorConfig(user_config_path=cfg_path)
        assert cfg == _cfg


def make_config(base: Path, source, refs, ids):
    dirs = [base / x for x in ("output", "sequences", "structures", "references")]
    kws = dict(
        source=source,
        out_dir=dirs[0],
        seq_dir=dirs[1],
        str_dir=dirs[2],
        ref_dir=dirs[3],
        references=refs,
        ids=ids,
        PDB_kwargs=dict(verbose=True),
        AF2_kwargs=dict(verbose=True),
    )
    if source.lower() in ("af", "af2"):
        kws["str_fmt"] = "cif"

    return ConstructorConfig(**kws), dirs


@pytest.mark.parametrize(
    "inp",
    [
        # ("SIFTS", "chain", "str"),
        ("UniProt", SeqCollectionConstructor),
        ("PDB", StrCollectionConstructor),
        ("AF2", StrCollectionConstructor),
    ],
)
@pytest.mark.parametrize("references", [()])
def test_constructor_setup(inp, references, tmp_path):
    source, constr_type = inp
    config, dirs = make_config(tmp_path, source, references, ())
    constructor = constr_type(config)
    assert all(x.exists() for x in dirs)
    assert isinstance(constructor.collection, Collection)


@pytest.mark.parametrize("source,const_type", [("UniProt", SeqCollectionConstructor)])
@pytest.mark.parametrize(
    "ref",
    [
        DATA / "Pkinase.hmm",
        DATA / "simple.fasta",
        PyHMMer(DATA / "Pkinase.hmm"),
        Alignment([("REF_SEQ", "KAL"), ("s2", "KKL")]),
        lxc.ChainSequence.from_string("KAL", name="REF_SEQ"),
        ("REF", DATA / "simple.fasta"),
        ("ALN", Alignment([("REF_SEQ", "KAL"), ("s2", "KKL")])),
        ("REF", "KAL"),
        ("REF", lxc.ChainSequence.from_string("KAL", name="REF_SEQ")),
        ("INVALID", None),
    ],
)
def test_setup_references(source, const_type, ref, tmp_path):
    config, dirs = make_config(tmp_path, source, [ref], ())

    if isinstance(ref, tuple) and ref[0] == "INVALID":
        with pytest.raises(TypeError):
            const_type(config)
    else:
        constructor = const_type(config)
        assert all(isinstance(x, PyHMMer) for x in constructor.references)
        written_refs = {x.stem for x in dirs[-1].glob("*.hmm")}
        ref_names = {r.hmm.name.decode("utf-8") for r in constructor.references}
        assert ref_names == written_refs


def test_callback_and_filter(tmp_path):
    def rename(x):
        x.name = "!"
        return x

    chains = lxc.ChainList(
        [
            lxc.ChainSequence.from_tuple(("?", "ABCDEG")),
            lxc.ChainSequence.from_tuple(("??", "ABC")),
        ]
    )
    chains[0].spawn_child(1, 3)
    chains[1].spawn_child(1, 2)

    config, dirs = make_config(tmp_path, "local", (), ())
    config["child_callback"] = rename
    config["child_filter"] = lambda x: len(x) > 2
    config["parent_callback"] = rename
    config["parent_filter"] = lambda x: len(x.children) > 0
    constructor = SeqCollectionConstructor(config)

    chains = constructor._callback_and_filter(chains)
    assert len(chains) == 1 and len(chains[0]) == 6
    assert len(chains.collapse_children()) == 1

    assert all(x.name == "!" for x in chains)
    assert all(x.name == "!" for x in chains.collapse_children())


TEST_BATCHES = [
    (
        SeqCollectionConstructor,
        "UniProt",
        ["P12931", "Q16644"],
        [DATA / "Pkinase.hmm"],
    ),
    (StrCollectionConstructor, "PDB", ["2SRC:A", "2OIQ:A"], [DATA / "Pkinase.hmm"]),
    (StrCollectionConstructor, "AF", ["P12931", "Q16644"], [DATA / "Pkinase.hmm"]),
]


@pytest.mark.parametrize("ct,source,ids,refs", TEST_BATCHES)
def test_run_batch(ct, source, ids, refs, tmp_path):
    config, dirs = make_config(tmp_path, source, refs, ())
    constructor = ct(config)
    res = constructor.run_batch(ids)
    assert isinstance(res, lxc.ChainList)
    assert len(res) == len(ids)
    assert len(res.collapse_children()) == len(ids)
    assert len(constructor.collection.get_ids()) == len(ids) * 2


@pytest.mark.parametrize("ct,source,ids,refs", TEST_BATCHES)
def test_run(ct, source, ids, refs, tmp_path):
    config, dirs = make_config(tmp_path, source, refs, ())
    config["batch_size"] = 1

    constructor = ct(config)
    for batch in constructor.run(ids):
        assert len(batch) == 1

    assert len(constructor.collection.get_ids()) == len(ids) * 2


@pytest.mark.parametrize("ct,source,ids,refs", TEST_BATCHES)
def test_fail_resume(ct, source, ids, refs, tmp_path):
    def bad_fn(_):
        # I always fail
        raise TestError()

    config, dirs = make_config(tmp_path, source, refs, ())
    config["batch_size"] = 1
    config["parent_callback"] = bad_fn
    constructor = ct(config)

    # whatever the error, it's caught and the runtime exception is raised
    with pytest.raises(RuntimeError):
        next(constructor.run(ids))

    assert constructor.last_failed_batch == ids[:1]
    assert len(constructor.history) == 0
    assert len(constructor.collection.get_ids()) == 0

    # Remove callback from config and continue from the last failed batch
    config["parent_callback"] = None
    batches = list(constructor.resume_with(constructor.last_failed_batch))
    assert len(batches) == len(ids)
    assert len(constructor.history) == len(ids)
    assert len(constructor.collection.get_ids()) == len(ids) * 2

    constructor = ct(config)
    batches = list(constructor.run(ids))
    assert len(batches) == len(ids)
    assert len(constructor.history) == len(ids)


@pytest.mark.parametrize(
    "inp_ids,out_ids,done_ids,missed_ids",
    [
        ((), (), (), ()),
        (("XXXX", "YYYY"), (), (), ("XXXX", "YYYY")),
        (
            ("1ABC:A,B", "2ABC:A"),
            ("1ABC:A|1-10", "1ABC:B|1-5"),
            ("1ABC:A,B",),
            ("2ABC:A",),
        ),
        (("1ABC:A,B", "2ABC:A"), ("1ABC:A|1-10",), ("1ABC:A",), ("2ABC:A", "1ABC:B")),
    ],
)
def test_batch_data(inp_ids, out_ids, done_ids, missed_ids):
    bd = BatchData(1, inp_ids, out_ids, None)
    hist = BatchesHistory([bd])
    assert hist.last_step() == 1
    assert set(hist.iter_tried()) == set(inp_ids)
    assert set(hist.iter_done()) == set(done_ids)
    assert set(hist.iter_missed()) == set(missed_ids)
