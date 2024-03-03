import tempfile
from pathlib import Path

import pytest
from more_itertools import consume, unique_everseen

import lXtractor.chain as lxc
from lXtractor.chain import ChainIO
from lXtractor.collection import SeqCollectionConstructor, StrCollectionConstructor
from lXtractor.collection import (
    SequenceCollection,
    MappingCollection,
    StructureCollection,
)
from lXtractor.collection.collection import Collection
from lXtractor.collection.constructor import MapCollectionConstructor
from lXtractor.collection.support import (
    ConstructorConfig,
    StrItem,
    SeqItem,
    MapItem,
    SeqItemList,
    StrItemList,
    MapItemList,
)
from lXtractor.core import Alignment
from lXtractor.core.exceptions import MissingData, ConfigError, FormatError
from lXtractor.ext import PyHMMer
from lXtractor.variables import SeqEl
from test.common import TestError, DATA, SEQUENCES

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


def get_all_ids(chains: lxc.ChainList, nested_structures=True):
    if not chains:
        return []
    if isinstance(chains[0], lxc.Chain):
        ids = chains.ids + chains.structures.ids + chains.collapse_children().ids
        if nested_structures:
            ids += chains.collapse_children().structures.ids
            ids += chains.structures.collapse_children().ids
        return list(unique_everseen(ids))
    return chains.ids + chains.collapse_children().ids


def iter_parent_child_ids(chains: lxc.ChainList):
    for child in chains.collapse_children():
        yield child.parent.id, child.id
    if isinstance(chains[0], lxc.Chain):
        for child in chains.structures.collapse_children():
            yield child.parent.id, child.id


@pytest.fixture()
def chain_structures(simple_chain_structure) -> lxc.ChainList[lxc.ChainStructure]:
    c = simple_chain_structure.spawn_child(1, 50)
    c.spawn_child(30, 40)
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


@pytest.fixture()
def chains_child_parent_maps(chains):
    base_chains = chains.collapse()
    structures = chains.structures.collapse()
    c2parent, c2children = {}, {}

    for c in base_chains:
        c2parent[c] = c.parent
        c2children[c] = c.children
    for c in structures:
        c2parent[c] = c.parent
        c2children[c] = c.children

    return c2parent, c2children


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

    df_chains = col.get_table("chains", as_df=True)
    added_ids = list(df_chains.id)

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

    # Chains raw data is stored
    assert all(
        isinstance(c, (lxc.ChainSequence, lxc.ChainStructure, lxc.Chain))
        for c in df_chains.data
    )

    # Raw stored chains do not have parent/children unless explicitly loaded
    for c in df_chains.data:
        assert len(c.children) == 0
        assert c.parent is None
        if ct is MapCollectionConstructor:
            assert len(c.structures) == 0


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
        chain_ids = cs.collapse().ids
        if cls is MappingCollection:
            chain_ids += cs.collapse().structures.ids
        assert added_ids == set(chain_ids)

    # 4. Test for updating behavior
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = list(io.write((cs[0],), Path(tmpdir)))
        col.link(paths)
        df = col.get_table("paths", as_df=True)
        sub = df.loc[df.chain_id == cs[0].id, "chain_path"]
        assert len(sub) == 1
        path = sub[0]
        assert path.exists() and path == paths[0]


def test_load_strs(chain_structures):
    col = StructureCollection()
    col.add(chain_structures)

    chains_loaded = col.load(2, 0)
    assert chain_structures.collapse() == chains_loaded.collapse()


def test_load_chains(chains):
    col = MappingCollection()
    col.add(chains)

    chains_loaded = col.load(3, 0)

    assert chains == chains_loaded
    all_str = chains.collapse().structures.collapse().drop_duplicates().sort()
    all_str_loaded = (
        chains_loaded.collapse().structures.collapse().drop_duplicates().sort()
    )
    assert all_str == all_str_loaded


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


@pytest.mark.parametrize(
    "item_type,inp_chain,exp_items",
    [
        (SeqItem, lxc.ChainSequence.from_string("A", name="X"), [SeqItem("X")]),
        (StrItem, lxc.ChainStructure(None, "A", "X"), [StrItem("X", "A")]),
        (
            MapItem,
            lxc.Chain(
                lxc.ChainSequence.from_string("A", name="X"),
                [
                    lxc.ChainStructure(None, "A", "X"),
                    lxc.ChainStructure(None, "B", "X"),
                ],
            ),
            [
                MapItem(SeqItem("X"), StrItem("X", "A")),
                MapItem(SeqItem("X"), StrItem("X", "B")),
            ],
        ),
    ],
)
def test_item_from_chain(item_type, inp_chain, exp_items):
    assert list(item_type.from_chain(inp_chain)) == exp_items


@pytest.mark.parametrize(
    "item_type,inp_str,exp_items",
    [
        (SeqItem, "X", [SeqItem("X")]),
        (
            StrItem,
            "1ABC:A,B,C",
            [StrItem("1ABC", "A"), StrItem("1ABC", "B"), StrItem("1ABC", "C")],
        ),
        (
            MapItem,
            "X=>1ABC:A;2ABC:A,B",
            [
                MapItem(SeqItem("X"), StrItem("1ABC", "A")),
                MapItem(SeqItem("X"), StrItem("2ABC", "A")),
                MapItem(SeqItem("X"), StrItem("2ABC", "B")),
            ],
        ),
    ],
)
def test_item_from_str(item_type, inp_str, exp_items):
    assert list(item_type.from_str(inp_str)) == exp_items


@pytest.mark.parametrize(
    "item_type,inp,exp_items",
    [
        (StrItem, ("1ABC", ["A", "B"]), [StrItem("1ABC", "A"), StrItem("1ABC", "B")]),
        (MapItem, ("X", "1ABC:A"), [MapItem(SeqItem("X"), StrItem("1ABC", "A"))]),
        (
            MapItem,
            ("X", ["1ABC:A", "2ABC:A"]),
            [
                MapItem(SeqItem("X"), StrItem("1ABC", "A")),
                MapItem(SeqItem("X"), StrItem("2ABC", "A")),
            ],
        ),
        (
            MapItem,
            ("X", [("2ABC", ["A", "B"])]),
            [
                MapItem(SeqItem("X"), StrItem("2ABC", "A")),
                MapItem(SeqItem("X"), StrItem("2ABC", "B")),
            ],
        ),
    ],
)
def test_item_from_tuple(item_type, inp, exp_items):
    assert list(item_type.from_tuple(inp)) == exp_items


@pytest.mark.parametrize(
    "itl,expected",
    [
        (SeqItemList([SeqItem("S1")]), ["S1"]),
        (StrItemList([StrItem("S1", "A"), StrItem("S1", "B")]), ["S1:A,B"]),
        (
            MapItemList(
                [
                    MapItem(SeqItem("X"), StrItem("S1", "A")),
                    MapItem(SeqItem("X"), StrItem("S1", "B")),
                    MapItem(SeqItem("Y"), StrItem("S1", "A")),
                ]
            ),
            ["X=>S1:A,B", "Y=>S1:A"],
        ),
    ],
)
def test_item_list_as_strings(itl, expected):
    assert list(itl.as_strings()) == expected


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


def make_config(base: Path, source, refs, local=False):
    if local:
        dirs = [
            base / "output",
            DATA / "sequences",
            DATA / "structures",
        ]
    else:
        dirs = [base / x for x in ("output", "sequences", "structures")]
    kws = dict(
        source=source,
        out_dir=dirs[0],
        seq_dir=dirs[1],
        str_dir=dirs[2],
        references=refs,
        PDB_kwargs=dict(verbose=True),
        AF2_kwargs=dict(verbose=True),
        write_batches=True,
        debug=True,
        max_proc=1,
    )
    if source.lower() in ("af", "af2", "alphafold"):
        kws["str_fmt"] = "cif"
    if local:
        kws["source"] = "local"
        kws["default_chain"] = "A"

    return ConstructorConfig(**kws), dirs


@pytest.mark.parametrize(
    "inp",
    [
        ("UniProt", SeqCollectionConstructor),
        ("PDB", StrCollectionConstructor),
        ("AF2", StrCollectionConstructor),
        ("SIFTS", MapCollectionConstructor),
        ("INVALID", SeqCollectionConstructor),
    ],
)
@pytest.mark.parametrize("references", [()])
def test_constructor_setup(inp, references, tmp_path):
    source, constr_type = inp
    config, dirs = make_config(tmp_path, source, references)
    if source == "INVALID":
        with pytest.raises(ConfigError):
            constr_type(config)
    else:
        constructor = constr_type(config)
        assert all(x.exists() for x in dirs)
        assert isinstance(constructor.collection, Collection)


@pytest.mark.parametrize("source,const_type", [("UniProt", SeqCollectionConstructor)])
@pytest.mark.parametrize(
    "ref",
    [
        DATA / "Pkinase.hmm",
        DATA / "void.hmm",
        DATA / "void.hmm.gz",
        SEQUENCES / "fasta" / "simple.fasta",
        PyHMMer(DATA / "Pkinase.hmm"),
        Alignment([("REF_SEQ", "KAL"), ("s2", "KKL")]),
        lxc.ChainSequence.from_string("KAL", name="REF_SEQ"),
        ("REF", SEQUENCES / "fasta" / "simple.fasta"),
        ("ALN", Alignment([("REF_SEQ", "KAL"), ("s2", "KKL")])),
        ("REF", "KAL"),
        ("REF", lxc.ChainSequence.from_string("KAL", name="REF_SEQ")),
        ("INVALID", None),
    ],
)
def test_setup_references(source, const_type, ref, tmp_path):
    config, dirs = make_config(tmp_path, source, [ref])

    if isinstance(ref, tuple) and ref[0] == "INVALID":
        with pytest.raises(TypeError):
            const_type(config)
    elif isinstance(ref, Path) and ref.suffix not in (".hmm", ".fasta"):
        with pytest.raises(NameError):
            const_type(config)
    elif isinstance(ref, Path) and not ref.exists():
        with pytest.raises(FileNotFoundError):
            const_type(config)
    else:
        constructor = const_type(config)
        assert all(isinstance(x, PyHMMer) for x in constructor.references)
        written_refs = {x.stem for x in constructor.paths.references.glob("*.hmm")}
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

    config, dirs = make_config(tmp_path, "local", ())
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


@pytest.mark.parametrize(
    "ct,inputs,valid,exp_items",
    [
        (
            SeqCollectionConstructor,
            [SeqItem("S"), "s"],
            True,
            [SeqItem("S"), SeqItem("s")],
        ),
        (
            SeqCollectionConstructor,
            [StrItem("S", "A")],
            False,
            [],
        ),
        (
            StrCollectionConstructor,
            [StrItem("S", "A"), "s:A,B", ("ss", ["A"])],
            True,
            [
                StrItem("S", "A"),
                StrItem("s", "A"),
                StrItem("s", "B"),
                StrItem("ss", "A"),
            ],
        ),
        (StrCollectionConstructor, ["S"], False, []),
        (StrCollectionConstructor, [("S", "A")], False, []),
        (
            MapCollectionConstructor,
            [MapItem(SeqItem("S"), StrItem("s", "A")), "X=>s:A", ("Y", "s:A")],
            True,
            [
                MapItem(SeqItem("S"), StrItem("s", "A")),
                MapItem(SeqItem("X"), StrItem("s", "A")),
                MapItem(SeqItem("Y"), StrItem("s", "A")),
            ],
        ),
        (MapCollectionConstructor, [("S", "A:B", "C:D")], False, []),
    ],
)
def test_inp_parsing(ct, inputs, valid, exp_items, tmp_path):
    if ct is SeqCollectionConstructor:
        source = "uniprot"
    elif ct is StrCollectionConstructor:
        source = "pdb"
    else:
        source = "sifts"
    config, _ = make_config(tmp_path, source, [])
    constructor = ct(config)
    if valid:
        assert list(constructor.parse_inputs(inputs)) == exp_items
    else:
        with pytest.raises(FormatError):
            list(constructor.parse_inputs(inputs))


PKP = DATA / "Pkinase.hmm"
TEST_BATCHES = [
    (SeqCollectionConstructor, "UniProt", ["P12931", "Q16644"], [PKP]),
    (StrCollectionConstructor, "PDB", ["2SRC:A", "2OIQ:A"], [PKP]),
    (StrCollectionConstructor, "AF", ["P12931", "Q16644"], [PKP]),
    (MapCollectionConstructor, "SIFTS", ["P12931=>2SRC:A;2OIQ:A,B"], [PKP]),
]


@pytest.mark.parametrize("ct,source,ids,refs", TEST_BATCHES)
@pytest.mark.parametrize("local", [True, False])
def test_run_batch(ct, source, ids, refs, local, tmp_path):
    config, _ = make_config(tmp_path, source, refs, local)

    if ct is MapCollectionConstructor:
        config["references_annotate_kw"] = dict(str_map_from="map_canonical")

    constructor = ct(config)
    itl = constructor.item_list_type(constructor.parse_inputs(ids))
    res = constructor.run_batch(itl)
    assert isinstance(res, lxc.ChainList)
    if ct is MapCollectionConstructor:
        assert len(res) == len(ids)
        assert len(res.collapse_children()) == len(ids)
        # 1 seq and 1 seq child, 3 str and 3 str children
        assert len(constructor.collection.get_ids()) == 8
    else:
        assert len(res) == len(itl)
        assert len(res.collapse_children()) == len(itl)
        assert len(constructor.collection.get_ids()) == len(itl) * 2

    assert len(constructor.history) == 0


@pytest.mark.parametrize(
    "ct", [SeqCollectionConstructor, StrCollectionConstructor, MapCollectionConstructor]
)
def test_run_empty_batch(ct, tmp_path):
    config, _ = make_config(tmp_path, "", [PKP], True)
    constructor = ct(config)
    res = constructor.run_batch(constructor.item_list_type())
    assert isinstance(res, lxc.ChainList)
    assert len(res) == 0


@pytest.mark.parametrize("ct,source,ids,refs", TEST_BATCHES[:-1])
@pytest.mark.parametrize("local", [True])
def test_run(ct, source, ids, refs, local, tmp_path):
    config, dirs = make_config(tmp_path, source, refs, local)
    config["batch_size"] = 1
    config["keep_chains"] = True

    constructor = ct(config)
    for batch in constructor.run(constructor.parse_inputs(ids)):
        assert len(batch.items_in) == 1

    hist = constructor.history
    assert hist.last_step == 2
    assert len(hist.items_done()) == len(ids)
    assert len(hist.items_tried()) == len(ids)

    assert len(hist.items_missed()) == 0
    assert len(hist.items_failed()) == 0

    assert isinstance(hist.join_chains(), lxc.ChainList)

    assert len(constructor.collection.get_ids()) == len(ids) * 2


@pytest.mark.parametrize("ct,source,ids,refs", TEST_BATCHES[:-1])
@pytest.mark.parametrize("local", [True])
def test_fail_resume(ct, source, ids, refs, local, tmp_path):
    def bad_fn(_):
        # I always fail
        raise TestError()

    config, _ = make_config(tmp_path, source, refs, local)
    config["batch_size"] = 1
    config["parent_callback"] = bad_fn
    constructor = ct(config)

    items = list(constructor.parse_inputs(ids))

    # whatever the error, it's caught and the runtime exception is raised
    with pytest.raises(RuntimeError):
        next(constructor.run(items))

    assert constructor.last_failed_batch == items[:1]
    assert len(constructor.history) == 0
    assert len(constructor.collection.get_ids()) == 0

    # Remove callback from config and continue from the last failed batch
    config["parent_callback"] = None
    batches = list(constructor.resume_with(constructor.last_failed_batch))
    assert len(batches) == len(items)
    assert len(constructor.history) == len(items)
    assert len(constructor.collection.get_ids()) == len(items) * 2

    # run to completion without stopping on failed batches
    config["parent_callback"] = bad_fn
    constructor = ct(config)
    consume(constructor.run(items, stop_on_batch_failure=False))
    assert len(constructor.history) == 2
    assert len(list(constructor.history.items_failed())) == 2

    # However, the collection was previously constructed and remains unchanged
    assert len(constructor.collection.get_ids()) == len(items) * 2


def test_run_mapping(tmp_path):
    inputs = ["P12931=>2SRC:A;2OIQ:A,B"]
    config, _ = make_config(tmp_path, "SIFTS", [PKP], True)
    config["batch_size"] = 1
    config["references_annotate_kw"] = dict(str_map_from="map_canonical")
    config["keep_chains"] = True
    config["max_proc"] = 1

    constructor = MapCollectionConstructor(config)
    itl = constructor.parse_inputs(inputs)
    batches = list(constructor.run(itl))
    assert len(batches) == 3
    hist = constructor.history
    assert len(hist.items_done()) == len(batches)
    assert len(hist.join_chains()) == len(batches)

    # 1 seq and 1 seq child, 3 str and 3 str children
    assert len(constructor.collection.get_ids()) == 8
    # all paths were linked
    if config["write_batches"]:
        assert len(constructor.collection.get_table("paths", as_df=True)) == 8
