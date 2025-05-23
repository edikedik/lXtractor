import pytest

from lXtractor.chain import ChainSequence, ChainList, Chain, ChainStructure
from lXtractor.chain import (
    list_ancestors,
    list_ancestors_names,
    make_filled,
    node_name,
    make_id_tree,
    recover,
)


@pytest.fixture
def chains():
    c13 = ChainSequence.from_string("ccc", start=1, end=3, name="C")
    c12 = c13.spawn_child(1, 2)
    c45 = ChainSequence.from_string("cc", start=4, end=5, name="C")
    c13.meta["id"] = "C|1-3<-(C|1-5)"
    c45.meta["id"] = "C|4-5<-(C|1-5)"
    return ChainList([c13, c12, c45])


def test_ancestors_list(chains, simple_chain_structure):
    par_list = list_ancestors(chains["C|1-2"].pop())
    assert len(par_list) == 1
    assert [x.id for x in par_list] == ["C|1-3"]
    c = simple_chain_structure.spawn_child(1, 2)
    par_list = list_ancestors(c)
    assert [x.id for x in par_list] == [str(simple_chain_structure)]


def test_list_ancestors_names(chains, simple_chain_structure):
    par_list = list_ancestors_names(chains["C|1-2"].pop())
    assert par_list == ["C|1-3"]
    par_list = list_ancestors_names("C|1-2<-(C|1-3<-(C|1-5))")
    assert par_list == ["C|1-3", "C|1-5"]
    c = simple_chain_structure.spawn_child(1, 2)
    par_list = list_ancestors_names(c)
    assert par_list == [simple_chain_structure.meta["id"]]


@pytest.mark.parametrize(
    "inp,is_valid,example_obj,expected",
    [
        ("C|1-5", True, ChainSequence.make_empty(), (5, True)),
        ("C|1-1", True, Chain.make_empty(), (1, True)),
        ("C-some-info|1-5", True, ChainStructure.make_empty(), (0, True)),
        ("C|1-5", True, ChainSequence, (5, False)),
        ("C|1-1", True, Chain, (1, False)),
        ("C-some-info|1-5", True, ChainStructure, (0, False)),
        ("C|0-0", True, ChainSequence, (0, False)),
        ("C|0-0", True, Chain, (0, False)),
        ("C-some-info|0-0", True, ChainStructure, (0, False)),
        ("C|1~5", False, None, None),
        ("C|0-1", False, None, None),
        ("C/1-5", False, None, None),
    ],
)
def test_make_filled(inp, is_valid, example_obj, expected):
    if is_valid:
        obj = make_filled(inp, example_obj)
        size = len(obj.seq) if isinstance(obj, Chain) else len(obj)
        assert (size, isinstance(obj, type(example_obj))) == expected
    else:
        with pytest.raises(Exception):
            _ = make_filled(inp, example_obj)


# @pytest.mark.skip("Object trees are equivalent to str trees due to redefined hashing")
# def test_make_obj_tree(chains, simple_chain_structure):
#     tree = make_obj_tree(chains, connect=False)
#     assert len(tree.nodes) == 4
#     assert tree.is_directed()
#     diff = set(map(node_name, tree.nodes)) - set(map(node_name, chains))
#     assert diff == {"C|1-5"}
#     name2node = {node_name(c): c for c in tree.nodes}
#     c15 = name2node["C|1-5"]
#     assert len(c15.children) == 0
#     assert c15.parent is None
#     c13 = name2node["C|1-3"]
#     assert c13.parent is None
#     assert c13.children == ChainList([name2node["C|1-2"]])
#
#     tree = make_obj_tree([c13], connect=False)
#     assert len(tree.nodes) == 2
#     assert len(tree.edges) == 1
#
#     tree = make_obj_tree(chains, connect=True)
#     name2node = {node_name(c): c for c in tree.nodes}
#     assert len(name2node["C|1-5"].children) == 2
#     assert name2node["C|1-3"].parent == name2node["C|1-5"]
#     assert name2node["C|4-5"].parent == name2node["C|1-5"]
#     assert name2node["C|1-2"].id == "C|1-2<-(C|1-3<-(C|1-5))"
#
#     c = simple_chain_structure
#     child = c.spawn_child(1, 2)
#     child.parent = None
#     make_obj_tree([child, c], connect=True)
#     assert child.parent == c


def test_make_str_tree(chains, simple_chain_structure):
    tree = make_id_tree(chains, connect=False)
    assert len(tree.nodes) == 4
    assert tree.is_directed()
    assert all(isinstance(n, str) for n in tree.nodes)
    assert all("objs" in tree.nodes[n] for n in tree.nodes)
    for node in tree.nodes:
        objs = tree.nodes[node]["objs"]
        assert len(objs) >= 1
        for obj in objs:
            obj_name = node_name(obj)
            assert obj_name == node

    diff = set(tree.nodes) - set(map(node_name, chains))
    assert diff == {"C|1-5"}

    c13 = chains.filter(lambda x: x.start == 1 and x.end == 3).pop()
    tree = make_id_tree([c13], connect=False)
    assert len(tree.nodes) == 2
    assert len(tree.edges) == 1

    c15 = tree.nodes["C|1-5"]["objs"][0]
    assert len(c15.children) == 0
    tree = make_id_tree([c13], connect=True)
    c15 = tree.nodes["C|1-5"]["objs"][0]
    assert len(c15.children) == 1


def test_recover_tree(chains, simple_chain_structure):
    c45 = chains["C|4-5"].pop()

    c45_chain = Chain.from_seq(c45)
    c45_chain_ = recover(c45_chain)
    assert c45_chain == c45_chain_
    assert c45_chain_.parent is not None
    assert c45_chain_.seq.parent is not None
    assert c45_chain.parent is not None
    assert c45_chain.seq.parent is not None

    c45_ = recover(c45)
    assert c45_.parent is not None
    assert c45.parent is not None

    cs_child = simple_chain_structure.spawn_child(1, 2)
    cs_child.parent = None
    recover(cs_child)
    assert cs_child.parent.id == simple_chain_structure.id
    assert cs_child.seq.parent.id == simple_chain_structure.seq.id

    c12 = ChainSequence.from_string("cc", start=1, end=2, name="C")
    o12 = ChainSequence.from_string("oo", start=1, end=2, name="C")
    c12.meta["id"] = "C|1-2<-(X|1-5)"
    o12.meta["id"] = "O|1-2<-(Y|1-5)"
