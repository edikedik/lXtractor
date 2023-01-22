import pytest

from lXtractor.core.chain import ChainSequence, ChainList, Chain, ChainStructure
from lXtractor.core.chain.tree import (
    list_ancestors,
    list_ancestors_names,
    make_obj_tree,
    make_filled,
    node_name,
    make_str_tree,
)


@pytest.fixture
def chains():
    c13 = ChainSequence.from_string('ccc', start=1, end=3, name='C')
    c12 = c13.spawn_child(1, 2)
    c45 = ChainSequence.from_string('cc', start=4, end=5, name='C')
    c13.meta['id'] = 'C|1-3<-(C|1-5)'
    c45.meta['id'] = 'C|4-5<-(C|1-5)'
    return ChainList([c13, c12, c45])


def test_ancestors_list(chains, simple_chain_structure):
    par_list = list_ancestors(chains['C|1-2'].pop())
    assert len(par_list) == 1
    assert [x.id for x in par_list] == ['C|1-3']
    c = simple_chain_structure.spawn_child(1, 2)
    par_list = list_ancestors(c)
    assert [x.id for x in par_list] == [str(simple_chain_structure)]


def test_list_ancestors_names(chains, simple_chain_structure):
    par_list = list_ancestors_names(chains['C|1-2'].pop())
    assert par_list == ['C|1-3']
    par_list = list_ancestors_names('C|1-2<-(C|1-3<-(C|1-5))')
    assert par_list == ['C|1-3', 'C|1-5']
    c = simple_chain_structure.spawn_child(1, 2)
    par_list = list_ancestors_names(c)
    assert par_list == [simple_chain_structure.meta['id']]


@pytest.mark.parametrize(
    'inp,is_valid,example_obj,expected',
    [
        ('C|1-5', True, ChainSequence.make_empty(), (5, True)),
        ('C|1-1', True, Chain.make_empty(), (1, True)),
        ('C|some|info|1-5', True, ChainStructure.make_empty(), (0, True)),
        ('C|1-5', True, ChainSequence, (5, False)),
        ('C|1-1', True, Chain, (1, False)),
        ('C|some|info|1-5', True, ChainStructure, (0, False)),
        ('C|0-0', True, ChainSequence, (0, False)),
        ('C|0-0', True, Chain, (0, False)),
        ('C|some|info|0-0', True, ChainStructure, (0, False)),
        ('C|1~5', False, None, None),
        ('C|0-1', False, None, None),
        ('C/1-5', False, None, None),
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


def test_make_obj_tree(chains, simple_chain_structure):
    tree = make_obj_tree(chains, connect=False)
    assert len(tree.nodes) == 4
    assert tree.is_directed()
    diff = set(map(node_name, tree.nodes)) - set(map(node_name, chains))
    assert diff == {'C|1-5'}
    name2node = {node_name(c): c for c in tree.nodes}
    c15 = name2node['C|1-5']
    assert len(c15.children) == 0
    assert c15.parent is None
    c13 = name2node['C|1-3']
    assert c13.parent is None
    assert c13.children == ChainList([name2node['C|1-2']])

    tree = make_obj_tree([c13], connect=False)
    assert len(tree.nodes) == 2
    assert len(tree.edges) == 1

    tree = make_obj_tree(chains, connect=True)
    name2node = {node_name(c): c for c in tree.nodes}
    assert len(name2node['C|1-5'].children) == 2
    assert name2node['C|1-3'].parent == name2node['C|1-5']
    assert name2node['C|4-5'].parent == name2node['C|1-5']
    assert name2node['C|1-2'].id == 'C|1-2<-(C|1-3<-(C|1-5))'

    c = simple_chain_structure
    child = c.spawn_child(1, 2)
    child.parent = None
    make_obj_tree([child, c], connect=True)
    assert child.parent == c


def test_make_str_tree(chains, simple_chain_structure):
    tree = make_str_tree(chains, connect=False)
    assert len(tree.nodes) == 4
    assert tree.is_directed()
    assert all(isinstance(n, str) for n in tree.nodes)
    assert all('objs' in tree.nodes[n] for n in tree.nodes)
    for node in tree.nodes:
        objs = tree.nodes[node]['objs']
        assert len(objs) >= 1
        for obj in objs:
            obj_name = node_name(obj)
            assert obj_name == node

    diff = set(tree.nodes) - set(map(node_name, chains))
    assert diff == {'C|1-5'}

    c13 = chains.filter(lambda x: x.start == 1 and x.end == 3).pop()
    tree = make_str_tree([c13], connect=False)
    assert len(tree.nodes) == 2
    assert len(tree.edges) == 1

    c15 = tree.nodes['C|1-5']['objs'][0]
    assert len(c15.children) == 0
    tree = make_str_tree([c13], connect=True)
    c15 = tree.nodes['C|1-5']['objs'][0]
    assert len(c15.children) == 1
