"""
A module to handle the ancestral tree of the Chain*-type objects defined
by their ``parent``/``children`` attributes and/or ``meta`` info.
"""
import re
import typing as t
from collections import abc

import networkx as nx
from more_itertools import windowed

from lXtractor.core.chain import Chain, ChainSequence, ChainStructure, ChainList
from lXtractor.core.config import EMPTY_PDB_ID, EMPTY_CHAIN_ID
from lXtractor.core.exceptions import MissingData, FormatError

T = t.TypeVar('T')
CT = t.TypeVar('CT', ChainSequence, ChainStructure, Chain)
CT_: t.TypeAlias = Chain | ChainSequence | ChainStructure

_SEP = '<-('
FILLER = '*'
NODE_PATTERN = re.compile(r'(.+)\|(\d+-\d+)')


def node_name(c: CT_) -> str:
    """
    :param c: Chain*-type object.
    :return:
    """
    return f'{c.name}|{c.start}-{c.end}'


def list_ancestors_names(id_or_chain: CT_ | str) -> list[str]:
    """
    >>> list_ancestors_names('C|1-5<-(C|1-3<-(C|1-2))')
    ['C|1-3', 'C|1-2']

    :param id_or_chain: Chain*-type object or its id.
    :return: A list of parents '{name}|{start}-{end}' representations parsed
        from the object's id.
    """
    if not isinstance(id_or_chain, str):
        try:
            _id = id_or_chain.meta['id']
        except KeyError as e:
            raise MissingData(f'Missing ID property in meta of {id_or_chain}') from e
    else:
        _id = id_or_chain

    obj_ids = []

    while _SEP in _id:
        _id_sep = _id.split(_SEP)
        _id = _SEP.join(_id_sep[1:]).removesuffix(')')
        obj_ids.append(_id_sep[0])
    obj_ids.append(_id)
    return obj_ids[1:]


def list_ancestors(c: CT_) -> list[CT_]:
    """
    >>> o = ChainSequence.from_string('x' * 5, 1, 5, 'C')
    >>> c13 = o.spawn_child(1, 3)
    >>> c12 = c13.spawn_child(1, 2)
    >>> list_ancestors(c12)
    [C|1-3<-(C|1-5), C|1-5]

    :param c: Chain*-type object.
    :return: A list ancestor objects obtained from the ``parent`` attribute..
    """
    if c.parent is None:
        return []
    parents = []
    while c.parent is not None:
        parents.append(c.parent)
        c = c.parent
    return parents


def make_filled(name: str, _t: CT | t.Type[CT]) -> CT:
    """
    Make a "filled" version of an object to occupy the tree.

    :param name: Name of the node obtained via :func:`node_name`.
    :param _t: Some Chain*-type object.
    :return: An object with filled sequence. If it's a
        :class:`ChainStructure <lXtractor.core.chain.structure.ChainStructure>`
        object, it will have an empty structure.
    """
    re_find = NODE_PATTERN.findall(name)
    if len(re_find) != 1:
        raise FormatError(
            f'Failed to parse name {name} into {{name}}|{{start}}-{{end}} format. '
            f're.findall results: {re_find}'
        )
    match = re_find[0]
    if len(match) != 2:
        raise FormatError(
            f'Unexpected match from {name}. Expected to find exactly two items: '
            f'name and boundaries. Found {len(match)}: {match}'
        )
    real_name, bounds = match
    start, end = map(int, bounds.split('-'))
    if start == end == 0:
        if issubclass(_t, (Chain, ChainStructure, ChainSequence)):
            return _t.make_empty()
        return _t.__class__.make_empty()
    size = end - start + 1
    seq = ChainSequence.from_string(size * FILLER, start, end, real_name)
    is_chain_instance = (
        isinstance(_t, Chain),
        isinstance(_t, ChainSequence),
        isinstance(_t, ChainStructure),
    )
    if any(is_chain_instance):
        if is_chain_instance[0]:
            return Chain.from_seq(seq)
        if is_chain_instance[1]:
            return seq
        if is_chain_instance[2]:
            return ChainStructure(EMPTY_PDB_ID, EMPTY_CHAIN_ID, None, seq=seq)
        raise RuntimeError('...')
    try:
        is_chain_subclass = (
            issubclass(_t, Chain),
            issubclass(_t, ChainSequence),
            issubclass(_t, ChainStructure),
        )
    except TypeError:
        raise TypeError(f'Failed to infer type of {_t}')

    if is_chain_subclass[0]:
        return Chain.from_seq(seq)
    if is_chain_subclass[1]:
        return seq
    if is_chain_subclass[2]:
        return ChainStructure(EMPTY_PDB_ID, EMPTY_CHAIN_ID, None, seq=seq)

    raise RuntimeError('...')


def make_tree(chains: abc.Sequence[CT], connect: bool = False) -> nx.DiGraph:
    """
    Make an ancestral tree -- a directed graph representing ancestral
    relationships between chains. The nodes of the tree are Chain*-type
    objects. Hence, they must be hashable. This restricts types of sequences
    valid for :class:`ChainSequence <lXtractor.core.chain.sequence.
    ChainSequence>` to ``abc.Sequence[abc.Hashable]``.

    As a useful side effect, this function can aid in filling the gaps in the
    actual tree indicated by the id-relationship suggested by the "id" field
    of the ``meta`` property. In other words, if a segment S|1-2 was obtained
    by spawning from S|1-5, S|1-2's id will reflect this:

    >>> s = make_filled('S|1-5', ChainSequence.make_empty())
    >>> c12 = s.spawn_child(1, 2)
    >>> c12
    S|1-2<-(S|1-5)

    However, if S|1-5 was lost (e.g., by writing/reading S|1-2 to/from disk),
    and S|1-2.parent is None, we can use ID stored in meta to recover ancestral
    relationships. This function will attend to such cases and create a filler
    object S|1-5 with a "*"-filled sequence.

    >>> c12.parent = None
    >>> c12
    S|1-2
    >>> c12.meta['id']
    'S|1-2<-(S|1-5)'
    >>> ct = make_tree([c12], connect=True)
    >>> assert len(ct.nodes) == 2
    >>> [n.id for n in ct.nodes]
    ['S|1-2<-(S|1-5)', 'S|1-5']

    :param chains: A homogeneous sequence of Chain*-type objects.
    :param connect: If ``True``, connect both supplied and created filler
        objects via ``children`` and ``parent`` attributes.
    :return: A networkx's directed graph with Chain*-type objects as nodes.
    """
    if not isinstance(chains, ChainList):
        chains = ChainList(chains)
    tree = nx.DiGraph()
    chains = chains.collapse_children() + chains

    # Populate objects' tree
    for c in chains:
        if not tree.has_node(c):
            tree.add_node(c)
        parents = list_ancestors(c)
        if parents:
            # fails to recognize parents as iterable
            for child, parent in windowed([c, *parents], 2):  # type: ignore
                tree.add_edge(parent, child)

    # Make tree fully connected and populate `parent`, `children`
    # attributes
    name2node = {node_name(n): n for n in tree.nodes}
    node_example = chains[0]
    node: CT
    for node in list(tree.nodes):
        names = [node_name(node), *list_ancestors_names(node)]
        for child_name, parent_name in windowed(names, 2):
            assert child_name is not None
            if parent_name is None:
                continue
            parent_obj = name2node.get(
                parent_name, make_filled(parent_name, node_example)
            )
            name2node[parent_name] = parent_obj
            child_obj = name2node[child_name]
            tree.add_edge(parent_obj, child_obj)
            if connect:
                if child_obj not in parent_obj.children:
                    parent_obj.children.append(child_obj)
                child_obj.parent = parent_obj

    assert nx.is_tree(tree), 'networkx confirms it is a tree'

    return tree


if __name__ == '__main__':
    raise RuntimeError
