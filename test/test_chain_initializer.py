import operator as op
from itertools import chain

import pytest

from lXtractor.core.chain import ChainInitializer, ChainSequence, ChainStructure, Chain
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import InitError


@pytest.fixture
def items(
    simple_structure, simple_structure_path, chicken_src_seq_path, simple_chain_seq
):
    return [
        ('SEQ', 'ABCD'),
        simple_structure,
        simple_structure_path,
        chicken_src_seq_path,
        (simple_structure_path, ['A']),
        simple_chain_seq[1],
    ]


@pytest.fixture
def mapping(chicken_src_str_path, chicken_src_seq_path, simple_structure):
    simple_seq = "".join(map(op.itemgetter(0), simple_structure.get_sequence()))
    return {
        # chicken_src_seq_path: [chicken_src_str_path],
        ChainSequence.from_file(chicken_src_seq_path): [(chicken_src_str_path, ['A'])],
        ('S', simple_seq): [simple_structure],
    }


def assert_iterable(io, items, num_proc=1):
    res = list(io.from_iterable(items, num_proc=num_proc))
    assert len(res) == 6
    assert all([isinstance(r, (ChainSequence, ChainStructure, list)) for r in res])


def test_iterable(items):
    io = ChainInitializer()
    assert_iterable(io, items)


def test_iterable_parallel(items):
    io = ChainInitializer()
    assert_iterable(io, items, 2)


def assert_mapping(mapping, io, num_proc=1):
    chains = io.from_mapping(
        mapping,
        num_proc_read_str=num_proc,
        num_proc_read_seq=num_proc,
        num_proc_map_numbering=num_proc,
    )
    assert len(chains) == 2
    assert all(isinstance(x, Chain) for x in chains)
    assert len(chains[0].structures) == 1
    assert len(chains[1].structures) == 1
    assert all(
        SeqNames.map_canonical in x.seq
        for x in chain.from_iterable(c.structures for c in chains)
    )


def test_mapping(mapping):
    io = ChainInitializer(tolerate_failures=False)
    assert_mapping(mapping, io)


def test_mapping_parallel(mapping):
    io = ChainInitializer()
    assert_mapping(mapping, io, num_proc=2)


def test_mapping_invalid_objects(simple_chain_seq):
    _, s = simple_chain_seq
    m = {s: [1, None]}

    io = ChainInitializer()
    with pytest.raises(InitError):
        io.from_mapping(m)

    io = ChainInitializer(tolerate_failures=True)
    chains = list(io.from_mapping(m))
    assert len(chains) == 1
    assert SeqNames.map_canonical not in chains[0].seq


def test_callbacks(items, mapping):
    def accept(obj):
        if isinstance(obj, ChainSequence):
            obj.name = 'X'
        return obj

    def empty_structures(item):
        return item[0], []

    io = ChainInitializer()
    xs = list(io.from_iterable(items, callbacks=[accept]))
    assert len(xs) == 6
    assert xs[0].name == 'X'

    chains = io.from_mapping(mapping, key_callbacks=[accept])
    assert all([c.seq.name == 'X' for c in chains])

    # Emptying structures results in subsequent filtering to produce
    # an empty mapping
    chains = io.from_mapping(mapping, item_callbacks=[empty_structures])
    assert len(chains) == 0
