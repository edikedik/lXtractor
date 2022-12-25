import pytest

from lXtractor.core.chain import ChainList, ChainSequence, ChainStructure
from lXtractor.core.segment import Segment


@pytest.fixture
def structure(simple_structure) -> ChainStructure:
    return ChainStructure('1234', 'A', simple_structure)


@pytest.fixture
def sequence() -> ChainSequence:
    return ChainSequence.from_string('a', name='X')


def test_init(simple_structure, structure, sequence):
    cl = ChainList([])
    assert len(cl) == 0
    cl.insert(0, sequence)
    assert len(cl) == 1
    with pytest.raises(TypeError):
        ChainList([1])
    with pytest.raises(TypeError):
        ChainList([structure, sequence])
    assert len(ChainList([structure])) == 1
    assert len(ChainList([sequence])) == 1


def test_basic(sample_chain_list, sequence, structure):
    cl = ChainList([sequence, sequence])
    assert len(cl) == 2
    assert len(list(cl.iter_sequences())) == 2
    assert len(list(cl.iter_structures())) == 0

    cl = ChainList([structure, structure])
    assert len(cl) == 2
    assert len(list(cl.iter_sequences())) == 2
    assert len(list(cl.iter_structures())) == 2

    assert sequence not in cl
    assert structure in cl
    assert 'ChainStructure(1234:A|1-207)' in cl

    with pytest.raises(TypeError):
        cl.insert(1, 1)
    with pytest.raises(TypeError):
        cl[0] = 1

    cl += [structure]
    assert len(cl) == 3
    with pytest.raises(TypeError):
        cl += [sequence]

    cl.remove(structure)
    assert len(cl) == 2

    cl = sample_chain_list
    assert len(cl) == 2
    assert cl[0].seq.name == 'c_root'
    assert cl[1].seq.name == 'k_root'
    assert cl[:1].pop().seq.name == 'c_root'
    assert len(list(iter(cl))) == 2


def test_modifying(sequence, structure):
    cl = ChainList([])
    cl.append(sequence)
    assert sequence in cl
    s = cl.pop()
    assert s is sequence
    s2 = ChainSequence.from_string('aa', name='Y')
    cl += [s, s2]
    assert len(cl) == 2
    cl.insert(0, s)
    assert [x.id for x in cl] == [s.id, s.id, s2.id]
    cl.remove(s)
    assert len(cl) == 2
    cl[0:2] = [s2, s]
    assert [x.id for x in cl] == [s2.id, s.id]

    cl += [s]
    assert len(cl) == 3

    with pytest.raises(TypeError):
        cl.insert(1, structure)
        print(cl)

    with pytest.raises(TypeError):
        cl += [structure]


def test_objects_retrieval(sample_chain_list):
    cl = sample_chain_list
    cls = list(cl.iter_children())
    assert all(isinstance(x, ChainList) for x in cls)
    assert len(cls) == 3
    l1, l2, l3 = cls
    assert len(l1) == 4
    assert len(l2) == 8
    assert len(l3) == 2
    assert (l3[0].seq.name, l3[1].seq.name) == ('c1_2_1', 'k1_2_1')

    assert len(cl.collapse_children()) == 14

    seqs = list(cl.iter_sequences())
    assert len(seqs) == 2
    assert (seqs[0].name, seqs[1].name) == ('c_root', 'k_root')

    structures = list(cl.iter_structures())
    assert len(structures) == 0
    structures = list(l3.iter_structures())
    assert len(structures) == 2


def test_filter_pos(sample_chain_list):
    cl = sample_chain_list
    seqs = cl.filter_pos(Segment(5, 7))
    assert all(isinstance(x, ChainSequence) for x in seqs)
    assert len(seqs) == 2
    assert len(list(cl.filter_pos(Segment(11, 12)))) == 0
    # assert len(list(cl.filter_pos(Segment(1, 2), obj_type='struc'))) == 0
    children_it = cl.iter_children()
    l1 = next(children_it)
    assert len(list(l1.filter_pos(Segment(5, 9)))) == 4
    assert len(list(l1.filter_pos(Segment(6, 9)))) == 2
    # assert len(list(l1.filter_pos(Segment(1, 2), obj_type='struc'))) == 2

    # the match is bounded by segment
    assert len(list(l1.filter_pos(Segment(1, 5), match_type='bounded'))) == 2
    # the match is bounding the segment
    assert len(list(l1.filter_pos(Segment(1, 5), match_type='bounding'))) == 4

    # match by positions
    m = list(l1.filter_pos([1, 2]))
    assert len(m) == 4
    assert m[0].name == 'c1'
    assert len(list(l1.filter_pos([1, 2, 6]))) == 2
    assert len(list(l1.filter_pos([1, 2, 6, 10]))) == 0


def test_iter_children(sample_chain_list):
    it = sample_chain_list.iter_children()
    l1 = next(it)
    assert len(l1) == 4
    l2 = next(it)
    assert len(l2) == 8
    l3 = next(it)
    assert len(l3) == 2
    assert [x.seq.name for x in l3] == ['c1_2_1', 'k1_2_1']
    with pytest.raises(StopIteration):
        next(it)

    # Unequal depth of a child tree
    s = ChainSequence.from_string('ABCDE', name='A')
    child1 = s.spawn_child(1, 4)
    child1.spawn_child(2, 3)
    x = ChainSequence.from_string('XXXX', name='X')
    x.spawn_child(1, 3)
    cl = ChainList([s, x])
    l = list(cl.iter_children())
    assert len(l) == 2
    assert len(l[0]) == 2
    assert len(l[1]) == 1
