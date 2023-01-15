import pytest

from lXtractor.core.chain import Chain, ChainStructure
from lXtractor.core.config import Sep, SeqNames
from lXtractor.core.exceptions import NoOverlap
from test.common import sample_chain


def test_basic(four_chain_structure_seq_path, four_chain_structure):
    p = Chain.from_seq(four_chain_structure_seq_path)[0]
    assert len(p.seq) == 1657
    assert p.id == p.seq.id
    chains = list(four_chain_structure.split_chains())
    chain_a = ChainStructure.from_structure(chains[0], pdb_id='3i6x')
    p.add_structure(chain_a)
    assert len(p.structures) == 1
    start, end = chain_a.seq.start, chain_a.seq.end
    assert f'3i6x{Sep.chain}A{Sep.start_end}{start}-{end}'

    with pytest.raises(ValueError):
        p.add_structure(chain_a)

    p.add_structure(chain_a, check_ids=False)
    assert len(p.structures) == 2


def test_spawn(chicken_src_seq, human_src_seq, chicken_src_str):
    p = Chain.from_seq(chicken_src_seq)
    chains = chicken_src_str.split_chains()
    chain_a = ChainStructure.from_structure(next(chains))
    p.add_structure(chain_a, map_name=SeqNames.map_canonical)

    # should work on any full protein chain
    child = p.spawn_child(1, 2, keep=False, subset_structures=False)
    assert not p.children
    assert not p.seq.children
    assert all(not s.children for s in p.structures)
    assert len(child.seq) == 2
    assert child.seq.seq1 == 'MG'
    assert len(child.structures) == 0

    # Using segment's boundaries
    with pytest.raises(NoOverlap):
        _ = p.spawn_child(1, 480, 'child', keep=True)
    child = p.spawn_child(1, 260, 'child', keep=True)
    assert len(child.seq) == 260
    assert len(child.structures) == 1
    assert len(child.structures[0].seq) == 260
    assert 'child' in [c.seq.name for c in p.children]

    # Using canonical seq numbering
    # +-----------------------|----|------------+
    # 1                       256  260
    #                         +----|------------+
    #                         1    5
    child = p.spawn_child(
        1, 260, str_map_from=SeqNames.map_canonical, str_map_closest=True
    )
    assert len(child.seq) == 260
    s = child.structures[0]
    assert len(s.seq) == 5
    assert s.seq.start == 1
    assert s.seq.end == 5
    assert child.seq.seq1[-5:] == s.seq.seq1
    s_num = s.seq[SeqNames.map_canonical]
    assert s_num[0] == 256
    assert s_num[-1] == 260

    child_of_child = child.spawn_child(256, 260, str_map_from=SeqNames.map_canonical)
    children = list(p.iter_children())
    assert len(children) == 2
    assert child_of_child.seq.name in [c.seq.name for c in children[-1]]

    with pytest.raises(KeyError):
        _ = p.spawn_child(
            1, 4, 'child', keep=False, str_map_from=SeqNames.enum, str_map_closest=False
        )


def test_iter(chicken_src_str):
    def get_name(_c):
        return _c.seq.name

    c = sample_chain()

    levels = list(c.iter_children())
    assert len(levels) == 3
    assert list(map(get_name, levels[0])) == ['c1', 'c2']
    assert list(map(get_name, levels[1])) == ['c1_1', 'c1_2', 'c2_1', 'c2_2']
    assert list(map(get_name, levels[2])) == ['c1_2_1']
