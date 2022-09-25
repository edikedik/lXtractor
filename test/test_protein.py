import pytest

from lXtractor.core.config import Sep, ProteinSeqNames
from lXtractor.core.exceptions import NoOverlap
from lXtractor.core.protein import Protein, ProteinStructure


def test_basic(four_chain_structure_seq_path, four_chain_structure):
    p = Protein.from_seq(four_chain_structure_seq_path)[0]
    assert len(p.seq) == 1657
    assert p.id == p.seq.id
    chains = list(four_chain_structure.split_chains())
    chain_a = ProteinStructure.from_structure(chains[0], pdb_id='3i6x')
    p.add_structure(chain_a)
    assert len(p.structures) == 1
    start, end = chain_a.seq.start, chain_a.seq.end
    assert f'3i6x{Sep.chain}A{Sep.start_end}{start}-{end}'

    with pytest.raises(ValueError):
        p.add_structure(chain_a)

    p.add_structure(chain_a, check_ids=False)
    assert len(p.structures) == 2


def test_spawn(chicken_src_seq, human_src_seq, chicken_src_str):
    p = Protein.from_seq(chicken_src_seq)
    chains = chicken_src_str.split_chains()
    chain_a = ProteinStructure.from_structure(next(chains))
    p.add_structure(chain_a, map_name=ProteinSeqNames.map_canonical)

    # should work on any full protein chain
    child = p.spawn_child(1, 2, keep=False, subset_structures=False)
    assert len(child.seq) == 2
    assert child.seq.seq1 == 'MG'
    assert len(child.structures) == 0

    with pytest.raises(NoOverlap):
        _ = p.spawn_child(1, 4, 'child', keep=False, subset_structures=True)

    # The structure starts from 256 -> five residues must be extracted
    child = p.spawn_child(
        1, 260, 'child', map_name=ProteinSeqNames.map_canonical)
    assert len(child.seq) == 260
    assert len(child.structures) == 1
    assert len(child.structures[0].seq) == 5

    child_of_child = child.spawn_child(
        256, 260, 'sub_child', map_name=ProteinSeqNames.map_canonical)
    assert len(child_of_child.seq) == 5
    assert child_of_child.seq.start == 256
    assert child_of_child.seq.end == 260

    # Use a human sequence and chicken structure so their numberings don't match
    p = Protein.from_seq(human_src_seq)
    p.add_structure(chain_a, map_name=ProteinSeqNames.map_canonical)
    child = p.spawn_child(
        256, 260, 'child', map_name=ProteinSeqNames.map_canonical)
    assert len(child.seq) == 5
    assert len(child.structures) == 1
    assert len(child.structures[0].seq) == 5
