import pytest

from lXtractor.core.exceptions import InitError, NoOverlap
from lXtractor.core.protein import ProteinStructure


def test_init(simple_structure, four_chain_structure):
    pdb_id = 'xxxx'
    # no structure init
    s = ProteinStructure(pdb_id, 'A')
    assert s.seq is None
    assert s.pdb.structure is None

    # chain must match
    with pytest.raises(InitError):
        ProteinStructure(pdb_id, 'B', simple_structure)

    # must be a single chain
    with pytest.raises(InitError):
        ProteinStructure(pdb_id, 'ABCD', four_chain_structure)

    # init from structure or atom array work exactly the same
    s = ProteinStructure.from_structure(simple_structure.array)
    assert s.pdb.chain == 'A'
    assert s.pdb.structure is not None
    assert s.seq is not None

    s = ProteinStructure.from_structure(simple_structure)
    assert s.pdb.chain == 'A'
    assert s.pdb.structure is not None
    assert s.seq is not None

    # ensure seq is initialized
    fields = s.seq.field_names()
    assert fields.seq1 in s.seq
    assert fields.seq3 in s.seq
    assert fields.enum in s.seq


def test_spawn(simple_structure):
    s = ProteinStructure.from_structure(simple_structure)

    # invalid boundaries
    with pytest.raises(ValueError):
        s.spawn_child(2, 1)

    with pytest.raises(NoOverlap):
        s.spawn_child(200, 300)

    # The spawned child may contain a single residue
    sub = s.spawn_child(1, 1)
    assert len(sub.seq) == 1

    # Using the mapping
    s.seq.add_seq('map_something', [x.i + 1000 for x in s.seq])
    sub = s.spawn_child(1000, 1100, map_name='map_something')
    assert len(sub.seq) == 100

    # Should find the closest mapped boundary and give the same result
    sub = s.spawn_child(1, 1100, map_name='map_something')
    assert len(sub.seq) == 100
