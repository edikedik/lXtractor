from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lXtractor.core.config import DumpNames, MetaNames
from lXtractor.core.exceptions import InitError, NoOverlap
from lXtractor.core.chain import ChainStructure
from lXtractor.util.io import get_files, get_dirs


def test_init(simple_structure, four_chain_structure):
    pdb_id = 'xxxx'
    # no structure init
    s = ChainStructure(pdb_id, 'A')
    assert s.seq is None
    assert s.pdb.structure is None

    # chain must match
    with pytest.raises(InitError):
        ChainStructure(pdb_id, 'B', simple_structure)

    # must be a single chain
    with pytest.raises(InitError):
        ChainStructure(pdb_id, 'ABCD', four_chain_structure)

    # init from structure or atom array work exactly the same
    s = ChainStructure.from_structure(simple_structure.array)
    assert s.pdb.chain == 'A'
    assert s.pdb.structure is not None
    assert s.seq is not None

    s = ChainStructure.from_structure(simple_structure)
    assert s.pdb.chain == 'A'
    assert s.pdb.structure is not None
    assert s.seq is not None

    # ensure seq is initialized
    fields = s.seq.field_names()
    assert fields.seq1 in s.seq
    assert fields.seq3 in s.seq
    assert fields.enum in s.seq


def test_spawn(simple_structure):
    s = ChainStructure.from_structure(simple_structure)

    # invalid boundaries
    with pytest.raises(ValueError):
        s.spawn_child(2, 1)

    with pytest.raises(NoOverlap):
        s.spawn_child(200, 300)

    sub = s.spawn_child(1, 1, keep=True, keep_seq_child=False)
    assert sub.id in s.children
    assert not sub.seq.children
    assert len(sub.seq) == 1

    # Using the mapping
    s.seq.add_seq('map_something', [x.i + 1000 for x in s.seq])
    sub = s.spawn_child(1000, 1100, map_from='map_something')
    assert len(sub.seq) == 100

    # Should find the closest mapped boundary and give the same result
    with pytest.raises(KeyError):
        _ = s.spawn_child(1, 1100, map_from='map_something', map_closest=False)
    sub = s.spawn_child(1, 1100, map_from='map_something', map_closest=True)
    assert len(sub.seq) == 100


def test_iterchildren(simple_structure):
    s = ChainStructure.from_structure(simple_structure)
    child1 = s.spawn_child(1, 10)
    child2 = child1.spawn_child(5, 6)
    levels = list(s.iter_children())
    assert len(levels) == 2
    assert levels == [[child1], [child2]]


def test_io(simple_structure):
    s = ChainStructure.from_structure(simple_structure)
    child1 = s.spawn_child(1, 10)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        s.write(tmp_path, write_children=True)

        files = get_files(tmp_path)
        dirs = get_dirs(Path(tmp))

        assert f'{DumpNames.structure_base_name}.cif' in files
        assert DumpNames.sequence in files
        assert DumpNames.meta in files
        assert DumpNames.segments_dir in dirs

        s_r = ChainStructure.read(tmp_path, search_children=True)
        assert s_r.seq is not None
        assert s_r.pdb.structure is not None
        assert s_r.pdb.id == s.pdb.id
        assert s_r.pdb.chain == s.pdb.chain

        assert len(s_r.children) == 1
        s_r_child = s_r.children[child1.seq.name]
        assert not s_r_child.seq.children
        assert s_r_child.seq.meta[MetaNames.pdb_id] == child1.pdb.id
        assert s_r_child.seq.meta[MetaNames.pdb_chain] == child1.pdb.chain