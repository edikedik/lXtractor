from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import biotite.structure as bst
import pytest

from lXtractor.core.chain import ChainStructure
from lXtractor.core.config import DumpNames, MetaNames
from lXtractor.core.exceptions import InitError, NoOverlap, LengthMismatch, MissingData
from lXtractor.util.io import get_files, get_dirs
from test.common import mark_meta
from test.conftest import EPS


@pytest.fixture
def simple_chain_structure(simple_structure):
    return ChainStructure.from_structure(simple_structure, 'xxxx')


def test_init(simple_structure, four_chain_structure):
    pdb_id = 'xxxx'
    # no structure init
    s = ChainStructure(pdb_id, 'A', simple_structure)
    assert s.seq is not None

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


def test_degenerate(simple_chain_structure):
    simple_cs = simple_chain_structure
    cs = ChainStructure('xxxx', 'x', None)
    assert len(cs) == 0 and cs.is_empty
    assert ChainStructure.make_empty('xxxx', 'x') == cs
    assert len(cs.seq) == 0
    assert cs.rm_solvent() == cs
    with pytest.raises(MissingData):
        simple_cs.superpose(cs)
    with pytest.raises(MissingData):
        cs.superpose(simple_cs)
    with pytest.raises(MissingData):
        cs.superpose(cs)
    with pytest.raises(MissingData):
        cs.spawn_child(1, 1)
    with pytest.raises(MissingData):
        cs.write(Path('./anywhere'))


def test_spawn(simple_chain_structure):
    s = simple_chain_structure

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


def test_iterchildren(simple_chain_structure):
    s = simple_chain_structure
    child1 = s.spawn_child(1, 10)
    child2 = child1.spawn_child(5, 6)
    levels = list(s.iter_children())
    assert len(levels) == 2
    assert levels == [[child1], [child2]]


def test_io(simple_chain_structure):
    s = simple_chain_structure
    child1 = s.spawn_child(1, 10)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        s.write(tmp_path, write_children=True)

        files = get_files(tmp_path)
        dirs = get_dirs(tmp_path)

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
        s_r_child = s_r.children.pop()
        assert not s_r_child.seq.children
        assert s_r_child.seq.meta[MetaNames.pdb_id] == child1.pdb.id
        assert s_r_child.seq.meta[MetaNames.pdb_chain] == child1.pdb.chain


def test_superpose(simple_chain_structure):
    s = simple_chain_structure

    superposed, rmsd, _ = s.superpose(s)
    assert rmsd < EPS
    assert f'rmsd_xxxx:A' in superposed.seq.meta

    _, rmsd, _ = s.superpose(s, [1])
    assert rmsd < EPS

    _, rmsd, _ = s.superpose(s, [1], ['C', 'CA'])
    assert rmsd < EPS

    _, rmsd, _ = s.superpose(s, [1], [['C']])

    with pytest.raises(LengthMismatch):
        _ = s.superpose(s, [1, 2], [['C', 'A']])

    # using mappings
    s_cp = deepcopy(s)
    s_cp.pdb.structure.array.res_id += 1
    s_cp.seq['numbering'] = [x + 1 for x in s_cp.seq['numbering']]
    s_cp.seq.map_numbering(s.seq, name='original')

    # mapping is from `self` sequence numeration to the `other`'s
    # => {1=>2,2=>3}.
    superposed, rmsd, _ = s.superpose(s_cp, [1, 2], map_name_other='original')
    assert rmsd < EPS


def test_rm_solvent(simple_chain_structure):
    s = simple_chain_structure
    _, seq = bst.get_residues(s.array)
    assert 'HOH' in seq
    n_hoh = sum(1 for x in seq if x == 'HOH')
    n_rest = sum(1 for x in seq if x != 'HOH')
    assert n_hoh + n_rest == len(seq)
    srm = s.rm_solvent()
    _, seq_rm = bst.get_residues(srm.array)
    assert 'HOH' not in seq_rm
    assert len(seq_rm) == n_rest
    assert len(seq_rm) + n_hoh == len(seq)


def test_filter_children(simple_chain_structure):
    s = simple_chain_structure
    s.spawn_child(s.seq.start + 1, s.seq.end - 1, 'X1')
    s.spawn_child(s.seq.start + 1, s.seq.end - 1, 'X2')
    s_new = s.filter_children(lambda x: x.seq.name != 'X1')
    assert (len(s.children) - 1) == len(s_new.children) == 1
    assert s_new.children[0].seq.name == 'X2'
    _ = s.filter_children(lambda x: x.seq.name != 'X1', inplace=True)
    assert len(s.children) == 1
    assert s.children[0].seq.name == 'X2'


def test_apply_children(simple_chain_structure):
    s = simple_chain_structure
    s.spawn_child(s.seq.start + 1, s.seq.end - 1, 'X1')
    s.spawn_child(s.seq.start + 1, s.seq.end - 1, 'X2')
    s_new = s.apply_children(mark_meta)
    assert len(s.children) == 2
    assert all('X' not in c.meta for c in s.children)
    assert all('X' in c.meta for c in s_new.children)
    s.apply_children(mark_meta, inplace=True)
    assert all('X' in c.meta for c in s.children)
