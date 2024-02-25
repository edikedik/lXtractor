from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import biotite.structure as bst
import pandas as pd
import pytest

from lXtractor.chain import ChainStructure, ChainSequence
from lXtractor.core.config import DefaultConfig
from lXtractor.core.exceptions import InitError, NoOverlap, LengthMismatch, MissingData
from lXtractor.util.io import get_files, get_dirs
from test.common import mark_meta
from test.conftest import EPS


@pytest.fixture
def test_structure(simple_structure):
    return ChainStructure(simple_structure, "A")


def test_init(simple_structure, four_chain_structure):
    pdb_id = "xxxx"
    # no structure init
    s = ChainStructure(simple_structure)
    assert s.seq is not None

    # chain ID is ignored if the structure is not empty
    s = ChainStructure(simple_structure, "B")
    assert s.chain_id == "A"

    # must be a single chain
    with pytest.raises(InitError):
        ChainStructure(four_chain_structure)

    # init from structure or atom array work exactly the same
    s = ChainStructure(simple_structure.array)
    assert s.chain_id == "A"
    assert s.structure is not None
    assert s.seq is not None

    s = ChainStructure(simple_structure)
    assert s.chain_id == "A"
    assert s.structure is not None
    assert s.seq is not None

    # ensure _seq is initialized
    assert DefaultConfig["mapnames"]["seq1"] in s.seq
    assert DefaultConfig["mapnames"]["seq3"] in s.seq
    assert DefaultConfig["mapnames"]["enum"] in s.seq


def test_degenerate(test_structure):
    simple_cs = test_structure
    cs = ChainStructure(None)
    assert len(cs) == 0 and cs.is_empty
    assert ChainStructure.make_empty() == cs
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
        cs.write(Path("./anywhere"))

    seq = ChainSequence.make_empty()
    s = ChainStructure(None, seq=seq)
    assert len(s) == 0 and len(s.seq) == 0


def test_spawn(test_structure):
    s = test_structure

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
    s.seq.add_seq("map_something", [x.i + 1000 for x in s.seq])
    sub = s.spawn_child(1000, 1100, map_from="map_something")
    assert len(sub.seq) == 100

    # Should find the closest mapped boundary and give the same result
    with pytest.raises(KeyError):
        _ = s.spawn_child(1, 1100, map_from="map_something", map_closest=False)
    sub = s.spawn_child(1, 1100, map_from="map_something", map_closest=True)
    assert len(sub.seq) == 100


def test_iterchildren(test_structure):
    s = test_structure
    child1 = s.spawn_child(1, 10)
    child2 = child1.spawn_child(5, 6)
    levels = list(s.iter_children())
    assert len(levels) == 2
    assert levels == [[child1], [child2]]


@pytest.mark.parametrize("fmt", ["cif", "cif.gz", "mmtf.gz"])
def test_io(test_structure, fmt, tmp_path):
    s = test_structure
    child1 = s.spawn_child(1, 10)

    fnames, mnames = DefaultConfig["filenames"], DefaultConfig["metadata"]
    path = s.write(tmp_path, fmt=fmt, write_children=True)

    files = get_files(path)
    dirs = get_dirs(path)

    assert f"{fnames['structure_base_name']}.{fmt}" in files
    assert fnames["sequence"] in files
    assert fnames["meta"] in files
    assert fnames["segments_dir"] in dirs

    s_r = ChainStructure.read(path, search_children=True)
    assert s_r.seq is not None
    assert s_r.structure is not None
    assert s_r.structure.name == s.structure.name
    assert s_r.chain_id == s.chain_id

    assert len(s_r.children) == 1
    s_r_child = s_r.children.pop()
    assert not s_r_child.seq.children
    assert s_r_child.meta[mnames["structure_id"]] == child1.structure.name
    assert s_r_child.meta[mnames["structure_chain_id"]] == child1.chain_id


def test_superpose(test_structure):
    s = test_structure

    superposed, rmsd, _ = s.superpose(s)
    assert rmsd < EPS
    assert f"rmsd_{test_structure.structure.name}:A" in superposed.seq.meta

    _, rmsd, _ = s.superpose(s, [1])
    assert rmsd < EPS

    _, rmsd, _ = s.superpose(s, [1], ["C", "CA"])
    assert rmsd < EPS

    _, rmsd, _ = s.superpose(s, [1], [["C"]])

    with pytest.raises(LengthMismatch):
        _ = s.superpose(s, [1, 2], [["C", "A"]])

    # using mappings
    s_cp = deepcopy(s)
    s_cp.structure.array.res_id += 1
    s_cp.seq["numbering"] = [x + 1 for x in s_cp.seq["numbering"]]
    s_cp.seq.map_numbering(s.seq, name="original")

    # mapping is from `self` sequence numeration to the `other`'s
    # => {1=>2,2=>3}.
    superposed, rmsd, _ = s.superpose(s_cp, [1, 2], map_name_other="original")
    assert rmsd < EPS


def test_rm_solvent(test_structure):
    s = test_structure
    _, seq = bst.get_residues(s.array)
    assert "HOH" in seq
    n_hoh = sum(1 for x in seq if x == "HOH")
    n_rest = sum(1 for x in seq if x != "HOH")
    assert n_hoh + n_rest == len(seq)
    srm = s.rm_solvent()
    _, seq_rm = bst.get_residues(srm.array)
    assert "HOH" not in seq_rm
    assert len(seq_rm) == n_rest
    assert len(seq_rm) + n_hoh == len(seq)


def test_filter_children(test_structure):
    s = test_structure
    s.spawn_child(s.seq.start + 1, s.seq.end - 1, "X1")
    s.spawn_child(s.seq.start + 1, s.seq.end - 1, "X2")
    s_new = s.filter_children(lambda x: x._seq.name != "X1")
    assert (len(s.children) - 1) == len(s_new.children) == 1
    assert s_new.children[0].seq.name == "X2"
    _ = s.filter_children(lambda x: x._seq.name != "X1", inplace=True)
    assert len(s.children) == 1
    assert s.children[0].seq.name == "X2"


def test_apply_children(test_structure):
    s = test_structure
    s.spawn_child(s.seq.start + 1, s.seq.end - 1, "X1")
    s.spawn_child(s.seq.start + 1, s.seq.end - 1, "X2")
    s_new = s.apply_children(mark_meta)
    assert len(s.children) == 2
    assert all("X" not in c.meta for c in s.children)
    assert all("X" in c.meta for c in s_new.children)
    s.apply_children(mark_meta, inplace=True)
    assert all("X" in c.meta for c in s.children)


@pytest.mark.parametrize("meta", [True, False])
@pytest.mark.parametrize("ligands", [True, False])
def test_summary(chicken_src_str, meta, ligands):
    s = ChainStructure(next(chicken_src_str.split_chains()))
    df = s.summary(meta=meta, ligands=ligands)
    assert isinstance(df, pd.DataFrame)
    assert "ObjectID" in df.columns
    meta_keys = list(s.meta)
    if meta:
        assert set(meta_keys).issubset(set(df.columns))
    else:
        assert len(set(meta_keys) & set(df.columns)) == 0
    if ligands:
        assert "Ligand_ObjectID" in df.columns
