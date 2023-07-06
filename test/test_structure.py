from copy import deepcopy
from pathlib import Path

import biotite.structure as bst
import pytest

from lXtractor.core.config import EMPTY_STRUCTURE_ID
from lXtractor.core.exceptions import LengthMismatch, MissingData, NoOverlap
from lXtractor.core.structure import GenericStructure
from test.conftest import EPS

DATA = Path(__file__).parent / "data"


def test_init(simple_structure_path):
    s = GenericStructure.read(simple_structure_path)
    assert len(s) > 0


def test_degenerate(simple_structure):
    s = GenericStructure.make_empty()
    assert len(s) == 0
    assert s.is_empty
    assert s._structure_id == EMPTY_STRUCTURE_ID

    assert len(list(s.get_sequence())) == 0
    assert len(list(s.split_chains())) == 0

    with pytest.raises(NoOverlap):
        s.extract_segment(1, 1)
    with pytest.raises(MissingData):
        _ = simple_structure.superpose(s)
    with pytest.raises(MissingData):
        _ = s.superpose(simple_structure)
    with pytest.raises(MissingData):
        _ = s.superpose(s)


def test_split_chains(simple_structure):
    s = simple_structure
    chains = list(s.split_chains())
    assert len(chains) == 1
    assert len(chains[0]) == len(s)
    assert isinstance(chains.pop(), GenericStructure)


@pytest.mark.parametrize(
    "inp", [(DATA / "1aki.pdb", [""]), (DATA / "1rdq.mmtf", ["", "A", "B"])]
)
def test_read_altloc(inp):
    path, expected_ids = inp
    s = GenericStructure.read(path, altloc=True)
    assert s.altloc_ids == expected_ids


@pytest.mark.parametrize(
    "inp", [(DATA / "1aki.pdb", 1), (DATA / "1rdq.mmtf", 2)]
)
def test_split_altloc(inp):
    path, n_chains = inp
    s = GenericStructure.read(path, altloc=True)
    chains = list(s.split_altloc())
    assert len(chains) == n_chains
    s = GenericStructure.read(path, altloc=False)
    chains = list(s.split_altloc())
    assert len(chains) == 1


def test_sequence(simple_structure):
    seq = list(simple_structure.get_sequence())
    seq_, _ = bst.get_residues(simple_structure.array_polymer)
    assert len(seq) == len(seq_) == 129


def test_subsetting(simple_structure):
    sub = simple_structure.extract_segment(1, 2)
    seq = list(sub.get_sequence())
    assert len(seq) == 2


def test_write(simple_structure):
    # TODO: implement when providing paths is fixed in biotite
    pass


def test_superpose(chicken_src_str):
    a = chicken_src_str
    bb_atoms = ["N", "CA", "C"]

    _, rmsd, _ = a.superpose(a)
    assert rmsd <= EPS

    a_cp = deepcopy(a)
    a_cp.array.res_id += 1

    # align backbone of the first three residues

    _, rmsd, _ = a.superpose(
        a_cp, res_id_self=[256, 257, 258], res_id_other=[257, 258, 259]
    )
    assert rmsd < EPS

    _, rmsd, _ = a.superpose(
        a_cp,
        res_id_self=[256, 257, 258],
        res_id_other=[257, 258, 259],
        atom_names_self=bb_atoms,
        atom_names_other=bb_atoms,
    )
    assert rmsd < EPS

    with pytest.raises(LengthMismatch):
        _ = a.superpose(a, res_id_self=[256])

    with pytest.raises(MissingData):
        _ = a.superpose(a, res_id_self=[0], res_id_other=[0])
