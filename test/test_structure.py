from copy import deepcopy
from pathlib import Path

import biotite.structure as bst
import pytest

from lXtractor.core.config import EMPTY_STRUCTURE_ID, AtomMark
from lXtractor.core.exceptions import LengthMismatch, MissingData, NoOverlap
from lXtractor.core.structure import (
    GenericStructure,
    NucleotideStructure,
    ProteinStructure,
)
from test.conftest import EPS

DATA = Path(__file__).parent / "data"


@pytest.fixture()
def peptide_dna_complex() -> Path:
    path = DATA / "1yrn.cif.gz"
    assert path.exists()
    return path


def test_init(simple_structure_path):
    s = GenericStructure.read(simple_structure_path)
    assert len(s) > 0


def test_degenerate(simple_structure):
    s = GenericStructure.make_empty()
    assert len(s) == 0
    assert s.is_empty
    assert s._name == EMPTY_STRUCTURE_ID

    assert len(list(s.get_sequence())) == 0
    assert len(list(s.split_chains())) == 0
    assert len(list(s.split_altloc())) == 1

    assert len(s.array) == 0

    assert len(s.mask.primary_polymer) == 0
    assert len(s.mask.ligand_pep) == 0
    assert len(s.mask.ligand_nuc) == 0
    assert len(s.mask.ligand_carb) == 0
    assert len(s.mask.ligand) == 0
    assert len(s.mask.solvent) == 0
    assert len(s.mask.unk) == 0

    with pytest.raises(NoOverlap):
        s.extract_segment(1, 1, "A")
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


@pytest.mark.parametrize("inp", [(DATA / "1aki.pdb", 1), (DATA / "1rdq.mmtf", 2)])
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
    seq_, _ = bst.get_residues(
        simple_structure.array[simple_structure.mask.primary_polymer]
    )
    assert len(seq) == len(seq_) == 129


def test_subsetting(simple_structure):
    sub = simple_structure.extract_segment(1, 2, simple_structure.chain_ids[0])
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


def test_atom_marks(peptide_dna_complex):
    s = GenericStructure.read(peptide_dna_complex)
    assert set(s.atom_marks) == {
        AtomMark.SOLVENT,
        AtomMark.PEP,
        AtomMark.NUC | AtomMark.LIGAND,
    }
    s = ProteinStructure.read(peptide_dna_complex)
    assert set(s.atom_marks) == {
        AtomMark.SOLVENT,
        AtomMark.PEP,
        AtomMark.NUC | AtomMark.LIGAND,
    }
    s = NucleotideStructure.read(peptide_dna_complex)
    assert set(s.atom_marks) == {
        AtomMark.SOLVENT,
        AtomMark.NUC,
        AtomMark.PEP | AtomMark.LIGAND,
    }


def test_atom_marks_splitting(peptide_dna_complex):
    s = GenericStructure.read(peptide_dna_complex)
    # Expecting four chains: two DNA strands and two protein chains
    chains = list(s.split_chains())
    assert len(chains) == 4
    # Expecting two protein chains; DNA strands become attached ligands
    chains = list(s.split_chains(polymer=True))
    assert len(chains) == 2


def test_atom_marks_extracting(peptide_dna_complex):
    s = GenericStructure.read(peptide_dna_complex)
    seg = s.extract_segment(110, 125, "A")
    # DNA becomes a primary polymer, while peptide segment becomes a ligand
    assert seg.chain_ids_polymer == ["C", "D"]
    assert seg.mask.ligand_pep.sum() > 0

    s = ProteinStructure.read(peptide_dna_complex)
    seg = s.extract_segment(110, 125, "A")
    # DNA stays being a polymer ligand since peptide is a primary polymer
    assert seg.chain_ids_polymer == ["A"]
    assert seg.chain_ids_ligand == ["C", "D"]
    assert seg.mask.unk.sum() == 0

    s = NucleotideStructure.read(peptide_dna_complex)
    seg = s.extract_segment(22, 30, "D")
    assert seg.chain_ids_polymer == ["D"]
    assert seg.chain_ids_ligand == ["A"]
    assert seg.mask.unk.sum() == 0

    # (!) No "stitch" mode implemented by default

    # seg = seg.extract_positions([22, 23, 24, 26, 28, 29, 30], "D")
    # assert seg.mask.unk.sum() == 0
    # Should recognize position 25 as part of the polymer
    # assert bst.get_residue_count(seg.array[seg.mask.primary_polymer]) == 7
