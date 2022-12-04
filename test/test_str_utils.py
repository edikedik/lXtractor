import biotite.structure as bst
import biotite.structure.info as bstinfo
import pytest

from lXtractor.util.structure import filter_to_common_atoms


@pytest.fixture()
def ala() -> bst.AtomArray: return bstinfo.residue('ALA')


@pytest.fixture()
def gly() -> bst.AtomArray: return bstinfo.residue('GLY')

@pytest.fixture()
def phe() -> bst.AtomArray: return bstinfo.residue('PHE')


def test_common_atoms_filter(ala, gly, phe):
    m1, m2 = filter_to_common_atoms(ala, ala)
    assert m1.sum() == m2.sum() == len(ala)

    m1, m2 = filter_to_common_atoms(bst.array([*ala, *gly]), bst.array([*ala, *gly]))
    assert m1.sum() == m2.sum() == len(ala) + len(gly)


    m1, m2 = filter_to_common_atoms(ala[:5], ala)
    assert m1.sum() == 5

    m1, m2 = filter_to_common_atoms(ala, phe, allow_residue_mismatch=True)
    assert set(ala[m1].atom_name) == set(phe[m2].atom_name) == {'N', 'C', 'CA', 'CB'}

    m1, m2 = filter_to_common_atoms(ala, gly, allow_residue_mismatch=True)
    assert set(ala[m1].atom_name) == set(gly[m2].atom_name) == {'N', 'C', 'CA'}

    m1, m2 = filter_to_common_atoms(
        bst.array([*gly, *ala]), bst.array([*ala, *gly]), allow_residue_mismatch=True
    )
    assert m1.sum() == m2.sum() == 6