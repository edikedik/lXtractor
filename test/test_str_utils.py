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


def test_common_atoms_filter(ala, gly, phe, abl_str, src_str):
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

    abl_hrd = abl_str.spawn_child(136, 140, 'LHRDL')
    src_hrd = src_str.spawn_child(128, 132, 'VHRDL')

    assert len(abl_hrd.seq) == len(src_hrd.seq) == 5

    with pytest.raises(ValueError):
        filter_to_common_atoms(abl_hrd.array, src_hrd.array)

    m1, m2 = filter_to_common_atoms(abl_hrd.array, src_hrd.array, allow_residue_mismatch=True)

    a1, a2 = abl_hrd.array[m1], src_hrd.array[m2]
    rs1, rs2 = map(lambda x: list(bst.residue_iter(x)), [a1, a2])

    assert len(a1) == len(a2)
    assert len(rs1) == len(rs2) == 5

    for i, pair in enumerate(zip(bst.residue_iter(a1), bst.residue_iter(a2))):
        r1, r2 = pair
        if i == 0:
            assert len(r1) == len(r2) == 4
        else:
            print(r1, r2)
            assert len(r1) == len(r2) > 4