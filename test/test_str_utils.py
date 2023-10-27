from pathlib import Path
from tempfile import TemporaryDirectory

import biotite.structure as bst
import biotite.structure.info as bstinfo
import numpy as np
import pytest
import rustworkx as rx

from test.common import ALL_STRUCTURES
from lXtractor.core.config import STRUCTURE_FMT, EMPTY_ALTLOC
from lXtractor.util.structure import (
    filter_to_common_atoms,
    get_missing_atoms,
    get_observed_atoms_frac,
    load_structure,
    save_structure,
    iter_altloc_masks,
    to_graph,
)

DATA = Path(__file__).parent / "data"


@pytest.fixture()
def ala() -> bst.AtomArray:
    return bstinfo.residue("ALA")


@pytest.fixture()
def gly() -> bst.AtomArray:
    return bstinfo.residue("GLY")


@pytest.fixture()
def phe() -> bst.AtomArray:
    return bstinfo.residue("PHE")


def test_common_atoms_filter(ala, gly, phe, abl_str, src_str):
    m1, m2 = filter_to_common_atoms(ala, ala)
    assert m1.sum() == m2.sum() == len(ala)

    m1, m2 = filter_to_common_atoms(bst.array([*ala, *gly]), bst.array([*ala, *gly]))
    assert m1.sum() == m2.sum() == len(ala) + len(gly)

    m1, m2 = filter_to_common_atoms(ala[:5], ala)
    assert m1.sum() == 5

    m1, m2 = filter_to_common_atoms(ala, phe, allow_residue_mismatch=True)
    assert set(ala[m1].atom_name) == set(phe[m2].atom_name) == {"N", "C", "CA", "CB"}

    m1, m2 = filter_to_common_atoms(ala, gly, allow_residue_mismatch=True)
    assert set(ala[m1].atom_name) == set(gly[m2].atom_name) == {"N", "C", "CA"}

    m1, m2 = filter_to_common_atoms(
        bst.array([*gly, *ala]), bst.array([*ala, *gly]), allow_residue_mismatch=True
    )
    assert m1.sum() == m2.sum() == 6

    abl_hrd = abl_str.spawn_child(136, 140, "LHRDL")
    src_hrd = src_str.spawn_child(128, 132, "VHRDL")

    assert len(abl_hrd._seq) == len(src_hrd._seq) == 5

    with pytest.raises(ValueError):
        filter_to_common_atoms(abl_hrd.array, src_hrd.array)

    m1, m2 = filter_to_common_atoms(
        abl_hrd.array, src_hrd.array, allow_residue_mismatch=True
    )

    a1, a2 = abl_hrd.array[m1], src_hrd.array[m2]
    rs1, rs2 = map(lambda x: list(bst.residue_iter(x)), [a1, a2])

    assert len(a1) == len(a2)
    assert len(rs1) == len(rs2) == 5

    for i, pair in enumerate(zip(bst.residue_iter(a1), bst.residue_iter(a2))):
        r1, r2 = pair
        if i == 0:
            assert len(r1) == len(r2) == 4
        else:
            assert len(r1) == len(r2) > 4


def test_missing_atoms_getter(gly, ala):
    res = list(get_missing_atoms(gly))
    assert len(res) == 1
    res = res.pop()
    assert len(res) == 0

    gly = gly[gly.atom_name != "CA"]
    res = list(get_missing_atoms(bst.array([*gly, *ala])))
    assert len(res) == 2
    assert len(res[0]) == 1 and len(res[1]) == 0
    assert res[0][0] == "CA"

    gly.res_name = np.array(["?"] * len(gly))
    res = list(get_missing_atoms(gly))
    assert len(res) == 1 and res[0] is None


def test_observed_atoms_fraction(gly, ala):
    res = list(get_observed_atoms_frac(gly))
    assert len(res) == 1
    assert res.pop() == 1.0

    gly_size = len(gly[(gly.element != "H") & (gly.atom_name != "OXT")])
    gly = gly[gly.atom_name != "CA"]
    expected_frac = (gly_size - 1) / gly_size
    res = next(get_observed_atoms_frac(gly))
    assert res == expected_frac


@pytest.mark.parametrize(
    "path",
    [
        DATA / "1aki.pdb",
        DATA / "2oiq.cif",
        DATA / "2oiq.cif.gz",
        DATA / "1rdq.mmtf",
        DATA / "1rdq.mmtf.gz",
    ],
)
def test_load_structure(path):
    suffix = path.suffix if not path.suffix == ".gz" else path.suffixes[-2]
    fmt = suffix.removeprefix(".")
    gz = path.suffix == ".gz"
    mode = "rb" if gz or fmt == "mmtf" else "r"

    assert isinstance(load_structure(path), bst.AtomArray)

    with path.open(mode) as f:
        content = f.read()
        f.seek(0)
        assert isinstance(load_structure(f, fmt, gz=gz), bst.AtomArray)
        assert isinstance(load_structure(content, fmt, gz=gz), bst.AtomArray)


@pytest.mark.parametrize("fmt", [*STRUCTURE_FMT, *(x + ".gz" for x in STRUCTURE_FMT)])
def test_save_structure(simple_structure, fmt):
    s = simple_structure
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / f"structure.{fmt}"
        save_structure(s.array, path)
        sl = load_structure(path)
        assert isinstance(sl, bst.AtomArray)


@pytest.mark.parametrize("inp_path", ALL_STRUCTURES)
def test_iter_altloc_masks(inp_path):
    a = load_structure(inp_path, altloc="all")
    ids = set(a.altloc_id)
    masks = list(iter_altloc_masks(a))
    assert (
        len(ids) == 1
        and ids.issubset(EMPTY_ALTLOC)
        or len(masks) == len(set(a.altloc_id)) - 1
    )
    m = np.vstack(masks)
    assert np.all(np.any(m, axis=0))


@pytest.mark.parametrize("inp_path", ALL_STRUCTURES)
def test_graph_construction(inp_path):
    a = load_structure(inp_path)
    g = to_graph(a)
    assert isinstance(g, rx.PyGraph)
