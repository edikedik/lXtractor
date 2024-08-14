import operator as op
from copy import deepcopy

import biotite.structure as bst
import numpy as np
import rustworkx as rx
import pytest

from lXtractor.core import GenericStructure, Interface
from lXtractor.core.exceptions import MissingData, AmbiguousData
from lXtractor.core.interface import RetainCondition, InterfaceComparator
from test.common import STRUCTURES


@pytest.fixture()
def generic_structure_2oiq() -> GenericStructure:
    return GenericStructure.read(STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz")


@pytest.fixture()
def interface_2oiq(generic_structure_2oiq) -> Interface:
    return Interface(generic_structure_2oiq, "A_B")


@pytest.fixture()
def interface_2oiq_10(generic_structure_2oiq):
    return Interface(generic_structure_2oiq, "A_B", cutoff=10)


def test_invalid(generic_structure_2oiq):
    gs = generic_structure_2oiq
    with pytest.raises(MissingData):
        _ = Interface(gs, "AB_C")
    with pytest.raises(AmbiguousData):
        _ = Interface(gs, "AB_B")
    graph = rx.PyGraph()
    with pytest.raises(ValueError):
        _ = Interface(gs, "A_B", graph=graph)
    iface = Interface(gs, "A_B")
    graph = iface.G.copy()
    for node in graph.nodes():
        node.idx -= 1
    with pytest.raises(ValueError):
        _ = Interface(gs, "A_B", graph=graph)


@pytest.mark.parametrize(
    "cutoff",
    [1.0, 6.0, 10.0],
)
def test_interface_basic(generic_structure_2oiq, cutoff):
    gs = generic_structure_2oiq
    iface = Interface(gs, "A_B", cutoff=cutoff)
    array = iface.parent_structure.array
    mask_a = iface._chain_atom_mask(iface.partners_a)
    mask_b = iface._chain_atom_mask(iface.partners_b)

    solvent_mask = generic_structure_2oiq.mask.solvent

    a, b = array[mask_a & ~solvent_mask], array[mask_b & ~solvent_mask]
    d_a = np.linalg.norm(b.coord[:, np.newaxis] - a.coord, axis=-1)
    d_b = np.linalg.norm(a.coord[:, np.newaxis] - b.coord, axis=-1)
    d_min_a = np.min(d_a, axis=0)
    d_min_b = np.min(d_b, axis=0)
    mask_a = d_min_a <= cutoff
    mask_b = d_min_b <= cutoff

    num_atoms_a = np.sum(mask_a)
    num_atoms_b = np.sum(mask_b)
    res_count_a = 0 if num_atoms_a == 0 else bst.get_residue_count(a[mask_a])
    res_count_b = 0 if num_atoms_b == 0 else bst.get_residue_count(b[mask_b])

    assert iface.count_contact_atoms() == mask_a.sum() + mask_b.sum()
    assert iface.count_contact_residues() == res_count_a + res_count_b
    assert (
        len(iface.get_contact_idx_a())
        == len(iface.get_contact_idx_b())
        == len(iface.get_contact_idx())
    )


@pytest.mark.parametrize(
    "as_",
    ["nodes", "subgraph", "edges", "atoms"],
)
def test_iter_single_cc(interface_2oiq_10, as_):
    iface = interface_2oiq_10
    ccs = list(iface.iter_ccs(as_))
    assert len(ccs) == 1
    cc = ccs.pop()
    if as_ in ["nodes", "atoms", "subgraph"]:
        assert len(cc) == iface.count_contact_atoms()
    else:
        assert len(cc) == iface.count_contacts()


@pytest.mark.parametrize("into_pairs", [True, False])
@pytest.mark.parametrize("target", ["cc", "chains"])
@pytest.mark.parametrize(
    "op,condition_a,condition_b,num_results",
    [
        (
            op.and_,
            RetainCondition(),
            RetainCondition(),
            1,
        ),
        (
            op.and_,
            RetainCondition(min_res=100),
            RetainCondition(),
            0,
        ),
        (
            op.or_,
            RetainCondition(),
            RetainCondition(min_res=100),
            1,
        ),
    ],
)
def test_split_single_cc_single_pair(
    interface_2oiq_10,
    op,
    condition_a,
    condition_b,
    num_results,
    target,
    into_pairs,
):
    iface = interface_2oiq_10

    splits = list(
        iface.split_connected(
            condition_a,
            condition_b,
            conditions_op=op,
            conditions_apply_to=target,
            into_pairs=into_pairs,
            cutoff=iface.cutoff,
        )
    )
    assert len(splits) == num_results


def test_split_7fsh():
    def wrap_partners(iface):
        a = "".join(sorted(iface.partners_a))
        b = "".join(sorted(iface.partners_b))
        return f"{a}_{b}"

    path = STRUCTURES / "cif" / "7fsh.cif"
    gs = GenericStructure.read(path)

    iface = Interface(gs, "AC_BD", cutoff=6)
    pair_ifaces = iface.split_connected(into_pairs=True)
    partners = sorted(map(wrap_partners, pair_ifaces))
    assert partners == ["A_B", "C_D"]

    iface = Interface(gs, "A_CD", cutoff=6)
    chain_ifaces = iface.split_connected()
    assert sorted(map(wrap_partners, chain_ifaces)) == ["A_C"]

    iface = Interface(gs, "A_CD", cutoff=6)
    chain_ifaces = list(iface.split_connected(subset_parent_to_partners=True))
    assert len(chain_ifaces) == 1
    iface_subset = chain_ifaces.pop()
    chain_ids = bst.get_chains(iface_subset.parent_structure.array)
    assert set(chain_ids) == {"A", "C"}

    iface = Interface(gs, "A_CD", cutoff=6)
    chain_ifaces = list(iface.split_connected(subset_parent_to_partners=False))
    assert len(chain_ifaces) == 1
    iface_subset = chain_ifaces.pop()
    chain_ids = bst.get_chains(iface_subset.parent_structure.array)
    assert set(chain_ids) == {"A", "C", "D"}


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("name", [None, "A_B"])
@pytest.mark.parametrize("additional_meta", [None, True, {"A": "B"}])
def test_io(interface_2oiq, overwrite, name, tmp_path, additional_meta):
    iface = interface_2oiq
    dest = iface.write(
        tmp_path, overwrite=overwrite, name=name, additional_meta=additional_meta
    )
    if name is None:
        assert dest.name == iface.id
    else:
        assert dest.name == name

    iface_ = Interface.read(dest)
    assert iface_ == iface
    assert iface_.count_contact_residues(["A"]) == iface.count_contact_residues(["A"])


def test_sasa(interface_2oiq):
    iface = interface_2oiq
    sasa = iface.sasa()
    assert all(
        x > 0
        for x in [
            sasa.a_free,
            sasa.b_free,
            sasa.complex,
            sasa.a_complex,
            sasa.b_complex,
            sasa.bsa_complex,
            sasa.bsa_a,
            sasa.bsa_b,
        ]
    )
    assert sasa.a_free > sasa.a_complex
    assert sasa.b_free > sasa.b_complex
    assert (sasa.bsa_complex - sasa.bsa_a - sasa.bsa_b) < 1e-1


def test_counters(interface_2oiq):
    iface = interface_2oiq
    res_total = iface.count_contact_residues()
    res_a, res_b = map(
        lambda x: iface.count_contact_residues(x, strict=True), ["a", "b"]
    )
    assert res_a + res_b == res_total

    atoms_total = iface.count_contact_atoms()
    atoms_a, atoms_b = map(
        lambda x: iface.count_contact_atoms(x, strict=True), ["a", "b"]
    )
    assert atoms_a + atoms_b == atoms_total


def test_comparator_equal(interface_2oiq):
    comp = InterfaceComparator(interface_2oiq, interface_2oiq)
    assert comp.irmsd() < 1e-3
    assert comp.lrmsd() < 1e-3
    assert comp.fnat() == 1
    assert abs(comp.dockq() - 1) < 1e-3


def test_comparator_translated_full(interface_2oiq):
    p = interface_2oiq.parent_structure
    a = bst.translate(p.array, [10, 10, 10])
    gs = GenericStructure(a, p.name, atom_marks=p.atom_marks, graph=p.graph)
    iface_translated = Interface(gs, interface_2oiq.partners)
    comp = InterfaceComparator(interface_2oiq, iface_translated)
    assert comp.irmsd() < 1e-3
    assert comp.lrmsd() < 1e-3
    assert comp.fnat() == 1
    assert abs(comp.dockq() - 1) < 1e-3


def test_comparator_translated_b(interface_2oiq):
    p = interface_2oiq.parent_structure
    a = p.array.copy()
    mask = np.isin(p.array.chain_id, interface_2oiq.partners_b)
    coord = a.coord
    coord[mask] += 100
    a.coord = coord
    gs = GenericStructure(a, p.name, atom_marks=p.atom_marks, graph=p.graph)
    assert len(p) == len(gs)
    assert np.all(p.array.atom_name == gs.array.atom_name)
    iface_translated = Interface(gs, interface_2oiq.partners)
    comp = InterfaceComparator(interface_2oiq, iface_translated)
    assert comp.irmsd() > 50
    assert comp.lrmsd() > 25
    assert comp.fnat() == 0
    assert comp.dockq() < 1e-2