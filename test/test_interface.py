import operator as op

import biotite.structure as bst
import numpy as np
import rustworkx as rx
import pytest

from lXtractor.core import GenericStructure, Interface
from lXtractor.core.exceptions import MissingData, AmbiguousData
from lXtractor.core.interface import RetainCondition
from test.common import STRUCTURES


def test_invalid():
    path = STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz"
    gs = GenericStructure.read(path)
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
    "path,partners,cutoff",
    [
        (STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz", "A_B", 1.0),
        (STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz", "A_B", 6.0),
        (STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz", "A_B", 10.0),
    ],
)
def test_interface_basic(path, partners, cutoff):
    gs = GenericStructure.read(path)
    iface = Interface(gs, partners, cutoff=cutoff)
    array = iface.parent_structure.array
    mask_a = iface._chain_atom_mask(iface.partners_a)
    mask_b = iface._chain_atom_mask(iface.partners_b)

    a, b = array[mask_a], array[mask_b]
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

    assert iface.num_contact_atoms() == mask_a.sum() + mask_b.sum()
    assert iface.num_contact_residues() == res_count_a + res_count_b
    assert (
        len(iface.get_contact_idx_a())
        == len(iface.get_contact_idx_b())
        == len(iface.get_contact_idx_ab())
    )


@pytest.mark.parametrize(
    "path,partners,as_",
    [
        (
            STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz",
            "A_B",
            ["nodes", "subgraph", "edges", "atoms"],
        )
    ],
)
def test_iter_single_cc(path, partners, as_):
    gs = GenericStructure.read(path)
    iface = Interface(gs, partners, cutoff=10)
    for method in as_:
        ccs = list(iface.iter_ccs(method))
        assert len(ccs) == 1
        cc = ccs.pop()
        if method in ["nodes", "atoms", "subgraph"]:
            assert len(cc) == iface.num_contact_atoms()
        else:
            assert len(cc) == iface.num_contacts()


@pytest.mark.parametrize("into_pairs", [True, False])
@pytest.mark.parametrize("target", ["cc", "chains"])
@pytest.mark.parametrize(
    "path,partners,cutoff,op,condition_a,condition_b,num_results",
    [
        (
            STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz",
            "A_B",
            10,
            op.and_,
            RetainCondition(),
            RetainCondition(),
            1,
        ),
        (
            STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz",
            "A_B",
            10,
            op.and_,
            RetainCondition(min_res=100),
            RetainCondition(),
            0,
        ),
        (
            STRUCTURES / "mmtf.gz" / "2OIQ.mmtf.gz",
            "A_B",
            10,
            op.or_,
            RetainCondition(),
            RetainCondition(min_res=100),
            1,
        ),
    ],
)
def test_split_single_cc_single_pair(
    path,
    partners,
    cutoff,
    op,
    condition_a,
    condition_b,
    num_results,
    target,
    into_pairs,
):
    gs = GenericStructure.read(path)
    iface = Interface(gs, partners, cutoff=10)

    splits = list(
        iface.split_connected(condition_a, condition_b, conditions_op=op,
                              conditions_apply_to=target, into_pairs=into_pairs,
                              cutoff=cutoff)
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
