from collections import defaultdict

import pandas as pd
import pytest
from toolz import valmap

from lXtractor.core.structure import GenericStructure
from test.common import STRUCTURES


@pytest.mark.parametrize(
    "path,expected",
    [
        (STRUCTURES / "cif" / "2oiq.cif", {("STI", ("A",))}),
        (STRUCTURES / "cif" / "5hu9.cif", {("66K", ("A",))}),
        (STRUCTURES / "pdb" / "1aki.pdb", set()),
        (STRUCTURES / "cif" / "3uec.cif", set()),
        (
            STRUCTURES / "cif" / "7fsh.cif",
            {
                ("ALA", ("D",)),
                ("DGL", ("B", "D")),
                # ("GLY", ("C", "D")),  # num atoms too low
                ("UXA", ("C", "D")),
            },
        ),
        (STRUCTURES / "cif" / "3unk.cif", {("0BY", ("A",))}),
        (STRUCTURES / "mmtf.gz" / "4hvd.mmtf.gz", {("933", ("A",)), ("PHU", ("A",))}),
    ],
)
def test_find_ligands(path, expected):
    s = GenericStructure.read(path)
    d = defaultdict(set)
    for lig in s.ligands:
        d[lig.res_name] |= lig.parent_contact_chains
    d = valmap(lambda x: tuple(sorted(x)), d)
    assert set(d.items()) == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        (
            STRUCTURES / "mmtf.gz" / "7m0y.mmtf.gz",
            {
                ("A", "ANP", "A"),
                ("A", "QOM", "B"),
                ("B", "QOM", "B"),
                ("B", "ANP", "B"),
            },
        ),
        (
            STRUCTURES / "cif" / "7fsh.cif",
            {
                ("B", "DGL", "B"),
                # ("C", "GLY", "C"),
                ("C", "UXA", "C"),
                ("D", "ALA", "D"),
                ("D", "DGL", "D"),
                # ("D", "GLY", "D"),
                ("D", "UXA", "D"),
            },
        ),
        (
            STRUCTURES / "mmtf.gz" / "4TWC.mmtf.gz",
            {
                ("A", "37J", "A"),
                ("B", "37J", "B"),
                ("A", "BOG", "A"),
                ("B", "BOG", "A"),
            },
        ),
    ],
)
def test_split_chains(path, expected):
    # Test whether splitting chains correctly distributes connected ligands
    s = GenericStructure.read(path)
    outputs = []
    for c in s.split_chains():
        assert len(c.chain_ids_polymer) == 1
        structure_chain_id = c.chain_ids_polymer.pop()
        for lig in c.ligands:
            outputs.append((structure_chain_id, lig.res_name, lig.chain_id))
    assert set(outputs) == expected


@pytest.mark.parametrize(
    "inp_path,altloc,lig_names",
    [(STRUCTURES / "mmtf" / "1rdq.mmtf", ["A", "B"], [{"ADP_599:E"}, {"ATP_600:E"}])],
)
def test_split_altloc(inp_path, altloc, lig_names):
    s = GenericStructure.read(inp_path, altloc=True)
    for s_alt, alt_id, exp_lig in zip(s.split_altloc(), altloc, lig_names, strict=True):
        alt_ids = s_alt.altloc_ids
        assert len(alt_ids) == 2
        assert alt_ids[-1] == alt_id
        assert {lig.id.split("<-")[0] for lig in s_alt.ligands} == exp_lig


@pytest.mark.parametrize("inp", [STRUCTURES / "cif" / "3unk.cif"])
@pytest.mark.parametrize("meta", [True, False])
def test_summary(inp, meta):
    s = GenericStructure.read(inp)
    assert len(s.ligands) == 1
    lig = s.ligands[0]
    s = lig.summary(meta=meta)
    assert isinstance(s, pd.Series)
    assert "ObjectID" in s.index and "ParentID" in s.index
    meta_keys = list(lig.meta)
    if meta:
        assert set(meta_keys).issubset(set(s.index))
    else:
        assert len(set(meta_keys) & set(s.index)) == 0


@pytest.mark.parametrize(
    "inp,out", [(STRUCTURES / "mmtf.gz" / "5ACB.mmtf.gz", {("5I1", "C"), ("5I1", "D")})]
)
def test_covalent_ligand(inp, out):
    s = GenericStructure.read(inp)
    outputs = {(lig.res_name, lig.chain_id) for lig in s.ligands}
    assert outputs == out
