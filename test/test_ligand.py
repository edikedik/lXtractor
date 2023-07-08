from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest
from toolz import valmap

from lXtractor.core.structure import GenericStructure

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "path,expected",
    [
        (DATA / "2oiq.cif", {("STI", ("A",))}),
        (DATA / "5hu9.cif", {("66K", ("A",))}),
        (DATA / "1aki.pdb", set()),
        (DATA / "3uec.cif", set()),
        (
            DATA / "7fsh.cif",
            {
                ("ALA", ("D",)),
                ("DGL", ("B", "D")),
                ("GLY", ("C", "D")),
                ("UXA", ("C", "D")),
            },
        ),
        (DATA / "3unk.cif", {("0BY", ("A",))}),
        (DATA / "4hvd.mmtf.gz", {("933", ("A",)), ("PHU", ("A",))}),
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
            DATA / "7m0y.mmtf.gz",
            {
                ("A", "ANP", "A"),
                ("A", "QOM", "B"),
                ("B", "QOM", "B"),
                ("B", "ANP", "B"),
            },
        ),
        (
            DATA / "7fsh.cif",
            {
                ("B", "DGL", "B"),
                ("C", "GLY", "C"),
                ("C", "UXA", "C"),
                ("D", "ALA", "D"),
                ("D", "DGL", "D"),
                ("D", "GLY", "D"),
                ("D", "UXA", "D"),
            },
        ),
        (
            DATA / '4TWC.mmtf.gz',
            {
                ("A", "37J", "A"),
                ("B", "37J", "B"),
                ("A", "BOG", "A"),
                ("B", "BOG", "A"),
            }
        )
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


@pytest.mark.parametrize("inp", [DATA / "3unk.cif"])
@pytest.mark.parametrize("meta", [True, False])
def test_summary(inp, meta):
    s = GenericStructure.read(inp, ligands=True)
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
