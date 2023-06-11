from collections import defaultdict
from pathlib import Path

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
    ligands = s.ligands
    d = defaultdict(set)
    for lig in ligands:
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
                ("D", "UXA", "C"),
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
