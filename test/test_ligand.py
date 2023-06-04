from collections import defaultdict
from pathlib import Path

import pytest
from toolz import valmap

from lXtractor.core.structure import GenericStructure

DATA = Path(__file__).parent / 'data'


@pytest.mark.parametrize(
    'inp_id,expected',
    [
        (DATA / '2oiq.cif', {('STI', ('A',))}),
        (DATA / '5hu9.cif', {('66K', ('A',))}),
        (DATA / '1aki.pdb', set()),
        (DATA / '3uec.cif', set()),
        (
            DATA / '7fsh.cif',
            {
                ('ALA', ('D',)),
                ('DGL', ('B', 'D')),
                ('GLY', ('C', 'D')),
                ('UXA', ('C', 'D')),
            },
        ),
        (DATA / '3unk.cif', {('0BY', ('A',))}),
    ],
)
def test_find_ligands(inp_id, expected):
    s = GenericStructure.read(inp_id)
    ligands = s.ligands
    d = defaultdict(set)
    for lig in ligands:
        d[lig.res_name] |= lig.parent_contact_chains
    d = valmap(lambda x: tuple(sorted(x)), d)
    assert set(d.items()) == expected
