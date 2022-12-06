from copy import deepcopy
from itertools import repeat

import pytest

from lXtractor.core.exceptions import LengthMismatch, MissingData
from lXtractor.core.structure import GenericStructure
from test.conftest import EPS


def test_init(simple_structure_path):
    s = GenericStructure.read(simple_structure_path)
    assert len(s.array) > 0


def test_split(simple_structure):
    s = simple_structure
    chains = list(s.split_chains())
    assert len(chains) == 1
    assert len(chains[0].array) == len(s.array)
    assert isinstance(chains.pop(), GenericStructure)


def test_sequence(simple_structure):
    seq = list(simple_structure.get_sequence())
    assert len(seq) == 207


def test_subsetting(simple_structure):
    sub = simple_structure.sub_structure(1, 2)
    seq = list(sub.get_sequence())
    assert len(seq) == 2


def test_write(simple_structure):
    # TODO: implement when providing paths is fixed in biotite
    pass


def test_superpose(chicken_src_str):
    a = chicken_src_str
    bb_atoms = ['N', 'CA', 'C']

    _, rmsd, _ = a.superpose(a)
    assert rmsd <= EPS

    a_cp = deepcopy(a)
    a_cp.array.res_id += 1

    # align backbone of the first three residues

    _, rmsd, _ = a.superpose(
        a_cp, res_id_self=[256, 257, 258], res_id_other=[257, 258, 259])
    assert rmsd < EPS

    _, rmsd, _ = a.superpose(
        a_cp, res_id_self=[256, 257, 258], res_id_other=[257, 258, 259],
        atom_names_self=bb_atoms,
        atom_names_other=bb_atoms,
    )
    assert rmsd < EPS

    with pytest.raises(LengthMismatch):
        _ = a.superpose(a, res_id_self=[256])

    with pytest.raises(MissingData):
        _ = a.superpose(a, res_id_self=[0], res_id_other=[0])
