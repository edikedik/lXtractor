from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lXtractor.core.chain import ChainList, ChainStructure
from lXtractor.core.structure import GenericStructure
from lXtractor.util.seq import read_fasta
from lXtractor.variables.sequential import SeqEl
from lXtractor.variables.structural import Dist, PseudoDihedral
from test.common import sample_chain

DATA = Path(__file__).parent / 'data'


@pytest.fixture(scope='module')
def simple_structure_path() -> Path:
    path = DATA / '1aki.pdb'
    assert path.exists()
    return path


@pytest.fixture(scope='module')
def four_chain_structure_path() -> Path:
    path = DATA / '3i6x.pdb'
    assert path.exists()
    return path


@pytest.fixture(scope='module')
def four_chain_structure_seq_path() -> Path:
    path = DATA / 'P46940.fasta'
    assert path.exists()
    return path


@pytest.fixture(scope='module')
def chicken_src_str_path() -> Path:
    path = DATA / '2oiq.cif'
    assert path.exists()
    return path


@pytest.fixture(scope='module')
def chicken_src_seq_path() -> Path:
    path = DATA / 'P00523.fasta'
    assert path.exists()
    return path


@pytest.fixture(scope='module')
def human_src_seq_path() -> Path:
    path = DATA / 'P12931.fasta'
    assert path.exists()
    return path


@pytest.fixture(scope='module')
def pkinase_hmm_path() -> Path:
    path = DATA / 'Pkinase.hmm'
    assert path.exists()
    return path


@pytest.fixture(scope='module')
def simple_structure(simple_structure_path) -> GenericStructure:
    return GenericStructure.read(simple_structure_path)


@pytest.fixture(scope='module')
def four_chain_structure(four_chain_structure_path) -> GenericStructure:
    return GenericStructure.read(four_chain_structure_path)


@pytest.fixture(scope='module')
def chicken_src_str(chicken_src_str_path) -> GenericStructure:
    return GenericStructure.read(chicken_src_str_path)


@pytest.fixture(scope='module')
def chicken_src_seq(chicken_src_seq_path) -> tuple[str, str]:
    return next(read_fasta(chicken_src_seq_path))


@pytest.fixture(scope='module')
def human_src_seq(human_src_seq_path) -> tuple[str, str]:
    return next(read_fasta(human_src_seq_path))


@pytest.fixture(scope='module')
def simple_fasta_path() -> Path:
    path = DATA / 'simple.fasta'
    assert path.exists()
    return path


@pytest.fixture(scope='module')
def _sample_chain_list(simple_structure) -> ChainList:
    chain_str = ChainStructure.from_structure(simple_structure)
    return ChainList([
        sample_chain(prefix='c', structure=chain_str),
        sample_chain(prefix='k', structure=chain_str)
    ])


@pytest.fixture(scope='function')
def sample_chain_list(_sample_chain_list) -> ChainList:
    return deepcopy(_sample_chain_list)


@pytest.fixture(scope='module')
def simple_chain_variables() -> tuple[PseudoDihedral, Dist, SeqEl]:
    return PseudoDihedral(1, 2, 3, 4), Dist(1, 40, 'CB', 'CB'), SeqEl(1)
