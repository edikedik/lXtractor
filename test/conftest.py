from __future__ import annotations

from pathlib import Path

import pytest

from lXtractor.core.structure import Structure
from lXtractor.util.seq import read_fasta

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
def simple_structure(simple_structure_path) -> Structure:
    return Structure.read(simple_structure_path)


@pytest.fixture(scope='module')
def four_chain_structure(four_chain_structure_path) -> Structure:
    return Structure.read(four_chain_structure_path)


@pytest.fixture(scope='module')
def chicken_src_str(chicken_src_str_path) -> Structure:
    return Structure.read(chicken_src_str_path)


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
