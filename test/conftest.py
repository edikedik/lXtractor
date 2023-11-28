from __future__ import annotations

from pathlib import Path

import pytest

from lXtractor.chain import ChainList, ChainStructure, ChainSequence, Chain
from lXtractor.core.config import DefaultConfig
from lXtractor.core.structure import GenericStructure
from lXtractor.util.seq import read_fasta
from lXtractor.variables.sequential import SeqEl
from lXtractor.variables.structural import Dist, PseudoDihedral
from test.common import sample_chain, get_fst_chain

DATA = Path(__file__).parent / "data"
EPS = 10e-5


@pytest.fixture(scope="module")
def simple_structure_path() -> Path:
    path = DATA / "1aki.pdb"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def four_chain_structure_path() -> Path:
    path = DATA / "3i6x.pdb"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def four_chain_structure_seq_path() -> Path:
    path = DATA / "P46940.fasta"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def chicken_src_str_path() -> Path:
    path = DATA / "2oiq.cif"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def chicken_src_seq_path() -> Path:
    path = DATA / "P00523.fasta"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def human_src_seq_path() -> Path:
    path = DATA / "P12931.fasta"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def pkinase_hmm_path() -> Path:
    path = DATA / "Pkinase.hmm"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def human_abl_str_path() -> Path:
    path = DATA / "5hu9.cif"
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def simple_structure(simple_structure_path) -> GenericStructure:
    return GenericStructure.read(simple_structure_path)


@pytest.fixture(scope="module")
def four_chain_structure(four_chain_structure_path) -> GenericStructure:
    return GenericStructure.read(four_chain_structure_path)


@pytest.fixture(scope="module")
def chicken_src_str(chicken_src_str_path) -> GenericStructure:
    return GenericStructure.read(chicken_src_str_path)


@pytest.fixture(scope="module")
def human_abl_str(human_abl_str_path) -> GenericStructure:
    return GenericStructure.read(human_abl_str_path)


@pytest.fixture()
def chicken_src_seq(chicken_src_seq_path) -> tuple[str, str]:
    return next(read_fasta(chicken_src_seq_path))


@pytest.fixture()
def human_src_seq(human_src_seq_path) -> tuple[str, str]:
    return next(read_fasta(human_src_seq_path))


@pytest.fixture(scope="module")
def simple_fasta_path() -> Path:
    path = DATA / "simple.fasta"
    assert path.exists()
    return path


@pytest.fixture()
def sample_chain_list(simple_structure) -> ChainList:
    chain_str = ChainStructure(simple_structure)
    return ChainList(
        [
            sample_chain(prefix="c", structure=chain_str),
            sample_chain(prefix="k", structure=chain_str),
        ]
    )


@pytest.fixture()
def abl_str(human_abl_str):
    return get_fst_chain(human_abl_str)


@pytest.fixture()
def src_str(chicken_src_str):
    return get_fst_chain(chicken_src_str)


# @pytest.fixture(scope='function')
# def sample_chain_list(_sample_chain_list) -> ChainList:
#     return deepcopy(_sample_chain_list)


@pytest.fixture(scope="module")
def simple_chain_variables() -> tuple[PseudoDihedral, Dist, SeqEl]:
    return PseudoDihedral(1, 2, 3, 4), Dist(1, 40, "CB", "CB"), SeqEl(1)


@pytest.fixture()
def simple_chain_seq() -> ChainSequence:
    seq1 = DefaultConfig["mapnames"]["seq1"]
    s = ChainSequence(1, 5, "S", {seq1: "AAAAA"})
    return s


@pytest.fixture()
def simple_chain_structure(simple_structure) -> ChainStructure:
    return ChainStructure(simple_structure)


@pytest.fixture
def src_chain(chicken_src_seq, src_str, abl_str) -> Chain:
    c = Chain.from_seq(chicken_src_seq)
    c.add_structure(src_str)
    c.add_structure(abl_str)
    return c


@pytest.fixture(scope="session")
def fake_chain_dump(tmp_path_factory) -> tuple[Path, dict[str, Path]]:
    names = DefaultConfig["filenames"]
    base = tmp_path_factory.mktemp("base", numbered=False)
    paths = dict(
        base=base,
        X=base / "segments" / "X",
        x1=base / "segments" / "X" / "segments" / "x",
        Y=base / "segments" / "Y",
        x2=base / "segments" / "Y" / "segments" / "x",
        s1=base / "structures" / "s1",
        s2=base / "structures" / "s2",
    )

    for name in ["x1", "x2", "s1", "s2"]:
        p = paths[name]
        p.mkdir(parents=True)

    for name in ["base", "X", "x1", "Y", "s1", "s2"]:
        p = paths[name]
        open(p / names["meta"], "w").close()
        open(p / names["sequence"], "w").close()

    open(paths["s1"] / f"{names['structure_base_name']}.pdb", "w").close()

    return base, paths
