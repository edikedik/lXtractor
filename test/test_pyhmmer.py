from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pyhmmer.easel import DigitalSequence, TextMSA
from pyhmmer.plan7 import HMMFile, HMM

from lXtractor.chain import ChainStructure, ChainSequence
from lXtractor.core.exceptions import MissingData
from lXtractor.ext.hmm import PyHMMer


def duplicate_file_text(p: Path) -> str:
    text = p.read_text()
    sep = "" if text.endswith("\n") else "\n"
    return f"{text}{sep}{text}"


def test_init_multiple(pkinase_hmm_path):
    text = duplicate_file_text(pkinase_hmm_path)
    with NamedTemporaryFile("w") as f:
        print(text, file=f)
        f.seek(0)
        hmm_file = HMMFile(f.name)
        hmms = list(PyHMMer.from_hmm_collection(hmm_file))
        assert len(hmms) == 2
        hmms = list(PyHMMer.from_hmm_collection(Path(f.name)))
        assert len(hmms) == 2
        hmms = list(PyHMMer.from_hmm_collection(hmms[0].hmm))
        assert len(hmms) == 1


def test_init(pkinase_hmm_path):
    assert PyHMMer(pkinase_hmm_path).hmm is not None
    assert PyHMMer(str(pkinase_hmm_path)).hmm is not None
    with HMMFile(pkinase_hmm_path) as f:
        assert PyHMMer(next(f)).hmm is not None
    assert PyHMMer(HMMFile(pkinase_hmm_path)).hmm is not None

    with NamedTemporaryFile("w") as f:
        with pytest.raises(MissingData):
            PyHMMer(f.name)


def test_convert(pkinase_hmm_path, abl_str, simple_chain_seq):
    ann = PyHMMer(pkinase_hmm_path)
    chain_seq = simple_chain_seq
    for c in [abl_str, chain_seq, "AAAAA"]:
        assert isinstance(ann.convert_seq(c), DigitalSequence)


def test_domain_extraction(chicken_src_str, pkinase_hmm_path):
    seqs = [
        ChainStructure(chain_str, "2oiq")._seq
        for chain_str in chicken_src_str.split_chains()
    ]
    annotator = PyHMMer(pkinase_hmm_path, bit_cutoffs="trusted")
    extracted = list(annotator.annotate(seqs, "PK", keep=False))
    assert len(extracted) == 2
    assert len(extracted[0]) == 232
    assert len(extracted[1]) == 230
    assert all(not s.children for s in seqs)
    s = extracted[0]
    for meta_name in ["pvalue", "score", "bias", "cov_seq", "cov_hmm"]:
        assert f"PK_{meta_name}" in s.meta

    should_miss = [
        {"min_score": 300},
        {"min_size": 300},
        {"min_cov_hmm": 1.0},
        {"min_cov_seq": 1.0},
    ]
    for param in should_miss:
        assert not list(annotator.annotate(seqs, keep=False, **param))

    with pytest.raises(MissingData):
        list(annotator.annotate([]))


def test_align(human_src_seq, pkinase_hmm_path):
    annotator = PyHMMer(pkinase_hmm_path)
    seq_name, seq_str = human_src_seq
    cs = ChainSequence.from_string(seq_str, name=seq_name)
    seqs = [cs, human_src_seq, seq_str]
    msa = annotator.align(seqs)
    assert isinstance(msa, TextMSA)
    obtained_names = [x.decode("utf-8") for x in msa.names]
    expected_names = [cs.name, seq_name, str(hash(seq_str))]
    assert obtained_names == expected_names


def test_from_msa():
    seqs = [("seq1", "AAAAA"), ("seq2", "AAA-A")]
    pyhmm = PyHMMer.from_msa(seqs, "test", "amino")
    # successfully created an HMM with five nodes
    assert isinstance(pyhmm.hmm, HMM)
    assert pyhmm.hmm.M == 5
