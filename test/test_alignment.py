import pytest
from more_itertools import consume

import lXtractor.chain as lxc
from lXtractor.core.alignment import Alignment
from lXtractor.core.exceptions import InitError, MissingData
from lXtractor.util.seq import read_fasta


def test_init(simple_fasta_path):
    seqs = read_fasta(simple_fasta_path)
    with pytest.raises(InitError):
        _ = Alignment(seqs)
    with pytest.raises(InitError):
        _ = Alignment.read(simple_fasta_path)
    with pytest.raises(InitError):
        _ = Alignment([])

    aln = Alignment.make(read_fasta(simple_fasta_path))
    assert len(aln) == 2
    assert aln.seqs[0][0] == "seq1"
    assert aln.seqs[-1][1] == "--TRACT--"
    assert aln.shape == (2, len("lxtractor"))

    aln = Alignment.read_make(simple_fasta_path)
    assert len(aln) == 2


def test_iter(simple_fasta_path):
    aln = Alignment.read_make(simple_fasta_path)
    assert next(iter(aln)) == ("seq1", "LXTRACTOR")
    cols = list(aln.itercols(join=True))
    assert cols[0] == "L-"
    assert cols[-1] == "R-"
    cols = aln.itercols(join=False)
    assert next(cols) == ["L", "-"]


def test_slicing(simple_fasta_path):
    aln = Alignment.read_make(simple_fasta_path)
    sliced = aln.slice(0, 2, None)
    assert sliced.shape == (2, 2)
    assert sliced.seqs == [("seq1", "LX"), ("seq2", "--")]


def test_indexing(simple_fasta_path):
    aln = Alignment.read_make(simple_fasta_path)

    # test getters and indexing
    assert aln[0][0] == "seq1"
    assert aln["seq1"] == "LXTRACTOR"
    assert aln[1:] == [("seq2", "--TRACT--")]
    with pytest.raises(IndexError):
        _ = aln[3]
    with pytest.raises(KeyError):
        _ = aln["_seq"]


def test_modifying(simple_fasta_path):
    num_cols = len("LXTRACTOR")
    smaller = ("seq3", "TRACTOR")
    larger = ("seq4", "XXXTRACTORSHIK")

    aln = Alignment.read_make(simple_fasta_path)
    assert aln.shape == (2, num_cols)

    aligned = aln.align(smaller)
    assert aligned.shape == (1, num_cols)

    aligned = aln.align(larger)
    assert aligned.shape == (1, num_cols)

    aligned = aln.add([smaller, larger])
    assert aligned.shape == (4, num_cols)

    aligned = aligned.remove(smaller[0])
    assert aligned.shape == (3, num_cols)
    assert "seq4" in aligned
    with pytest.raises(MissingData):
        _ = aligned.remove(larger)
    aligned = aligned.remove(aligned[-1])
    assert aligned.shape == (2, num_cols)
    assert aln == aligned

    aligned = aligned.remove([aligned[-1]])
    assert aligned.shape == (1, num_cols)

    aligned += aln
    assert aligned.shape == (3, num_cols)


def test_mapping(simple_fasta_path):
    aln = Alignment.read_make(simple_fasta_path).map(
        lambda x: (x[0] + "X", x[1].lower())
    )
    assert len(aln) == 2
    assert aln["seq1X"] == "lxtractor"


def test_filtering(simple_fasta_path):
    aln = Alignment.read_make(simple_fasta_path)
    filtered = aln.filter(lambda x: "-" not in x[1])
    assert len(filtered) == 1
    assert "seq1" in filtered

    filtered = aln.filter_gaps(1.0, dim=0)
    assert len(filtered) == 2

    filtered = aln.filter_gaps(0.1, dim=0)
    assert "seq1" in filtered
    assert "seq2" not in filtered

    filtered = aln.filter_gaps(0.49, dim=1)
    assert filtered.shape != aln.shape
    assert filtered.shape == (2, len("lxtractor") - 4)


@pytest.fixture()
def ann_aln():
    return Alignment(
        [
            ("s1", "ACCADQK"),
            ("s2", "ACC-D-K"),
            ("s3", "ALCAQ--"),
            ("s4", "ACCID-K"),
            ("S5", "GLCADQR"),
        ]
    )


@pytest.mark.parametrize(
    "a,expected",
    [
        (
            lxc.ChainSequence.from_string("AAKACCADQ", name="a"),
            (4, 9, [1, 2, 3, 4, 5, 6], "ACCADQ"),
        ),
        (
            lxc.ChainSequence.from_string("AAKACADK", name="a"),
            (3, 8, [1, 2, 3, 4, 5, 7], "KACADK"),
        ),
        (
            lxc.ChainSequence.from_string("CCA", name="a"),
            (1, 3, [2, 3, 4], "CCA"),
        ),
    ],
)
def test_annotate_simple_case(a, expected, ann_aln):
    consume(ann_aln.annotate([a], "M"))
    assert len(a.children) == 1
    c = a.children[0]
    assert (c.start, c.end, c["M"], c.seq1) == expected


@pytest.mark.parametrize(
    "accept_fn,expected",
    [(None, 1), (lambda x: True, 1), (lambda x: False, 0), (lambda x: len(x) > 3, 0)],
)
def test_annotate_accept_fn(ann_aln, accept_fn, expected):
    a = lxc.ChainSequence.from_string("CCA", name="a")
    consume(ann_aln.annotate([a], "M", accept_fn=accept_fn))
    assert len(a.children) == expected
