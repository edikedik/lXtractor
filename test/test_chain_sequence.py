from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from lXtractor.core.alignment import Alignment
from lXtractor.chain import ChainSequence
from lXtractor.core.config import DefaultConfig
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.seq import read_fasta, biotite_align, mafft_align


@pytest.fixture
def simple_chain_with_child(simple_chain_seq) -> ChainSequence:
    s = simple_chain_seq
    s.add_seq("S", [1 for _ in range(len(s))])
    s.spawn_child(1, 3, "X1"), s.spawn_child(1, 1, "X2")
    return s


def test_init(simple_chain_seq):
    # with pytest.raises(MissingData):
    #     ChainSequence(1, 2)
    s = simple_chain_seq
    assert DefaultConfig["mapnames"]["seq1"] in s
    assert len(s) == 5


def test_spawn(simple_chain_seq):
    s = simple_chain_seq
    s13 = s.spawn_child(1, 3)
    s12 = s13.spawn_child(1, 2)
    assert s12.parent == s13
    assert s13.parent == s
    assert s12.name == "S"
    assert s13.name == "S"


def test_degenerate():
    s = ChainSequence.from_string("")
    assert len(s) == 0
    assert s.start == s.end == 0
    assert len(list(iter(s))) == 0
    assert s == ChainSequence.make_empty()
    c = ChainSequence.from_string("c")
    assert c.is_singleton
    o = c & s
    assert o.is_empty
    o = s & c
    assert o.is_empty
    with pytest.raises(ValueError):
        _ = s >> 1
    o = c >> 1
    assert o.is_singleton and o.start == o.end == 2


def test_map_accession(simple_chain_seq):
    s = simple_chain_seq
    mapping = s.get_map("i")
    assert mapping[1].seq1 == "A"
    assert mapping[1].i == 1


def test_convert(simple_chain_seq):
    s = simple_chain_seq
    df = s.as_df()
    assert DefaultConfig["mapnames"]["seq1"] in df.columns
    # assert fields.seq3 in df.columns
    # assert fields.enum in df.columns
    assert len(df) == len(s)


def test_closest_and_boundaries():
    s = ChainSequence(
        1,
        5,
        "S",
        seqs={
            DefaultConfig["mapnames"]["seq1"]: "ABCDE",
            "N": [1, 3, 5, 10, 20],
            "K": [None, 10, None, 20, None],
        },
    )
    assert s.get_closest("N", 2).N == 3
    assert s.get_closest("N", 0).N == 1
    assert s.get_closest("N", 12, reverse=True).N == 10
    assert s.get_closest("N", 21) is None
    assert s.get_closest("N", 0, reverse=True) is None
    assert s.get_closest("K", 2).N == 3
    assert s.get_closest("K", 21, reverse=True).N == 10

    b1, b2 = s.map_boundaries(1, 3, map_name="N")
    assert (b1.i, b2.i) == (1, 2)
    b1, b2 = s.map_boundaries(-1, 0, map_name="N", closest=True)
    assert b1.i == b2.i == 1
    b1, b2 = s.map_boundaries(22, 23, map_name="N", closest=True)
    assert b1.i == b2.i == 5


def test_map(simple_fasta_path, chicken_src_seq, human_src_seq):
    s1, s2 = read_fasta(simple_fasta_path)
    aln = Alignment.make([s1, s2])

    s1 = ChainSequence.from_string(s1[1])
    s2 = ChainSequence.from_string(s2[1])
    mapping = s1.map_numbering(
        s2, save=True, name="map_smaller", align_method=mafft_align
    )
    assert "map_smaller" in s1
    assert mapping == [None, None, 1, 2, 3, 4, 5, None, None]
    mapping = s2.map_numbering(
        s1, save=True, name="map_larger", align_method=mafft_align
    )
    assert "map_larger" in s2
    assert mapping == [3, 4, 5, 6, 7]

    mapping = s2.map_numbering(aln, save=True, name="map_aln", align_method=mafft_align)
    assert mapping == [3, 4, 5, 6, 7]

    s1 = ChainSequence.from_string(chicken_src_seq[1])
    s2 = ChainSequence.from_string(human_src_seq[1])

    mapping = s2.map_numbering(s1, save=False, align_method=biotite_align)
    assert len(mapping) >= len(s2)


def test_map_transfer(simple_chain_seq):
    s1 = simple_chain_seq
    s2 = deepcopy(s1)
    s1.add_seq("R", "PUTIN")
    s1.add_seq("V", "MUDAK")
    s2.add_seq("O", "PUKIN")
    s1.relate(s2, "V", "O", "R")
    assert "V" in s2
    assert s2["V"] == ["M", "U", None, "A", "K"]


def test_io(simple_chain_seq):
    fnames = DefaultConfig["filenames"]
    s = simple_chain_seq
    child = s.spawn_child(1, 2)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        s.write(tmp_path, write_children=True)

        files = get_files(tmp_path)
        dirs = get_dirs(Path(tmp))

        assert fnames["sequence"] in files
        assert fnames["meta"] in files
        assert fnames["segments_dir"] in dirs

        s_r = ChainSequence.read(tmp_path, search_children=True)
        assert s_r.seq1 == s.seq1
        assert s_r.id == s.id

        assert len(s_r.children) == 1
        assert child.id == s_r.children.pop().id


def name2argh(x: ChainSequence) -> ChainSequence:
    return ChainSequence(x.start, x.end, "argh", seqs=x._seqs)


def test_apply_children(simple_chain_with_child):
    s = simple_chain_with_child
    s_new = s.apply_children(name2argh)
    assert len(s.children) == 2
    assert all(x.name in ["X1", "X2"] for x in s.children)
    assert all(x.name == "argh" for x in s_new.children)
    _ = s.apply_children(name2argh, inplace=True)
    assert all(x.name == "argh" for x in s.children)


def test_filter_children(simple_chain_with_child):
    s = simple_chain_with_child
    s_new = s.filter_children(lambda c: c.name != "X1")
    assert len(s.children) == 2
    assert len(s_new.children) == 1
    assert s_new.children.pop().name == "X2"
    s.filter_children(lambda c: c.name != "X1", inplace=True)
    assert len(s.children) == 1


def test_match(simple_chain_with_child):
    s = simple_chain_with_child
    s.add_seq("X", "AXCDX")
    with pytest.raises(KeyError):
        s.match("XXX", "S")
    match = s.match("seq1", "X", as_fraction=False, save=False)
    assert match == 1
    s.match("seq1", "X", as_fraction=True, save=True)
    match = s.meta["Match_seq1_X"]
    assert match == 0.2


def to_str(s) -> list[str]:
    return [str(x) for x in s]


def is_iterable_of(xs, _type):
    return all(isinstance(x, _type) for x in xs)


def test_apply_to_map(simple_chain_with_child):
    s = simple_chain_with_child

    s_new = s.apply_to_map("S", to_str)
    assert is_iterable_of(s_new["S"], str)
    assert len(s_new.children) == 0
    for c in s.children:
        assert is_iterable_of(c["S"], int)

    # Children weren't changed
    s_new = s.apply_to_map("S", to_str, preserve_children=True)
    assert len(s_new.children) == 2
    for c in s_new.children:
        assert is_iterable_of(c["S"], int)

    # Children were both transferred and transformed
    s_new = s.apply_to_map("S", to_str, apply_to_children=True)
    assert len(s_new.children) == 2
    for c in s_new.children:
        assert is_iterable_of(c["S"], str)

    # Same but with in-place op
    s_new = s.apply_to_map("S", to_str, apply_to_children=True, inplace=True)
    assert id(s_new) == id(s)
    assert len(s_new.children) == len(s.children) == 2
    for c in s.children:
        assert is_iterable_of(c["S"], str)


def test_as_chain(simple_chain_seq, simple_chain_structure):
    seq = simple_chain_seq
    c1 = seq.spawn_child(1, 2, "C")
    s = simple_chain_structure
    c = seq.as_chain()
    assert c.seq == seq
    assert len(c.children) == 1
    assert c.children[0].seq == c1
    c = seq.as_chain(structures=[s])
    assert len(c.structures) == 1
    assert len(c.children[0].structures) == 0
    c = seq.as_chain(structures=[s], add_to_children=True)
    assert len(c.children[0].structures) == 1


@pytest.mark.parametrize("meta", [True, False])
def test_summary(simple_chain_seq, meta):
    seq = simple_chain_seq
    df = seq.summary(meta=meta)
    assert isinstance(df, pd.DataFrame)
    assert "ObjectID" in df.columns
    meta_keys = list(seq.meta)
    if meta:
        assert set(meta_keys).issubset(set(df.columns))
    else:
        assert len(set(meta_keys) & set(df.columns)) == 0


@pytest.mark.parametrize(
    "seq,other,kwargs,expected",
    [
        (
            ChainSequence.from_string("ABCD", r=list(range(10, 14))),
            ChainSequence.from_string("AABXDE", r=list(range(9, 15))),
            dict(
                template="seq1",
                target="seq1",
                link_name="r",
                link_points_to="r",
                transform="".join,
                empty=("X",),
            ),
            "AABCDE",
        ),
        (
            ChainSequence.from_string(
                "XXXX", r=list(range(10, 14)), k=["A", "N", 1, None]
            ),
            ChainSequence.from_string(
                "YYYYYY", r=list(range(10, 16)), k=[None, None, None, "T", "h", "i"]
            ),
            dict(
                template="k",
                target="k",
                link_name="r",
                link_points_to="r",
            ),
            ["A", "N", 1, "T", "h", "i"],
        ),
        (
            ChainSequence.from_string("ABCD", r=list(range(10, 14))),
            ChainSequence.from_string("AABXDE", r=list(range(9, 15))),
            dict(
                template="seq1",
                target="seq1",
                link_name="r",
                link_points_to="r",
                transform="".join,
                is_empty=lambda x: x == "X",
            ),
            "AABCDE",
        ),
    ],
)
def test_patch(seq, other, kwargs, expected):
    assert seq.patch(other, **kwargs) == expected


def test_patch_empty():
    seq = ChainSequence.from_string("A", r=[0])
    other = ChainSequence.from_string("", r=[])
    res = seq.patch(other, "seq1", "seq1", "r", "r")
    assert res == []

    res = other.patch(seq, "seq1", "seq1", "r", "r")
    assert res == ["A"]
