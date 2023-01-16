from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lXtractor.core.alignment import Alignment
from lXtractor.core.chain import ChainSequence
from lXtractor.core.config import DumpNames
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.seq import read_fasta, biotite_align


def test_init(simple_chain_seq):
    # with pytest.raises(MissingData):
    #     ChainSequence(1, 2)
    fields, s = simple_chain_seq
    assert fields.seq1 in s
    assert len(s) == 5


def test_degenerate():
    s = ChainSequence.from_string('')
    assert len(s) == 0
    assert s.start == s.end == 0
    assert len(list(iter(s))) == 0
    c = ChainSequence.from_string('c')
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
    fields, s = simple_chain_seq
    mapping = s.get_map('i')
    assert mapping[1].seq1 == 'A'
    assert mapping[1].i == 1


def test_convert(simple_chain_seq):
    fields, s = simple_chain_seq
    df = s.as_df()
    assert fields.seq1 in df.columns
    # assert fields.seq3 in df.columns
    # assert fields.enum in df.columns
    assert len(df) == len(s)


def test_closest_and_boundaries():
    fields = ChainSequence.field_names()
    s = ChainSequence(
        1,
        5,
        'S',
        seqs={
            fields.seq1: 'ABCDE',
            'N': [1, 3, 5, 10, 20],
            'K': [None, 10, None, 20, None],
        },
    )
    assert s.get_closest('N', 2).N == 3
    assert s.get_closest('N', 0).N == 1
    assert s.get_closest('N', 12, reverse=True).N == 10
    assert s.get_closest('N', 21) is None
    assert s.get_closest('N', 0, reverse=True) is None
    assert s.get_closest('K', 2).N == 3
    assert s.get_closest('K', 21, reverse=True).N == 10

    b1, b2 = s.map_boundaries(1, 3, map_name='N')
    assert (b1.i, b2.i) == (1, 2)
    b1, b2 = s.map_boundaries(-1, 0, map_name='N', closest=True)
    assert b1.i == b2.i == 1
    b1, b2 = s.map_boundaries(22, 23, map_name='N', closest=True)
    assert b1.i == b2.i == 5


def test_map(simple_fasta_path, chicken_src_seq, human_src_seq):
    s1, s2 = read_fasta(simple_fasta_path)
    aln = Alignment.make([s1, s2])

    s1 = ChainSequence.from_string(s1[1])
    s2 = ChainSequence.from_string(s2[1])
    mapping = s1.map_numbering(s2, save=True, name='map_smaller')
    assert 'map_smaller' in s1
    assert mapping == [None, None, 1, 2, 3, 4, 5, None, None]
    mapping = s2.map_numbering(s1, save=True, name='map_larger')
    assert 'map_larger' in s2
    assert mapping == [3, 4, 5, 6, 7]

    mapping = s2.map_numbering(aln, save=True, name='map_aln')
    assert mapping == [3, 4, 5, 6, 7]

    s1 = ChainSequence.from_string(chicken_src_seq[1])
    s2 = ChainSequence.from_string(human_src_seq[1])

    mapping = s2.map_numbering(s1, save=False, align_method=biotite_align)
    assert len(mapping) >= len(s2)


def test_map_transfer(simple_chain_seq):
    _, s1 = simple_chain_seq
    s2 = deepcopy(s1)
    s1.add_seq('R', 'PUTIN')
    s1.add_seq('V', 'MUDAK')
    s2.add_seq('O', 'PUKIN')
    s1.relate(s2, 'V', 'O', 'R')
    assert 'V' in s2
    assert s2['V'] == ['M', 'U', None, 'A', 'K']


def test_io(simple_chain_seq):
    _, s = simple_chain_seq
    child = s.spawn_child(1, 2)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        s.write(tmp_path, write_children=True)

        files = get_files(tmp_path)
        dirs = get_dirs(Path(tmp))

        assert DumpNames.sequence in files
        assert DumpNames.meta in files
        assert DumpNames.segments_dir in dirs

        s_r = ChainSequence.read(tmp_path, search_children=True)
        assert s_r.seq1 == s.seq1
        assert s_r.id == s.id

        assert len(s_r.children) == 1
        assert child.id == s_r.children.pop().id


# def test_apply_children(simple_chain_seq, simple_chain_structure):
#     pass


def test_match(simple_chain_seq, simple_chain_structure):
    _, s = simple_chain_seq
    s.add_seq('X', 'AXCDX')
    with pytest.raises(KeyError):
        s.match('XXX', 'S')
    match = s.match('seq1', 'X', as_fraction=False, save=False)
    assert match == 3
    s.match('seq1', 'X', as_fraction=True, save=True)
    match = s.meta['Match_seq1_X']
    assert match == 0.6
