from __future__ import annotations

import pytest

from lXtractor import Alignment
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData, AmbiguousMapping
from lXtractor.core.chain import ChainSequence
from lXtractor.util.seq import read_fasta


@pytest.fixture
def seq() -> tuple[SeqNames, ChainSequence]:
    fields = ChainSequence.field_names()
    s = ChainSequence(1, 5, 'S', {fields.seq1: 'ABCDE'})
    return fields, s


def test_init(seq):
    with pytest.raises(MissingData):
        ChainSequence(1, 2)
    fields, s = seq
    assert fields.seq3 in s
    assert fields.enum in s
    assert len(s.seq1) == len(s.seq3) == len(s.numbering)
    assert fields.variables in s.meta


def test_map_accession(seq):
    fields, s = seq
    mapping = s.get_map(fields.enum)
    assert mapping[1].seq1 == 'A'
    assert mapping[1].i == 1
    assert s.get_item(fields.enum, 1).seq1 == mapping[1].seq1


def test_convertage(seq):
    fields, s = seq
    df = s.as_df()
    assert fields.seq1 in df.columns
    assert fields.seq3 in df.columns
    assert fields.enum in df.columns
    assert len(df) == len(s)


def test_closest(seq):
    fields = ChainSequence.field_names()
    s = ChainSequence(1, 5, 'S', seqs={
        fields.seq1: 'ABCDE', 'N': [1, 3, 5, 10, 20], 'K': [None, 10, None, 20, None]})
    assert s.get_closest('N', 2).N == 3
    assert s.get_closest('N', 0).N == 1
    assert s.get_closest('N', 12, reverse=True).N == 10
    assert s.get_closest('N', 21) is None
    assert s.get_closest('N', 0, reverse=True) is None
    assert s.get_closest('K', 2).N == 3
    assert s.get_closest('K', 21, reverse=True).N == 10


def test_map(simple_fasta_path):
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


def test_map_boundaries(seq):
    fields, s = seq
    s.add_seq('map_other', [10, 11, 12, 13, 14])
    with pytest.raises(KeyError):
        _, _ = s.map_boundaries(9, 10, 'map_other')
    start, end = s.map_boundaries(9, 10, 'map_other', closest=True)
    assert start.i == end.i == 1
    start, end = s.map_boundaries(-100, 100, 'map_other', closest=True)
    assert start.map_other == 10
    assert end.map_other == 14


def test_spawn(seq):
    fields, s = seq
    s.add_seq('map_other', [10, 11, 12, 13, 14])
    child = s.spawn_child(1, 2)
    assert child.id in s.children
    assert len(child) == 2
    assert child.seq1 == 'AB'
    child = s.spawn_child(9, 10, map_from='map_other', map_closest=True)
    assert len(child) == 1
    assert child.seq1 == 'A'


def test_iterchildren(seq):
    fields, s = seq
    child1 = s.spawn_child(1, 4)
    child2 = child1.spawn_child(1, 2)
    levels = list(s.iter_children())
    assert len(levels) == 2
    assert levels == [[child1], [child2]]
