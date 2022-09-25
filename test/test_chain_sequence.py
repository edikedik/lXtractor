from __future__ import annotations

import pytest

from lXtractor import Alignment
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData
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
    rec = s.as_record()
    assert rec.id == s.id
    assert str(rec.seq) == s[fields.seq1]


def test_closest(seq):
    fields = ChainSequence.field_names()
    s = ChainSequence(1, 5, 'S', {fields.seq1: 'ABCDE', 'N': [1, 3, 5, 10, 20]})
    assert s.get_closest('N', 2).N == 3
    assert s.get_closest('N', 0).N == 1
    assert s.get_closest('N', 12, reverse=True).N == 10
    assert s.get_closest('N', 21) is None


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
