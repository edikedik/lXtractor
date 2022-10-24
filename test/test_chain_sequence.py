from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lXtractor import Alignment
from lXtractor.core.chain import ChainSequence
from lXtractor.core.config import DumpNames
from lXtractor.core.exceptions import MissingData
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.seq import read_fasta


def test_init(simple_chain_seq):
    with pytest.raises(MissingData):
        ChainSequence(1, 2)
    fields, s = simple_chain_seq
    assert fields.seq3 in s
    assert fields.enum in s
    assert len(s.seq1) == len(s.seq3) == len(s.numbering)
    assert fields.variables in s.meta


def test_map_accession(simple_chain_seq):
    fields, s = simple_chain_seq
    mapping = s.get_map(fields.enum)
    assert mapping[1].seq1 == 'A'
    assert mapping[1].i == 1
    assert s.get_item(fields.enum, 1).seq1 == mapping[1].seq1


def test_convertage(simple_chain_seq):
    fields, s = simple_chain_seq
    df = s.as_df()
    assert fields.seq1 in df.columns
    assert fields.seq3 in df.columns
    assert fields.enum in df.columns
    assert len(df) == len(s)


def test_closest():
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
        sr_child = s_r.children[child.name]
        assert child.id == sr_child.id
