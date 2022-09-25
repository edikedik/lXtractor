import shutil
from tempfile import NamedTemporaryFile

import pytest

from lXtractor.util.seq import read_fasta, write_fasta, mafft_align, mafft_add


def test_read_fasta(simple_fasta_path):
    # using path
    res = read_fasta(simple_fasta_path)
    assert next(res) == ('seq1', 'LXTRACTOR')
    assert next(res) == ('seq2', 'TRACT')
    with pytest.raises(StopIteration):
        next(res)

    # using opened path
    with simple_fasta_path.open() as buffer:
        res = read_fasta(buffer)
        assert next(res)[0] == 'seq1'

    # using lines
    text = simple_fasta_path.read_text()
    res = read_fasta(text.split('\n'))
    assert next(res)[0] == 'seq1'

    # using tempfile
    handle = NamedTemporaryFile('w+')
    handle.write(text)
    handle.seek(0)
    res = read_fasta(handle)
    assert next(res)[0] == 'seq1'


def test_write_fasta(simple_fasta_path):
    seqs = read_fasta(simple_fasta_path)
    handle = NamedTemporaryFile('w+')
    write_fasta(seqs, handle)
    handle.seek(0)
    seqs = read_fasta(handle)
    assert next(seqs)[0] == 'seq1'


@pytest.mark.skipif(not shutil.which('mafft'), reason='mafft is unavailable')
def test_mafft_align(simple_fasta_path):
    # using a path as it is
    aln = list(mafft_align(simple_fasta_path))
    assert len(aln) == 2
    s1, s2 = aln
    assert s1[0] == 'seq1'
    assert s2[0] == 'seq2'
    assert s1[1] == 'LXTRACTOR'
    assert s2[1] == '--TRACT--'

    aln2 = list(mafft_align(aln))
    assert aln == aln2


@pytest.mark.skipif(not shutil.which('mafft'), reason='mafft is unavailable')
def test_mafft_add(simple_fasta_path):
    aln = list(mafft_align(simple_fasta_path))
    add = list(mafft_add(aln, [('seq3', 'LXACTOR')]))
    assert len(add) == 1
    assert add[0][0] == 'seq3'
    assert add[0][1].replace('-', '') == 'LXACTOR'
