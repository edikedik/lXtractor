from io import StringIO

from lXtractor.ext.uniprot import fetch_uniprot
from lXtractor.util.seq import read_fasta


def test_fetch_uniprot():
    ids = ['P00523', 'P12931']
    results = fetch_uniprot(ids)
    seqs = list(read_fasta(StringIO(results)))
    assert len(seqs) == 2
    assert {x[0].split('|')[1] for x in seqs} == {'P00523', 'P12931'}
