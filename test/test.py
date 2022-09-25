import typing as t
from io import StringIO
from itertools import chain

import pytest
from Bio import SeqIO

from lXtractor.alignment import Alignment
from lXtractor.base import SeqRec, Seq, DomainDataNames
from lXtractor.domains import extract_pdb_domains
from lXtractor.lXt import convert_to_attributes, lXtractor
from lXtractor.pdb import PDB
from lXtractor.protein import Protein
from lXtractor.sifts import SIFTS, map_segment_numbering, Segment, OverlapError, LengthMismatch
from lXtractor.uniprot import fetch_uniprot, UniProt
from lXtractor.utils import try_fetching_until
from lXtractor.variables import parse_variables, SeqEl, Phi, Psi, Omega, Dist, PseudoDihedral, Dihedral


@pytest.fixture()
def sifts():
    return SIFTS()


def test_map_segment_numbering():
    # failed cases
    s1, s2 = [Segment(1, 5), Segment(3, 6)], [Segment(2, 6), Segment(8, 11)]
    with pytest.raises(OverlapError):
        map_segment_numbering(s1, s2)

    s1, s2 = [Segment(1, 5), Segment(7, 10)], [Segment(1, 5), Segment(7, 11)]
    with pytest.raises(LengthMismatch):
        map_segment_numbering(s1, s2)

    # must work
    # one segment, nothing is missing
    s1, s2 = [Segment(1, 3)], [Segment(1, 3)]
    assert dict(map_segment_numbering(s1, s2)) == {1: 1, 2: 2, 3: 3}
    # 4 is missing, numberings match
    s1, s2 = [Segment(1, 3), Segment(5, 6)], [Segment(1, 3), Segment(5, 6)]
    assert dict(map_segment_numbering(s1, s2)) == {1: 1, 2: 2, 3: 3, 4: None, 5: 5, 6: 6}
    # 4 is missing, numberings don't match
    s1, s2 = [Segment(1, 3), Segment(5, 6)], [Segment(2, 4), Segment(6, 7)]
    assert dict(map_segment_numbering(s1, s2)) == {1: 2, 2: 3, 3: 4, 4: None, 5: 6, 6: 7}
    # same, swapping segment collections swaps mapping
    s1, s2 = [Segment(2, 4), Segment(6, 7)], [Segment(1, 3), Segment(5, 6)]
    assert dict(map_segment_numbering(s1, s2)) == {2: 1, 3: 2, 4: 3, 5: None, 6: 5, 7: 6}


def test_alignment():
    seqs = [
        'AAAAWWWAAAA-',
        '-AAAW-WAAA--',
        '-AAAWWWAAA--',
        '-AA-WWW-AA--',
        'AAAAWWWAAA--',
        '----WWW-----'
    ]

    recs = [SeqRec(Seq(s), id=str(i)) for i, s in enumerate(seqs)]
    aln = Alignment(None, None, recs)

    assert len(aln) == len(seqs)
    assert 'AAAAWWWAAAA' in aln
    assert aln.shape == (6, 12)

    aln.remove_gap_columns(overwrite=True)
    assert aln.shape == (6, 11)
    aln.partition_gap_sequences(
        max_fraction_of_gaps=0.5, overwrite=True)
    assert aln.shape == (5, 11)

    addition = [
        SeqRec(Seq('AAAAAWWWAAAAAA'), id='long'),
        SeqRec(Seq('WWW'), id='short'),
        SeqRec(Seq('AWWWA'), id='short')
    ]
    aln.add_sequences(addition, overwrite=True)
    # number of columns hasn't changes
    # sequences are added
    assert aln.shape == (8, 11)
    # There are two short sequences and one long sequence
    assert isinstance(aln['long'], SeqRec)
    assert isinstance(aln['short'], t.List)
    assert len(aln['short']) == 2
    # expect the long sequence to be cut to accomodate
    # the keeplength parameter
    assert str(addition[0].seq) not in aln
    assert str(aln['long'].seq) == 'AAAAWWWAAAA'

    aln.remove_sequences(['short', 'long'], overwrite=True)
    assert aln['short'] is None
    assert aln['long'] is None
    assert aln.shape == (5, 11)

    # 'AAAAAWWWAAAAAA'
    map_seq = addition[0]
    map_seq_numbering = [
        10, 11, 12, 13, 14,  # AAAAA
        16, 17, 18,  # WWW (Notice the discontinuity)
        20, 21, 22, 23, 24, 25  # AAAAAA (again, no 19 number)
    ]
    mapping = aln.map_seq_numbering(
        map_seq, map_seq_numbering)
    # Sequence will be cut as following -AAAAWWWAAAA--
    # So we expect the numbering start from 11 and end with 23
    assert len(mapping) == aln.shape[1]
    assert min(mapping) == 11
    assert max(mapping) == 23
    # If we don't have 15 and 19, then sequence
    # numbering is used correctly
    assert 15 not in mapping
    assert 19 not in mapping
    assert sorted(mapping) == map_seq_numbering[1:-2]
    # We expect for a sequence to cover all the alignment columns
    assert sorted(mapping.values()) == list(range(1, 12))

    # now we take the shorter sequence AWWWA
    # mafft aligns it as --A-WWW-A--
    map_seq = addition[-1]
    map_seq_numbering = [10, 11, 12, 13, 14]
    mapping = aln.map_seq_numbering(
        map_seq, map_seq_numbering)
    assert sorted(mapping) == map_seq_numbering
    # Thus we expect two skips in the mapping values
    # (i.e., columns of the MSA)
    assert sorted(mapping.values()) == [3, 5, 6, 7, 9]


def test_basic_sifts_mapping(sifts):
    results = list(sifts.map_numbering('1A0I:A'))
    assert len(results) == 1
    assert len(results[0]) == 348
    # holes are between 122 and 139 (8), 307 and 314 (8)
    assert sum(1 for x in results[0].values() if x is None) == 8 + 8
    assert (x == y for x, y in results[0].items() if y is not None)

    # more than one UniProt ID
    results = list(sifts.map_numbering('1AXK:A'))
    assert len(results) == 2
    assert {m.id_to for m in results} == {'P18429', 'P23904'}


def test_basic_fetching():
    some_ids = ['P18429', 'P23904']

    def fetcher(acc: t.Iterable[str]) -> t.List[SeqRec]:
        results = fetch_uniprot(acc, chunk_size=1)
        return list(SeqIO.parse(StringIO(results), 'fasta'))

    def get_remaining(
            fetched: t.List[SeqRec],
            remaining: t.Sequence[str]) -> t.List[str]:
        current_ids = {s.id.split('|')[1] for s in fetched}
        return list(set(remaining) - current_ids)

    res, remains = try_fetching_until(some_ids, fetcher, get_remaining, max_trials=2)
    records = list(chain.from_iterable(res))
    assert all(isinstance(rec, SeqRec) for rec in records)
    assert {rec.id.split('|')[1] for rec in records} == set(some_ids)


def test_uniprot():
    ids = ('P12931', 'P00523')
    proteins = [Protein(uniprot_id=id_) for id_ in ids]
    uniprot = UniProt(max_retries=2)
    proteins = uniprot.fetch_fasta(proteins)
    assert tuple(map(lambda p: p.uniprot_id, proteins)) == ids
    assert tuple(map(lambda p: p.uniprot_seq.id.split('|')[1], proteins)) == ids
    proteins = uniprot.fetch_domains(proteins)
    domains1, domains2 = map(lambda p: p.children, proteins)
    expected_domains = {'SH2', 'SH3', 'Protein kinase'}
    expected_names = {'SRC_CHICK', 'SRC_HUMAN'}
    assert domains1 is not None
    assert {s.name for s in domains1} == {s.name for s in domains2} == expected_domains
    proteins, df = uniprot.fetch_meta(proteins, ['entry_name'])
    assert set(df['entry_name']) == {'SRC_CHICK', 'SRC_HUMAN'}
    assert set(df['UniProt_ID']) == set(ids)
    assert proteins[0].metadata[0][0] == proteins[1].metadata[0][0] == 'entry_name'
    assert {proteins[0].metadata[0][1], proteins[1].metadata[0][1]} == expected_names

    proteins = [Protein(uniprot_id='XXXXXX')]
    proteins = uniprot.fetch_fasta(proteins)
    proteins = uniprot.fetch_domains(proteins)
    proteins, _ = uniprot.fetch_meta(proteins, ['entry_name'])
    assert proteins[0].uniprot_seq is None
    assert proteins[0].children is None
    assert proteins[0].metadata is None


def test_cut_record():
    record = SeqRec(Seq('A' * 10 + 'B' * 10 + 'C' * 10))
    segments = [Segment(1, 10), Segment(11, 20), Segment(21, 30)]
    # extracted = [cut_record(record, s) for s in segments]
    # TODO: write when finalized


def test_extract_structure_domains():
    pdb = PDB()
    sifts = SIFTS()
    domains = [
        Segment(84, 145, 'SH3', 'P12931'),
        Segment(151, 248, 'SH2', 'P12931'),
        Segment(270, 523, 'Protein kinase', 'P12931')
    ]
    prot = Protein(
        pdb='2SRC', chain='A', uniprot_id='P12931',
        domains=domains)
    prot = pdb.fetch([prot])[0]
    prot = extract_pdb_domains(prot, sifts)
    for dom_init, dom_ext in zip(domains, prot.children):
        assert dom_init.name == dom_ext.name
        assert dom_init.parent_name == dom_ext.parent_name
        assert dom_init.start == dom_ext.start
        assert dom_init.end == dom_ext.end
        assert dom_ext.data is not None
        pdb_start, pdb_end = dom_ext.data[DomainDataNames.pdb_segment_boundaries]
        assert 0 < pdb_end - pdb_start <= dom_ext.end - dom_ext.start


def test_get_attributes():
    inp = '2OIQ'
    res = convert_to_attributes(inp)
    assert res.pdb_id == inp
    assert res.chain_id is None
    assert res.uniprot_id is None
    inp = '2OIQ:A'
    res = convert_to_attributes(inp)
    assert res.pdb_id == '2OIQ'
    assert res.chain_id == 'A'
    inp = 'P12931,2SRC:A'
    res = convert_to_attributes(inp)
    assert res.pdb_id == '2SRC'
    assert res.chain_id == 'A'
    assert res.uniprot_id == 'P12931'
    inp = 'P12931,2SRC:A::SH2,SH3'
    res = convert_to_attributes(inp)
    assert res.pdb_id == '2SRC'
    assert res.chain_id == 'A'
    assert res.uniprot_id == 'P12931'
    assert res.children == ['SH2', 'SH3']
    # inp = 'P12931:A,2SRC::SH2,Protein kinase'
    # res = get_atrributes(inp)
    # assert res.pdb_id == '2SRC'
    # assert res.chain_id == 'A'


def test_init():
    inp = '2OIQ'
    sifts = SIFTS()
    lx = lXtractor([inp], sifts=sifts)
    assert len(lx.proteins) == 2
    chain_a, chain_b = lx.proteins
    assert chain_a.pdb == chain_b.pdb == '2OIQ'
    assert chain_a.uniprot_id == chain_b.uniprot_id == 'P00523'
    assert chain_a.chain == 'A'
    assert chain_b.chain == 'B'
    assert chain_a.dir_name == 'P00523_2OIQ:A'
    assert chain_b.dir_name == 'P00523_2OIQ:B'


def test_parse_variable():
    # one variable, one position
    vs, ss, ds = parse_variables('1')
    assert len(vs) == len(ss) == len(ds) == 1
    assert isinstance(vs[0], SeqEl)
    assert ss == ds == [None]

    vs, _, _ = parse_variables('1_Phi')
    assert isinstance(vs[0], Phi)
    vs, _, _ = parse_variables('1_Psi')
    assert isinstance(vs[0], Psi)
    vs, _, _ = parse_variables('1_Omega')
    assert isinstance(vs[0], Omega)

    # one variable, two positions
    vs, _, _ = parse_variables('1-2')
    assert isinstance(vs[0], Dist)
    assert vs[0].pos1 == 1
    assert vs[0].pos2 == 2
    assert vs[0].com is True
    vs, _, _ = parse_variables('1:CB-2:CB')
    assert vs[0].atom1 == 'CB'
    assert vs[0].atom2 == 'CB'

    # one variable, four positions
    vs, _, _ = parse_variables('1-2-3-4')
    assert isinstance(vs[0], PseudoDihedral)
    vs, _, _, = parse_variables('1:N-2:CA-3:C-4:N')
    assert isinstance(vs[0], Dihedral)

    # several variables
    vs, _, _, = parse_variables('1-2-3-4,1,2:CB-3:CB,2_Phi')
    assert len(vs) == 4
    assert isinstance(vs[0], PseudoDihedral)
    assert isinstance(vs[1], SeqEl)
    assert isinstance(vs[2], Dist)
    assert isinstance(vs[3], Phi)

    # several variables, several proteins
    vs, ss, _ = parse_variables('1,2--ABCD:A,BLABLA23')
    assert ss == ['ABCD:A', 'BLABLA23']

    # several variables, several proteins and domains
    vs, ss, ds = parse_variables('1,2--ABCD:A,BLABLA23::SH2,SH3')
    assert ss == ['ABCD:A', 'BLABLA23']
    assert ds == ['SH2', 'SH3']


if __name__ == '__main__':
    pytest.main()
