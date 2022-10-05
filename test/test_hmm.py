from lXtractor.core.chain import ChainStructure
from lXtractor.ext.hmm import PyHMMer


def test_domain_extraction(chicken_src_str, pkinase_hmm_path):
    seqs = [
        ChainStructure.from_structure(chain_str, '2oiq').seq
        for chain_str in chicken_src_str.split_chains()
    ]
    annotator = PyHMMer(pkinase_hmm_path, bit_cutoffs='trusted')
    extracted = list(annotator.annotate(seqs, 'PK', keep=False))
    assert len(extracted) == 2
    assert len(extracted[0]) == 232
    assert len(extracted[1]) == 230
    assert all(not s.children for s in seqs)
    s = extracted[0]
    assert 'PK_score' in s.meta
