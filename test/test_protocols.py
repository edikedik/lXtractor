import biotite.structure as bst
import pytest

from lXtractor.core.chain import ChainSequence
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData
from lXtractor.protocols import filter_selection_extended, subset_to_matching, superpose_pairwise


def test_extended_selection_filter(src_str):

    m = filter_selection_extended(src_str)
    assert m.sum() == len(src_str.array)

    dfg = [404, 405, 406]
    m = filter_selection_extended(src_str, pos=dfg, atom_names=['C', 'CA'])
    assert m.sum() == 6

    child = src_str.spawn_child(404, 406, map_from=SeqNames.enum)
    child.seq.add_seq('X', [1, 2, 3])
    m = filter_selection_extended(
        child, pos=[2], atom_names=['N', 'C', 'CA'], map_name='X',
        exclude_hydrogen=True)
    assert m.sum() == 3
    assert set(child.array[m].atom_name) == {'N', 'C', 'CA'}

    with pytest.raises(MissingData):
        filter_selection_extended(
            child, pos=[10], atom_names=['N', 'C', 'CA'], map_name='X',
            exclude_hydrogen=True)

    m = filter_selection_extended(
        child, pos=[10], atom_names=['N', 'C', 'CA'], map_name='X',
        exclude_hydrogen=True, tolerate_missing=True)

    assert m.sum() == 0


def test_subset_to_matching(abl_str, src_str):
    # Both are kinase proteins
    # Abl structure is a kinase domain
    # Src structure is a kinase domains + two additional ones
    # => we expect both structures to align at the kinase domain
    # => and subsetting should yield structures with the number
    # => of residues exactly as in the smaller (Abl) structure.

    abl_sub, src_sub = subset_to_matching(abl_str, src_str, save=False)
    assert len(abl_sub.seq) == len(src_sub.seq), 'Same number of residues'

    # 5 of Abl residue were matched by gaps in Src
    # easy to verify running mafft
    num_abl_res = bst.get_residue_count(
        abl_str.array[bst.filter_canonical_amino_acids(abl_str.array)]
    )
    assert len(abl_sub.seq) == num_abl_res - 5

    # pre-aligning should yield the same result
    abl_str.seq.map_numbering(src_str.seq, name='SRC')
    abl_sub, src_sub = subset_to_matching(abl_str, src_str, map_name='SRC')
    assert len(abl_sub.seq) == len(src_sub.seq), 'Same number of residues'


def test_superpose_pairwise(abl_str, src_str, human_src_seq):

    # 1. Strict
    # =========

    # no arguments
    res = list(superpose_pairwise([abl_str, abl_str]))
    assert len(res) == 1
    res = res.pop()
    assert len(res) == 4
    id1, id2, rmsd, matrix = res
    assert id1 == id2
    assert rmsd < 0.001

    src_seq = ChainSequence.from_string(human_src_seq[1], name=human_src_seq[0])
    abl_str.seq.map_numbering(src_seq, name='REF')
    src_str.seq.map_numbering(src_seq, name='REF')

    # Align using backbone atoms, then calculate rmsd using all atoms
    res = list(superpose_pairwise(
        [abl_str], [src_str],
        selection_superpose=([407, 408, 409], ['CA', 'C', 'N']),
        selection_rmsd=([407, 408], None),
        map_name='REF',
    ))
    assert len(res) == 1
    id1, id2, rmsd, matrix = res.pop()
    assert id1 == abl_str.id and id2 == src_str.id
    assert rmsd <= 1

    # Trying to do the same in parallel
    res = list(superpose_pairwise(
        [abl_str, src_str], [src_str, abl_str],
        selection_superpose=([407, 408, 409], ['CA', 'C', 'N']),
        selection_rmsd=([407, 408], None),
        map_name='REF', num_proc=2
    ))
    assert len(res) == 4

    # 2. Flexible
    # ===========

    # Using XHRDX motif
    # In Abl, it's VHRDL
    # in Src, it's IHRDL
    pos = [386, 387, 388, 389, 390]

    # Strict should fail since the first residues have different numbers of atoms
    with pytest.raises(ValueError):
        next(superpose_pairwise(
            [abl_str], [src_str],
            selection_superpose=(pos, None),
            map_name='REF'
        ))

    res = next(superpose_pairwise(
            [abl_str], [src_str],
            selection_superpose=(pos, None),
            map_name='REF', strict=False
        ))

    assert len(res) == 5
    diff = res[-1]

    assert diff.SeqSuperposeFixed == diff.SeqSuperposeMobile == 0
    assert diff.SeqRmsdFixed == diff.SeqRmsdMobile == 0
    assert diff.AtomsSuperposeFixed == diff.AtomsRmsdFixed
    assert diff.AtomsSuperposeMobile == diff.AtomsRmsdMobile
    assert res[2] < 1