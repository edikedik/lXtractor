from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lXtractor.core.chain import ChainSequence
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData
from lXtractor.ext import filter_by_method
from lXtractor.core.chain.structure import filter_selection_extended, subset_to_matching
from lXtractor.protocols.superpose import superpose_pairwise
from lXtractor.util.seq import biotite_align


def test_extended_selection_filter(src_str):
    m = filter_selection_extended(src_str)
    assert m.sum() == len(src_str.array)

    dfg = [404, 405, 406]
    m = filter_selection_extended(src_str, pos=dfg, atom_names=['C', 'CA'])
    assert m.sum() == 6

    child = src_str.spawn_child(404, 406, map_from=SeqNames.enum)
    child._seq.add_seq('X', [1, 2, 3])
    m = filter_selection_extended(
        child, pos=[2], atom_names=['N', 'C', 'CA'], map_name='X', exclude_hydrogen=True
    )
    assert m.sum() == 3
    assert set(child.array[m].atom_name) == {'N', 'C', 'CA'}

    with pytest.raises(MissingData):
        filter_selection_extended(
            child,
            pos=[10],
            atom_names=['N', 'C', 'CA'],
            map_name='X',
            exclude_hydrogen=True,
        )

    m = filter_selection_extended(
        child,
        pos=[10],
        atom_names=['N', 'C', 'CA'],
        map_name='X',
        exclude_hydrogen=True,
        tolerate_missing=True,
    )

    assert m.sum() == 0


def test_subset_to_matching(abl_str, src_str):
    # Both are kinase proteins
    # Abl structure is a kinase domain
    # Src structure is a kinase domains + two additional ones
    # => we expect both structures to align at the kinase domain
    # => and subsetting should yield structures with the number
    # => of residues exactly as in the smaller (Abl) structure.

    abl_str = abl_str.rm_solvent()
    src_str = src_str.rm_solvent()
    abl_sub, src_sub = subset_to_matching(abl_str, src_str, save=False)
    assert len(abl_sub.seq) == len(src_sub.seq), 'Same number of residues'

    # 5 of Abl residue were matched by gaps in Src
    # easy to verify running mafft
    # num_abl_res = bst.get_residue_count(
    #     abl_str.array[bst.filter_canonical_amino_acids(abl_str.array)]
    # )
    # assert len(abl_sub._seq) == num_abl_res - 5

    # pre-aligning should yield the same result
    abl_str._seq.map_numbering(src_str._seq, name='SRC', align_method=biotite_align)
    abl_sub, src_sub = subset_to_matching(abl_str, src_str, map_name='SRC')
    assert len(abl_sub.seq) == len(src_sub.seq), 'Same number of residues'


def test_superpose_pairwise(abl_str, src_str, human_src_seq):
    # 1. Strict
    # =========

    # no arguments
    res = list(superpose_pairwise([abl_str, abl_str]))
    assert len(res) == 1
    res = res.pop()
    assert len(res) == 5
    assert res.ID_fix == res.ID_mob
    assert res.RmsdSuperpose < 0.001

    src_seq = ChainSequence.from_string(human_src_seq[1], name=human_src_seq[0])
    abl_str._seq.map_numbering(src_seq, name='REF')
    src_str._seq.map_numbering(src_seq, name='REF')

    # Align using backbone atoms, then calculate rmsd using all atoms
    res = list(
        superpose_pairwise([abl_str], [src_str],
                           selection_superpose=([407, 408, 409], ['CA', 'C', 'N']),
                           selection_dist=([407, 408], None), map_name='REF')
    )
    assert len(res) == 1
    res = res.pop()
    assert res.ID_fix == abl_str.id and res.ID_mob == src_str.id
    assert res.RmsdSuperpose <= 1

    # Trying to do the same in parallel
    res = list(
        superpose_pairwise([abl_str, src_str], [src_str, abl_str],
                           selection_superpose=([407, 408, 409], ['CA', 'C', 'N']),
                           selection_dist=([407, 408], None), map_name='REF',
                           num_proc=2)
    )
    assert len(res) == 4

    # 2. Flexible
    # ===========

    # Using XHRDX motif
    # In Abl, it's VHRDL
    # in Src, it's IHRDL
    pos = [386, 387, 388, 389, 390]

    # Strict should fail since the first residues have different numbers of atoms
    with pytest.raises(ValueError):
        next(
            superpose_pairwise([abl_str], [src_str], selection_superpose=(pos, None),
                               map_name='REF')
        )

    res = next(
        superpose_pairwise([abl_str], [src_str], selection_superpose=(pos, None),
                           strict=False, map_name='REF')
    )

    assert len(res) == 5
    # diff_seq, diff_atoms = res[-2:]
    #
    # assert diff_seq.SuperposeFixed == diff_seq.SuperposeMobile == 0
    # assert diff_seq.RmsdFixed == diff_seq.RmsdMobile == 0
    # assert diff_atoms.SuperposeFixed == diff_atoms.RmsdFixed
    # assert diff_atoms.SuperposeMobile == diff_atoms.RmsdMobile
    assert res[2] < 1

    # 3. Flexible with backbone atoms, same selection
    # ===============================================

    res = next(
        superpose_pairwise([abl_str], [src_str],
                           selection_superpose=(pos, ['N', 'CA', 'C']), strict=False,
                           map_name='REF')
    )

    diff_seq, diff_atoms = res[-2:]

    # assert diff_seq.SuperposeFixed == diff_seq.SuperposeMobile == 0
    # assert diff_seq.RmsdFixed == diff_seq.RmsdMobile == 0
    # assert diff_atoms.SuperposeFixed == diff_atoms.RmsdFixed
    # assert diff_atoms.SuperposeMobile == diff_atoms.RmsdMobile
    assert res[2] < 1


@pytest.mark.parametrize(
    'method,pdb_ids,accepted',
    [
        ('X-ray', ['2src', '1x48'], ['2src']),
        ('x-ray', ['2src', '1x48'], []),
        ('NMR', ['2src', '1x48'], ['1x48']),
        ('X-ray', ['afoijagajng'], []),
    ],
)
@pytest.mark.parametrize('use_dir', [True, False])
def test_by_method(method, pdb_ids, use_dir, accepted):
    if use_dir:
        with TemporaryDirectory() as tmp:
            assert filter_by_method(pdb_ids, method=method, dir_=Path(tmp)) == accepted
    else:
        assert filter_by_method(pdb_ids, method=method, dir_=None) == accepted
