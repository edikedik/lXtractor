import biotite.structure as bst
import pytest

from lXtractor.core.chain import ChainStructure
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData
from lXtractor.core.structure import GenericStructure
from lXtractor.protocols import filter_selection_extended, subset_to_matching


def get_fst_chain(s: GenericStructure) -> ChainStructure:
    return ChainStructure.from_structure(next(s.split_chains()))


@pytest.fixture()
def abl_str(human_abl_str):
    return get_fst_chain(human_abl_str)


@pytest.fixture()
def src_str(chicken_src_str):
    return get_fst_chain(chicken_src_str)


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
