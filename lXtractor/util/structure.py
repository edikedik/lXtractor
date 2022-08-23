"""
Module to encompass helper functions extracting protein substructures and subsequences.
"""
import logging
import typing as t
from io import StringIO
from itertools import chain

from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from more_itertools import pairwise, rstrip, mark_ends
from toolz import pipe, identity
from toolz.curried import filter

from lXtractor.core.base import AminoAcidDict, Segment, AmbiguousMapping, NoOverlap

LOGGER = logging.getLogger(__name__)
PDB_CUT_RESULT = t.NamedTuple(
    'CutResult', [
        ('structure', Structure),
        ('seq1', str),
        ('seq3', t.Tuple[str, ...]),
        ('mapping', t.Dict[int, t.Optional[str]]),
        ('start', int), ('end', int)])


def cut_structure(
        structure: Structure, segment: Segment,
        mapping: t.Dict[int, t.Optional[int]],
) -> PDB_CUT_RESULT:
    """
    Extract a structure and subset the mapping according to
    a :class:`Segment` boundaries.

    :param structure: :class:`Structure` object with :attr:`Structure.id`
        in the format "PDB_Code:Chain".
    :param segment: segment with ``structure``'s UniProt subsequence's boundaries
        defined by :attr:`lXtractor.base.Segment.start`
        and :attr:`lXtractor.base.Segment.end`.
    :param mapping: mapping from ``segment``'s numbering to ``structure``'s numbering.
    :return: A namedtuple with substructure,
        substructure's sequence -- 1-letter code and 3-letter codes,
        subset of mapping corresponding to boundaries,
        cut mapping, start and end of segment boundaries.
    """

    pdb_chain = structure.id.split('_')[0]

    mapping_segment = Segment(min(mapping), max(mapping))
    if not segment.overlap(mapping_segment):
        raise NoOverlap(
            f'{segment} does not overlap with the '
            f'{mapping_segment}({pdb_chain}) of the mapped sequence')
    overlap = segment.overlap(mapping_segment)
    if segment.start != overlap.start:
        LOGGER.warning(
            f'Start {segment.start} of the {segment} '
            f'is lower than the mapping start {overlap.start}({pdb_chain}). '
            f'Will use the former instead of the latter.')
    if segment.end != overlap.end:
        LOGGER.warning(
            f'End {segment.end} of the {segment} '
            f'is larger than the mapping end {overlap.end}({pdb_chain}). '
            f'Will use the former instead of the latter.')

    # This prevents situations when the segment boundaries (UniProt sequence)
    # do not correspond to any PDB sequence elements (i.e., map to None)
    # Thus, we search for the closest mapped seq element from both ends
    try:
        start = next(
            v for k, v in mapping.items()
            if k >= overlap.start and v is not None)
    except StopIteration:
        raise AmbiguousMapping(
            f'Could not find the overlap start in the PDB sequence')
    try:
        end = next(
            v for k, v in reversed(list(mapping.items()))
            if k <= overlap.end and v is not None)
    except StopIteration:
        raise AmbiguousMapping(
            f'Could not find the overlap end in the PDB sequence')

    chain_id = pdb_chain.split(':')[1]
    sub_structure = extract_sub_structure(structure, chain_id, start, end)
    sub_seq_mapping = {
        k: v for k, v in mapping.items() if segment.start <= k <= segment.end}
    seq = get_sequence(sub_structure, convert=True)
    seq_raw = get_sequence(sub_structure, convert=False)
    LOGGER.debug(f'Extracted sub-structure {sub_structure.id}')

    return PDB_CUT_RESULT(sub_structure, seq, seq_raw, sub_seq_mapping, start, end)


def extract_sub_structure(
        structure: Structure,
        chain_id: t.Optional[str],
        res_start: t.Optional[int],
        res_end: t.Optional[int]) -> Structure:
    """
    Extract a specific region of a protein chain within structure.

    :param structure: biopython's ``Structure`` object.
    :param chain_id: a PDB chain identifier.
    :param res_start: a start of the desired segment.
    :param res_end: an end of the desired segment.
    :return: biopython's ``Structure`` object containing
        sub-structure resulting from the selection.
    """
    # the shortest way to subset structures is by making io
    # write a selection into a handle, and then use this handle
    # to setup a new `Structure` object
    selector = Selector(chain_id, res_start, res_end)
    handle = StringIO()
    io = PDBIO()
    io.set_structure(structure)
    io.save(handle, selector)
    parser = PDBParser(QUIET=True)
    handle.seek(0)
    new_id = f'{structure.id}:{chain_id}_{res_start}-{res_end}'
    sub_structure = parser.get_structure(new_id, handle)
    LOGGER.debug(f'selected new sub-structure {new_id} from {structure.id}')
    return sub_structure


def split_chains(structure: Structure) -> t.Iterator[t.Tuple[str, Structure]]:
    """
    Split ``Structure`` based on chain IDs.

    :param structure: biopython's ``Structure`` object.
    :return: an iterator over tuples ("chain_id", "sub-structure").
    """
    chain_ids = [c.id for c in structure.get_chains()]
    LOGGER.debug(f'Found {len(chain_ids)} chains in {structure.id}')
    for c in chain_ids:
        yield c, extract_sub_structure(structure, c, None, None)


def get_sequence(
        structure: Structure,
        convert: bool = True,
        filter_out: t.Collection[str] = ('HOH',),
        trim_het_tail: bool = True,
        numbering: bool = False,
) -> t.Union[str, t.Tuple[str, ...], t.Tuple[int, ...]]:
    # TODO: either return numbering regardless or separate this functionality
    """
    Extract structure's residues -- either their sequence or numbering.

    Optionally convert the sequence into one-letter codes.
    Any unknown residue names are marked as "X".
    (Known residue codes are specified within :class:`lXtractor.base.AminoAcidDict`).

    :param structure: biopython's ``Structure`` object.
    :param convert: convert 3-letter codes into 1-letter codes.
    :param filter_out: a collection of 3-letter codes to filter out.
    :param trim_het_tail: cut discontinuous HETATOMs ending a chain.
    :param numbering: Instead of sequence, extract its numbering.
    :return: One of the following:
        (1) joined one-letter codes of a protein sequence,
        (2) the same as (1) but without converting to one-letter codes,
        (3) a numbering of sequence elements from the PDB structure.
    """
    mapping = AminoAcidDict(any_unk='X')

    def trim_tail(residues: t.Iterable[Residue]):
        def pred(
                pair: t.Tuple[Residue, Residue]
        ) -> bool:
            r1, r2 = pair
            i1, i2 = r1.id[1], r2.id[2]
            return i1 + 1 != i2 and (len(r2.resname) == 1 or r2.resname not in mapping)

        pairs = pairwise(residues)
        pairs_stripped = rstrip(pairs, pred)
        pairs_marked = mark_ends(pairs_stripped)
        residues = chain.from_iterable(
            # if is the last element get both residues else get only the first one
            (x[2][0], x[2][1]) if x[1] else (x[2][0],)
            for x in pairs_marked)
        return residues

    return pipe(
        structure.get_residues(),
        filter(lambda r: r.get_resname() not in filter_out),
        trim_tail if trim_het_tail else identity,
        lambda residues: tuple(
            (r.get_id()[1] if numbering else r.get_resname())
            for r in residues),
        (lambda resnames: "".join(mapping[name] for name in resnames))
        if convert and not numbering else identity
    )


class Selector(Select):
    """
    Biopython's way of sub-setting structures.
    `None` attributes are omitted during the selection.
    """

    def __init__(
            self, chain_id: t.Optional[str],
            res_start: t.Optional[int],
            res_end: t.Optional[int]):
        self.chain_id = chain_id
        self.res_start = res_start
        self.res_end = res_end

    def accept_residue(self, residue):
        full_id = residue.full_id
        match_chain = self.chain_id is None or full_id[2] == self.chain_id
        match_lower = self.res_start is None or full_id[3][1] >= self.res_start
        match_upper = self.res_end is None or full_id[3][1] <= self.res_end
        return all([match_chain, match_lower, match_upper])

    def accept_chain(self, _chain):
        return self.chain_id is None or _chain.id == self.chain_id


if __name__ == '__main__':
    raise RuntimeError