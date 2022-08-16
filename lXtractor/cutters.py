import logging
import typing as t

from Bio.PDB.Structure import Structure

from lXtractor.base import (
    SeqRec, Segment, MissingData, AmbiguousMapping, NoOverlap, Seq, LengthMismatch)
from lXtractor.pdb import get_sequence, extract_sub_structure
from lXtractor.protein import Protein

LOGGER = logging.getLogger(__name__)
PDB_CUT_RESULT = t.NamedTuple(
    'CutResult', [
        ('Structure', Structure),
        ('Seq', str),
        ('SeqRaw', t.Tuple[str, ...]),
        ('Mapping', t.Dict[int, t.Optional[str]]),
        ('Start', int), ('End', int)])


def cut_record(
        rec: SeqRec, segment: Segment
) -> t.Tuple[int, int, SeqRec]:
    overlap = segment.overlap(Segment(1, len(rec)))

    if segment.end != overlap.end:
        LOGGER.warning(
            f"Segment's {segment} end of segment is larger "
            f"than the sequence it supposedly belongs to. "
            f"Will cut at sequence's end.")
    if segment.start != overlap.start:
        LOGGER.warning(
            f"Segment's {segment} start is lower than 0. "
            f"Will correct it to 1.")

    start, end = overlap.start, overlap.end
    add = f'{start}-{end}'
    domain_rec = SeqRec(
        rec.seq[start - 1: end],
        id=f'{rec.id}/{add}',
        name=f'{rec.name}/{add}',
        description=rec.description)

    return start, end, domain_rec


def cut_structure(
        structure: Structure, segment: Segment,
        mapping: t.Dict[int, t.Optional[int]],
) -> PDB_CUT_RESULT:
    """
    :param structure: biopython's ``Structure`` object with ``id``
        in the format "PDB_Code:Chain"
    :param segment:
    :param mapping:
    :return:
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


def extract_uniprot_domains(
        protein: Protein,
        keep_parent: bool = True
) -> Protein:
    if protein.domains is None:
        raise MissingData(f'No domains for protein {protein.id}')

    if protein.uniprot_seq is None:
        raise MissingData(f'No UniProt sequence for {protein.id}')

    for domain in protein.domains.values():
        start, end, rec = cut_record(protein.uniprot_seq, domain)
        domain.start, domain.end = start, end
        domain.uniprot_seq = rec

    if not keep_parent:
        protein.uniprot_seq = None

    return protein


def extract_pdb_domains(
        protein: Protein,
        keep_parent: bool = True
) -> Protein:
    if protein.structure is None:
        raise MissingData(f'Protein {protein.id} has no structure')
    if ':' not in protein.structure.id:
        raise MissingData(f'Protein {protein.id} is expected to have an ``id`` '
                          f'in the format "PDB_code:Chain_*"')
    if not protein.domains:
        raise MissingData(f'Protein {protein.id} has no domains to extract')
    if not protein.uni_pdb_map:
        raise MissingData(f'Protein {protein.id} does not have a map between '
                          f'UniProt and PDB sequence numbering (``uni_pdb_map``)')

    pdb_chain = protein.structure.id.split('_')[0]
    for domain in protein.domains.values():
        try:
            cut_result = cut_structure(
                protein.structure, domain, protein.uni_pdb_map)
        except (MissingData, NoOverlap, AmbiguousMapping,
                LengthMismatch, ValueError) as e:
            LOGGER.warning(
                f'failed to extract {domain.name} from {protein.id} '
                f'due to error {e}')
            continue

        _id = f'{pdb_chain}/{cut_result.Start}-{cut_result.End}'
        domain.structure = cut_result.Structure
        domain.pdb_segment_boundaries = cut_result.Start, cut_result.End
        domain.pdb_seq = SeqRec(Seq(cut_result.Seq), _id, _id, _id)
        domain.pdb_seq_raw = cut_result.SeqRaw
        domain.uni_pdb_map = cut_result.Mapping

    if not keep_parent:
        protein.structure = None

    return protein


if __name__ == '__main__':
    raise RuntimeError
