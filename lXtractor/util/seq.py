import logging
import typing as t
from io import StringIO
from itertools import tee
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord as SeqRec
from more_itertools import split_at, ilen, partition
from toolz import curry

from lXtractor.core.base import SeqRec, LengthMismatch, Segment, Seq, _Align_method
from lXtractor.util.io import run_sp

LOGGER = logging.getLogger(__name__)


@curry
def mafft_add(
        msa: t.Union[t.Iterable[SeqRec], Path],
        seqs: t.Iterable[SeqRec],
        thread: int = 1, keeplength: bool = True,
        mafft: str = 'mafft'
) -> t.Tuple[t.List[SeqRec], t.List[SeqRec]]:
    """
    Add a sequence using mafft.
    All sequences are assumed to be :class:`Bio.SeqRecord.SeqRecord` objects.
    This is a curried function: incomplete argument set yield
    partially evaluated function (e.g., ``mafft_add(thread=10)``).

    :param msa: an iterable over sequences with the same length.
    :param seqs: an iterable over sequences comprising the addition.
    :param thread: how many threads to dedicate for `mafft`.
    :param keeplength: force to preserve the MSA's length.
    :param mafft: `mafft` executable.
    :return: A tuple of two lists of `SeqRecord` objects:
        with (1) alignment sequences with addition,
        and (2) aligned addition, separately.
    """
    seqs, seqs_ = tee(seqs)
    num_to_add = ilen(seqs_)
    LOGGER.debug(f'Will add {num_to_add} sequences')

    if isinstance(msa, Path):
        if not msa.exists():
            raise ValueError(f'Invalid path to sequences {msa}')
        msa_path = str(msa)
        msa_handle = None
        LOGGER.debug(f'Found existing path to MSA seqs {msa_path}')
    else:
        msa_handle = NamedTemporaryFile('w')
        num_aln = SeqIO.write(msa, msa_handle, 'fasta')
        LOGGER.debug(f'Wrote {num_aln} sequences into {msa_handle.name}')
        msa_handle.seek(0)
        msa_path = msa_handle.name

    add_handle = NamedTemporaryFile('w')
    num_add = SeqIO.write(seqs, add_handle, 'fasta')
    LOGGER.debug(f'Wrote {num_add} sequences into {add_handle.name}')
    add_handle.seek(0)

    # Run the subprocess command
    keeplength = '--keeplength' if keeplength else ''
    cmd = f'{mafft} --add {add_handle.name} {keeplength} --anysymbol ' \
          f'--inputorder --thread {thread} {msa_path}'
    LOGGER.debug(f'Will run the command {cmd}')
    res = run_sp(cmd)

    # Read the seqs into `SeqRec` objects from the stdout
    alignment = list(SeqIO.parse(StringIO(res.stdout), 'fasta'))
    LOGGER.debug(
        f'Addition resulted in alignment with {len(alignment)} sequences')
    if not alignment:
        raise ValueError('Resulted in an empty seqs')

    # Close handles and return the results
    add_handle.close()
    if msa_handle:
        msa_handle.close()

    return alignment, alignment[-num_to_add:]


@curry
def mafft_align(
        seqs: t.Iterable[SeqRec],
        thread: int = 1,
        mafft: str = 'mafft-linsi'
) -> t.List[SeqRec]:
    """
    Align an arbitrary number of sequences using mafft.
    All sequences are assumed to be :class:`Bio.SeqRecord.SeqRecord` objects.
    This is a minimalistic function, not a proper CLI for mafft.
    This is a curried function: incomplete argument set yield
    partially evaluated function (e.g. ``mafft_align(mafft='mafft')``).

    :param seqs: an iterable over arbitrary sequences.
    :param thread: how many threads to dedicate for `mafft`.
    :param mafft: `mafft` executable (path or env variable).
    :return: a list of aligned sequences, the order is preserved.
    """
    # Write all sequences into a temporary file
    handle = NamedTemporaryFile('w')
    num_seq = SeqIO.write(seqs, handle, 'fasta')
    handle.seek(0)
    LOGGER.debug(f'Wrote {num_seq} seq records into {handle.name}')

    # Run the subprocess command
    cmd = f'{mafft} --anysymbol --thread {thread} --inputorder {handle.name}'
    LOGGER.debug(f'Will run the command {cmd}')
    res = run_sp(cmd)

    # Wrap the results from stdout into a list of `SeqRec` objects
    alignment = list(SeqIO.parse(StringIO(res.stdout), 'fasta'))
    LOGGER.debug(f'Obtained {len(alignment)} sequences')

    # Check and return the seqs
    if not alignment:
        raise ValueError('Resulted in empty seqs')
    handle.close()
    return alignment


def hmmer_align(
        seqs: t.Iterable[SeqRec],
        profile_path: Path,
        hmmalign_exe: str = 'hmmalign'
) -> t.List[SeqRec]:
    """
    Align sequences using hmmalign from hmmer.
    The latter must be installed and available.

    :param seqs: Sequences to align.
    :param profile_path: A path to a profile.
    :param hmmalign_exe: Name of the executable.
    :return: A list of aligned sequences.
    """
    handle = NamedTemporaryFile('w')
    SeqIO.write(seqs, handle, 'fasta')
    handle.seek(0)

    cmd = f'{hmmalign_exe} {profile_path} {handle.name}'
    return list(SeqIO.parse(StringIO(run_sp(cmd).stdout), 'stockholm'))


def remove_gap_columns(
        seqs: t.Sequence[SeqRec],
        max_fraction_of_gaps: float = 1.0
) -> t.Tuple[t.List[SeqRec], np.ndarray]:
    """
    Remove gap columns from a collection of sequences.

    :param seqs: a collection of equal length sequences.
    :param max_fraction_of_gaps: columns with a fraction of gaps below this value
        will be excluded. Default "1.0" value is equivalent to removing columns
        comprised of gaps only.
    :return: A tuple of (1) a list of sequences,
        and (2) indices of the selected columns in the provided seqs
    """
    # Convert sequences into a matrix
    arrays = np.vstack([np.array(list(s.seq)) for s in seqs])

    # Calculate fraction of gaps for each column
    fraction_of_gaps = (arrays == '-').sum(axis=0) / arrays.shape[0]

    # Create a boolean mask of such columns
    mask = fraction_of_gaps < max_fraction_of_gaps
    LOGGER.debug(f'Detected {np.sum(~mask)} gap columns, {np.sum(mask)} to be retained')

    # Use the boolean mask to filter columns in each of the arrays
    # Then wrap the filtered sequences into `Seq`,
    # and, subsequently, `SeqRec` objects
    new_seqs = (Seq(''.join(a)) for a in arrays.T[mask].T)
    new_seqs = [
        SeqRec(new_seq, id=s.id, name=s.id, description=s.description)
        for new_seq, s in zip(new_seqs, seqs)]
    return new_seqs, np.where(mask)[0]


def remove_gap_sequences(
        seqs: t.Iterable[SeqRec],
        max_fraction_of_gaps: float = 0.9,
        gap: str = '-'
) -> t.Tuple[t.List[SeqRec], t.List[SeqRec]]:
    """
    Removes sequences having fraction of gaps above the given threshold.

    :param seqs: a collection of arbitrary sequences.
    :param max_fraction_of_gaps: a threshold specifying an upper bound
        on allowed fraction of gap characters within a sequence.
    :param gap: a gap character.
    :return: a filtered list of sequences.
    """

    def fraction_of_gaps(seq): return sum(1 for c in seq if c == gap) / len(seq)

    # Partition seqs based on predicate "fraction of gaps < threshold"
    above_threshold, below_threshold = map(
        list, partition(
            lambda s: fraction_of_gaps(s) < max_fraction_of_gaps,
            seqs))
    above_ids = [s.id for s in above_threshold]
    above_ids = '' if len(above_ids) > 100 else above_ids
    LOGGER.debug(
        f'Detected {len(above_threshold)} sequences with the number of gaps '
        f'>= {max_fraction_of_gaps}: {above_ids}')

    # Return sequences passing the threshold
    return below_threshold, above_threshold


def parse_cdhit(clstr_file: Path) -> t.List[t.List[str]]:
    """
    Parse cd-hit cluster file into a (list of lists of) clusters with seq ids.
    """
    with clstr_file.open() as f:
        return list(map(
            lambda cc: list(map(lambda c: c.split('>')[1].split('...')[0], cc)),
            filter(bool, split_at(f, lambda x: x.startswith('>')))
        ))


def cluster_cdhit(
        seqs: t.Iterable[SeqRec], ts: float,
        cdhit_exec: t.Union[str, Path] = 'cd-hit'
) -> t.List[t.List[SeqRec]]:
    """
    Run cd-hit with params `-A 0.9 -g 1 -T 0 -d 0`.
    :param seqs: Collection of seq records.
    :param ts: Threshold value (`c` parameter).
    :param cdhit_exec: Path or name of the executable.
    :return: clustered seq record objects.
    """

    def get_word_length():
        """
        -n 5 for thresholds 0.7 ~ 1.0
        -n 4 for thresholds 0.6 ~ 0.7
        -n 3 for thresholds 0.5 ~ 0.6
        -n 2 for thresholds 0.4 ~ 0.5
        """
        if ts > 0.7:
            return 5
        if ts > 0.6:
            return 4
        if ts > 0.5:
            return 3
        return 2

    def ungap_seq(seq: SeqRec):
        return SeqRec(
            seq.seq.ungap(), id=seq.id, name=seq.name,
            description=seq.description)

    seqs_map = {s.id: s for s in seqs}
    seqs = list(map(ungap_seq, seqs_map.values()))
    msa_handle = NamedTemporaryFile('w')
    num_aln = SeqIO.write(seqs, msa_handle, 'fasta')
    LOGGER.debug(f'Wrote {num_aln} sequences into {msa_handle.name}')
    msa_handle.seek(0)
    out_handle = NamedTemporaryFile('w')
    cmd = f'{cdhit_exec} -i {msa_handle.name} -o {out_handle.name} ' \
          f'-c {round(ts, 2)} -g 1 -T 0 -M 0 -d 0 -n {get_word_length()}'
    run_sp(cmd)
    LOGGER.debug(f'successfully executed {cmd}')
    clusters = parse_cdhit(Path(f'{out_handle.name}.clstr'))
    return [[seqs_map[x] for x in c] for c in clusters]


def seq_identity(
        seq1: SeqRec, seq2: SeqRec, align: bool = True,
        align_method: _Align_method = mafft_align
) -> float:
    """
    Calculate sequence identity between a pair of sequences.

    :param seq1: Protein seq.
    :param seq2: Protein seq.
    :param align: Align before calculating.
        If ``False``, sequences are assumed to be aligned.
    :param align_method: Align method to use.
        Must be a callable accepting and returning a list of sequences.
    :return: A number of matching characters divided by a smaller sequence's length.
    """
    if align:
        seq1, seq2 = align_method([seq1, seq2])
    if len(seq1) != len(seq2):
        raise ValueError('Seq lengths must match')
    min_length = min(map(
        lambda s: len(str(s.seq).replace('-', '')),
        [seq1, seq2]))
    matches = sum(1 for c1, c2 in zip(seq1, seq2)
                  if c1 == c2 and c1 != '-' and c2 != '-')
    return matches / min_length


def seq_coverage(
        seq: SeqRec, cover: SeqRec, align: bool = True,
        align_method: _Align_method = mafft_align
) -> float:
    """
    Calculate which fraction of ``seq`` is covered by ``cover``.
    The latter is assumed to be a subsequence of the former
    (otherwise, 100% coverage is guaranteed).

    :param seq: A protein sequence.
    :param cover: A protein sequence to check against ``seq``.
    :param align: Align before calculating.
        If ``False``, sequences are assumed to be aligned.
    :param align_method: Align method to use.
        Must be a callable accepting and returning a list of sequences.
    :return: A number of non-gap characters divided by the ``seq``'s length.
    """
    if align:
        seq, cover = align_method([seq, cover])
    if len(seq) != len(cover):
        raise ValueError('Seq lengths must match')
    seq_len = len(str(seq.seq).replace('-', ''))
    num_cov = sum(1 for c1, c2 in zip(seq, cover)
                  if c1 != '-' and c2 != '-')
    return num_cov / seq_len


def map_pairs_numbering(
        s1: SeqRec, s1_numbering: t.Iterable[int],
        s2: SeqRec, s2_numbering: t.Iterable[int],
        align: bool = True, align_method: _Align_method = mafft_align
) -> t.Iterator[t.Tuple[t.Optional[int], t.Optional[int]]]:
    """
    Map numbering between a pair of sequences.

    >>> from Bio.SeqRecord import SeqRecord as SeqRec
    >>> from Bio.Seq import Seq
    >>> s1, s2 = SeqRec(Seq('-AAA')), SeqRec(Seq('B-AA'))
    >>> mapping = map_pairs_numbering(s1, [1, 2, 3], s2, [4, 5, 6], align=False)
    >>> assert list(mapping) == [(None, 4), (1, None), (2, 5), (3, 6)]

    :param s1: The first protein sequence.
    :param s1_numbering: The first sequence's numbering.
    :param s2: The second protein sequence.
    :param s2_numbering: The second sequence's numbering.
    :param align: Align before calculating.
        If ``False``, sequences are assumed to be aligned.
    :param align_method: Align method to use.
        Must be a callable accepting and returning a list of sequences.
    :return: Iterator over character pairs (`a`, `b`),
        where `a` and `b` are the original sequences' numberings.
        One of `a` or `b` in a pair can be ``None`` to represent a gap.
    """

    s1_numbering, s2_numbering = map(list, [s1_numbering, s2_numbering])

    if align:
        s1_aln, s2_aln = align_method([s1, s2])
    else:
        s1_aln, s2_aln = s1, s2

    s1_raw, s2_raw = map(
        lambda s: str(s.seq).replace('-', ''),
        [s1_aln, s2_aln])

    if len(s1_raw) > len(s1_numbering):
        raise LengthMismatch(
            f'{s1.id} is larger than a corresponding numbering')
    if len(s2_raw) > len(s2_numbering):
        raise LengthMismatch(
            f'{s2.id} is larger than a corresponding numbering')

    if len(s1_aln) != len(s2_aln):
        raise LengthMismatch(
            f'Lengths for both sequences do not match '
            f'(len({s1.id})={len(s1)} != len({s2.id})={len(s2)})')

    s1_pool, s2_pool = map(iter, [s1_numbering, s2_numbering])

    for c1, c2 in zip(s1_aln, s2_aln):
        try:
            n1 = next(s1_pool) if c1 != '-' else None
        except StopIteration:
            raise ValueError(f'Numbering pool for {s1.id} is exhausted')
        try:
            n2 = next(s2_pool) if c2 != '-' else None
        except StopIteration:
            raise ValueError(f'Numbering pool for {s2.id} is exhausted')
        yield n1, n2


def subset_by_idx(seq: SeqRec, idx: t.Sequence[int], start=1):
    sub = ''.join(c for i, c in enumerate(seq, start=start) if i in idx)
    start, end = idx[0], idx[-1]
    new_id = f'{seq.id}/{start}-{end}'
    return SeqRec(Seq(sub), new_id, new_id, new_id)


def cut_record(
        rec: SeqRec, segment: Segment
) -> t.Tuple[int, int, SeqRec]:
    """
    Cut sequence in ``rec`` using ``segment``'s boundaries.

    :param rec: Sequence record.
    :param segment: Arbitrary segment. Makes sense for
        :attr:`lXtractor.base.Segment.start` and :attr:`lXtractor.base.Segment.end`
        to  define some subsequence's boundaries.
    :return: A sequence record cut according to ``segment``'s boundaries.
        A suffix "/{start}-{end]" is appended to ``rec``'s id, name and  description.
    """

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


if __name__ == '__main__':
    raise RuntimeError
