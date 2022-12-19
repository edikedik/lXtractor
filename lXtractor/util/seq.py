"""
Low-level utilities to work with sequences (as strings) or sequence files.
"""
from __future__ import annotations

import logging
import operator as op
import typing as t
from collections import abc
from io import StringIO, TextIOBase
from itertools import filterfalse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import biotite.sequence as bseq
import biotite.sequence.align as balign
from more_itertools import split_at, partition, split_before, tail, take

from lXtractor.core.base import SupportsWrite, AlignMethod
from lXtractor.core.exceptions import LengthMismatch, MissingData
from lXtractor.util.io import run_sp

LOGGER = logging.getLogger(__name__)

GAP_CHARS = ('-',)


def read_fasta(
    inp: Path | TextIOBase | abc.Iterable[str], strip_id: bool = True
) -> abc.Generator[tuple[str, str]]:
    """
    Simple lazy fasta reader.

    :param inp: Path or opened file or an iterable over lines.
    :param strip_id: Strip ID to the first consecutive (spaceless) string.
    :return: A generator of (header, seq) pairs.
    """

    def _yield_seqs(buffer):
        buffer = filterfalse(lambda x: not x or x == '\n', buffer)
        items = split_before(map(str.rstrip, buffer), lambda x: x[0] == '>')
        for it in items:
            header, seq = it[0][1:], it[1:]
            if strip_id:
                header = header.split()[0]
            yield header, ''.join(seq)

    if isinstance(inp, Path):
        with inp.open() as f:
            yield from _yield_seqs(f)
    elif isinstance(inp, (TextIOBase, abc.Iterable)):
        yield from _yield_seqs(inp)
    else:
        raise TypeError(f'Unsupported input type {type(inp)}')


def write_fasta(inp: abc.Iterable[tuple[str, str]], out: Path | SupportsWrite) -> None:
    """
    Simple fasta writer.

    :param inp: Iterable over (header, seq) pairs.
    :param out: Something that supports `.write` method.
    :return: Nothing.
    """
    data = '\n'.join(f'>{header}\n{seq}' for header, seq in inp)
    if isinstance(out, Path):
        out.write_text(data)
    elif isinstance(out, SupportsWrite):
        out.write(data)
    else:
        raise TypeError(f'Unsupported output type {type(out)}')


def mafft_add(
    msa: abc.Iterable[tuple[str, str]] | Path,
    seqs: abc.Iterable[tuple[str, str]],
    *,
    mafft: str = 'mafft',
    thread: int = 1,
    keeplength: bool = True,
) -> abc.Iterator[tuple[str, str]]:
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

    seqs = list(seqs)
    if not seqs:
        raise MissingData('No sequences to add')

    if isinstance(msa, Path):
        if not msa.exists():
            raise ValueError(f'Invalid path to sequences {msa}')
        msa_path = str(msa)
    elif isinstance(msa, abc.Iterable):
        msa_handle = NamedTemporaryFile('w')
        write_fasta(msa, msa_handle)
        msa_handle.seek(0)
        msa_path = msa_handle.name
    else:
        raise TypeError(f'Unsupported msa type {type(msa)}')

    add_handle = NamedTemporaryFile('w')
    write_fasta(seqs, add_handle)
    add_handle.seek(0)

    # Run the subprocess command
    keeplength = '--keeplength' if keeplength else ''
    cmd = (
        f'{mafft} --add {add_handle.name} {keeplength} --anysymbol '
        f'--inputorder --thread {thread} {msa_path}'
    )

    return tail(len(seqs), read_fasta(StringIO(run_sp(cmd).stdout)))


def mafft_align(
    seqs: abc.Iterable[tuple[str, str]] | Path,
    *,
    mafft: str = 'mafft-linsi',
    thread: int = 1,
) -> t.Iterator[tuple[str, str]]:
    """
    Align an arbitrary number of sequences using mafft.

    :param seqs: An iterable over (header, seq) pairs
        or path to file with sequences to align.
    :param thread: How many threads to dedicate for `mafft`.
    :param mafft: `mafft` executable (path or env variable).
    :return: An Iterator over aligned (header, seq) pairs.
    """
    if isinstance(seqs, Path):
        cmd = f'{mafft} --anysymbol --thread {thread} --inputorder {seqs}'
        return read_fasta(StringIO(run_sp(cmd).stdout))
    else:
        with NamedTemporaryFile('w') as handle:
            write_fasta(seqs, handle)
            handle.seek(0)
            filename = handle.name
            cmd = f'{mafft} --anysymbol --thread {thread} --inputorder {filename}'
            return read_fasta(StringIO(run_sp(cmd).stdout))


def biotite_align(
    seqs: abc.Iterable[tuple[str, str]]
) -> tuple[tuple[str, str], tuple[str, str]]:
    """
    Align two sequences using biotite `align_optimal` function.

    :param seqs: An iterable with exactly two sequences.
    :return: A pair of aligned sequences.
    """
    (h1, seq1), (h2, seq2) = take(2, seqs)

    if not isinstance(seq1, bseq.ProteinSequence):
        seq1 = bseq.ProteinSequence(seq1)
    if not isinstance(seq2, bseq.ProteinSequence):
        seq2 = bseq.ProteinSequence(seq2)

    alignments = balign.align_optimal(
        seq1, seq2, balign.SubstitutionMatrix.std_protein_matrix()
    )

    seq1a, seq2a = alignments[0].get_gapped_sequences()

    return (h1, seq1a), (h2, seq2a)


# def hmmer_align(
#         seqs: t.Iterable[SeqRec],
#         profile_path: Path,
#         hmmalign_exe: str = 'hmmalign'
# ) -> t.List[SeqRec]:
#     """
#     Align sequences using hmmalign from hmmer.
#     The latter must be installed and available.
#
#     :param seqs: Sequences to align.
#     :param profile_path: A path to a profile.
#     :param hmmalign_exe: Name of the executable.
#     :return: A list of aligned sequences.
#     """
#     handle = NamedTemporaryFile('w')
#     SeqIO.write(seqs, handle, 'fasta')
#     handle.seek(0)
#
#     cmd = f'{hmmalign_exe} {profile_path} {handle.name}'
#     return list(SeqIO.parse(StringIO(run_sp(cmd).stdout), 'stockholm'))


def remove_gap_columns(
    seqs: abc.Iterable[str], max_gaps: float = 1.0
) -> tuple[abc.Iterator[str], np.ndarray]:
    """
    Remove gap columns from a collection of sequences.

    :param seqs: A collection of equal length sequences.
    :param max_gaps: Max fraction of gaps allowed per column.
    :return: Initial seqs with gap columns removed and removed columns' indices.
    """
    # Convert sequences into a matrix
    arrays = np.vstack(list(map(list, seqs)))

    # Calculate fraction of gaps for each column
    fraction_of_gaps = (np.isin(arrays, GAP_CHARS)).sum(axis=0) / arrays.shape[0]

    # Create a boolean mask of such columns
    mask = fraction_of_gaps < max_gaps

    # Use the boolean mask to filter columns in each of the arrays
    new_seqs = (''.join(a) for a in arrays.T[mask].T)
    return new_seqs, np.where(~mask)[0]


def partition_gap_sequences(
    seqs: abc.Iterable[tuple[str, str]], max_fraction_of_gaps: float = 1.0
) -> tuple[abc.Iterator[str], abc.Iterator[str]]:
    """
    Removes sequences having fraction of gaps above the given threshold.

    :param seqs: a collection of arbitrary sequences.
    :param max_fraction_of_gaps: a threshold specifying an upper bound
        on allowed fraction of gap characters within a sequence.
    :return: a filtered list of sequences.
    """

    def fraction_of_gaps(seq):
        return sum(1 for c in seq if c == '-') / len(seq)

    above_threshold, below_threshold = map(
        lambda seqs: map(op.itemgetter(0), seqs),
        partition(lambda s: fraction_of_gaps(s[1]) < max_fraction_of_gaps, seqs),
    )

    return below_threshold, above_threshold


def parse_cdhit(clstr_file: Path) -> t.List[t.List[str]]:
    """
    Parse cd-hit cluster file into a (list of lists of) clusters with seq ids.
    """
    with clstr_file.open() as f:
        return list(
            map(
                lambda cc: list(map(lambda c: c.split('>')[1].split('...')[0], cc)),
                filter(bool, split_at(f, lambda x: x.startswith('>'))),
            )
        )


# def cluster_cdhit(
#         seqs: t.Iterable[SeqRec], ts: float,
#         cdhit_exec: t.Union[str, Path] = 'cd-hit'
# ) -> t.List[t.List[SeqRec]]:
#     """
#     Run cd-hit with params `-A 0.9 -g 1 -T 0 -d 0`.
#     :param seqs: Collection of seq records.
#     :param ts: Threshold value (`c` parameter).
#     :param cdhit_exec: Path or name of the executable.
#     :return: clustered seq record objects.
#     """
#
#     def get_word_length():
#         """
#         -n 5 for thresholds 0.7 ~ 1.0
#         -n 4 for thresholds 0.6 ~ 0.7
#         -n 3 for thresholds 0.5 ~ 0.6
#         -n 2 for thresholds 0.4 ~ 0.5
#         """
#         if ts > 0.7:
#             return 5
#         if ts > 0.6:
#             return 4
#         if ts > 0.5:
#             return 3
#         return 2
#
#     def ungap_seq(seq: SeqRec):
#         return SeqRec(
#             seq.seq.ungap(), id=seq.id, name=seq.name,
#             description=seq.description)
#
#     seqs_map = {s.id: s for s in seqs}
#     seqs = list(map(ungap_seq, seqs_map.values()))
#     msa_handle = NamedTemporaryFile('w')
#     num_aln = SeqIO.write(seqs, msa_handle, 'fasta')
#     LOGGER.debug(f'Wrote {num_aln} sequences into {msa_handle.name}')
#     msa_handle.seek(0)
#     out_handle = NamedTemporaryFile('w')
#     cmd = f'{cdhit_exec} -i {msa_handle.name} -o {out_handle.name} ' \
#           f'-c {round(ts, 2)} -g 1 -T 0 -M 0 -d 0 -n {get_word_length()}'
#     run_sp(cmd)
#     LOGGER.debug(f'successfully executed {cmd}')
#     clusters = parse_cdhit(Path(f'{out_handle.name}.clstr'))
#     return [[seqs_map[x] for x in c] for c in clusters]


# def seq_identity(
#         seq1: SeqRec, seq2: SeqRec, align: bool = True,
#         align_method: _Align_method = mafft_align
# ) -> float:
#     """
#     Calculate sequence identity between a pair of sequences.
#
#     :param seq1: Protein seq.
#     :param seq2: Protein seq.
#     :param align: Align before calculating.
#         If ``False``, sequences are assumed to be aligned.
#     :param align_method: Align method to use.
#         Must be a callable accepting and returning a list of sequences.
#     :return: A number of matching characters divided by a smaller sequence's length.
#     """
#     if align:
#         seq1, seq2 = align_method([seq1, seq2])
#     if len(seq1) != len(seq2):
#         raise ValueError('Seq lengths must match')
#     min_length = min(map(
#         lambda s: len(str(s.seq).replace('-', '')),
#         [seq1, seq2]))
#     matches = sum(1 for c1, c2 in zip(seq1, seq2)
#                   if c1 == c2 and c1 != '-' and c2 != '-')
#     return matches / min_length
#
#
# def seq_coverage(
#         seq: SeqRec, cover: SeqRec, align: bool = True,
#         align_method: _Align_method = mafft_align
# ) -> float:
#     """
#     Calculate which fraction of ``seq`` is covered by ``cover``.
#     The latter is assumed to be a subsequence of the former
#     (otherwise, 100% coverage is guaranteed).
#
#     :param seq: A protein sequence.
#     :param cover: A protein sequence to check against ``seq``.
#     :param align: Align before calculating.
#         If ``False``, sequences are assumed to be aligned.
#     :param align_method: Align method to use.
#         Must be a callable accepting and returning a list of sequences.
#     :return: A number of non-gap characters divided by the ``seq``'s length.
#     """
#     if align:
#         seq, cover = align_method([seq, cover])
#     if len(seq) != len(cover):
#         raise ValueError('Seq lengths must match')
#     seq_len = len(str(seq.seq).replace('-', ''))
#     num_cov = sum(1 for c1, c2 in zip(seq, cover)
#                   if c1 != '-' and c2 != '-')
#     return num_cov / seq_len


def map_pairs_numbering(
    s1: str,
    s1_numbering: t.Iterable[int],
    s2: str,
    s2_numbering: t.Iterable[int],
    align: bool = True,
    align_method: AlignMethod = mafft_align,
    empty: t.Any = None,
    **kwargs,
) -> t.Iterator[t.Tuple[t.Optional[int], t.Optional[int]]]:
    """
    Map numbering between a pair of sequences.


    :param s1: The first protein sequence.
    :param s1_numbering: The first sequence's numbering.
    :param s2: The second protein sequence.
    :param s2_numbering: The second sequence's numbering.
    :param align: Align before calculating.
        If ``False``, sequences are assumed to be aligned.
    :param align_method: Align method to use.
        Must be a callable accepting and returning a list of sequences.
    :param empty: Empty numeration element in place of a gap.
    :param kwargs: Passed to `align_method`.
    :return: Iterator over character pairs (`a`, `b`),
        where `a` and `b` are the original sequences' numberings.
        One of `a` or `b` in a pair can be `empty` to represent a gap.
    """

    s1_numbering, s2_numbering = map(list, [s1_numbering, s2_numbering])

    if align:
        s1_aln, s2_aln = map(
            op.itemgetter(1), align_method([('s1', s1), ('s2', s2)], **kwargs)
        )
    else:
        s1_aln, s2_aln = s1, s2

    s1_raw, s2_raw = map(lambda s: s.replace('-', ''), [s1_aln, s2_aln])

    if len(s1_raw) > len(s1_numbering):
        raise LengthMismatch('s1 is larger than a corresponding numbering')
    if len(s2_raw) > len(s2_numbering):
        raise LengthMismatch('s2 is larger than a corresponding numbering')
    if len(s1_aln) != len(s2_aln):
        raise LengthMismatch(
            f'Lengths of aligned seqs must match; '
            f'(len(s1)={len(s1)} != len(s2)={len(s2)})'
        )

    s1_pool, s2_pool = map(iter, [s1_numbering, s2_numbering])

    for c1, c2 in zip(s1_aln, s2_aln):
        try:
            n1 = next(s1_pool) if c1 != '-' else empty
        except StopIteration as e:
            raise RuntimeError('Numbering pool for s1 is exhausted') from e
        try:
            n2 = next(s2_pool) if c2 != '-' else empty
        except StopIteration as e:
            raise RuntimeError('Numbering pool for s2 is exhausted') from e
        yield n1, n2


# def subset_by_idx(seq: SeqRec, idx: t.Sequence[int], start=1):
#     sub = ''.join(c for i, c in enumerate(seq, start=start) if i in idx)
#     start, end = idx[0], idx[-1]
#     new_id = f'{seq.id}/{start}-{end}'
#     return SeqRec(Seq(sub), new_id, new_id, new_id)


# def cut(
#         rec: SeqRec, segment: Segment
# ) -> t.Tuple[int, int, SeqRec]:
#     """
#     Cut sequence in ``rec`` using ``segment``'s boundaries.
#
#     :param rec: Sequence record.
#     :param segment: Arbitrary segment. Makes sense for
#         :attr:`lXtractor.base.Segment.start` and :attr:`lXtractor.base.Segment.end`
#         to  define some subsequence's boundaries.
#     :return: A sequence record cut according to ``segment``'s boundaries.
#         A suffix "/{start}-{end]" is appended to ``rec``'s id, name and  description.
#     """
#
#     overlap = segment.overlap_with(Segment(1, len(rec)))
#
#     if segment.end != overlap.end:
#         LOGGER.warning(
#             f"Segment's {segment} end of segment is larger "
#             f"than the sequence it supposedly belongs to. "
#             f"Will cut at sequence's end.")
#     if segment.start != overlap.start:
#         LOGGER.warning(
#             f"Segment's {segment} start is lower than 0. "
#             f"Will correct it to 1.")
#
#     start, end = overlap.start, overlap.end
#     add = f'{start}-{end}'
#     domain_rec = SeqRec(
#         rec.seq[start - 1: end],
#         id=f'{rec.id}/{add}',
#         name=f'{rec.name}/{add}',
#         description=rec.description)
#
#     return start, end, domain_rec


if __name__ == '__main__':
    raise RuntimeError
