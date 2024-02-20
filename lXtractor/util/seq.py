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
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile

import biotite.sequence as bseq
import biotite.sequence.align as balign
import numpy as np
from more_itertools import split_at, partition, split_before, tail, take

import lXtractor.util.io as lxio
from lXtractor.core.base import SupportsWrite, AlignMethod
from lXtractor.core.exceptions import LengthMismatch, MissingData

__all__ = (
    'read_fasta',
    'write_fasta',
    'mafft_add',
    'mafft_align',
    'biotite_align',
    'remove_gap_columns',
    'partition_gap_sequences',
    'map_pairs_numbering',
)

LOGGER = logging.getLogger(__name__)

GAP_CHARS = ('-',)


def read_fasta(
    inp: str | PathLike | TextIOBase | abc.Iterable[str], strip_id: bool = True
) -> abc.Iterator[tuple[str, str]]:
    """
    Simple lazy fasta reader.

    :param inp: Pathlike object compatible with ``open`` or opened file or an
        iterable over lines or raw text as str.
    :param strip_id: Strip ID to the first consecutive (spaceless) string.
    :return: An iterator of (header, seq) pairs.
    """

    def _yield_seqs(buffer):
        buffer = filterfalse(lambda x: not x or x == '\n', buffer)
        items = split_before(map(str.rstrip, buffer), lambda x: x[0] == '>')
        for it in items:
            header, seq = it[0][1:], it[1:]
            if strip_id:
                header = header.split()[0]
            yield header, ''.join(seq)

    if isinstance(inp, PathLike):
        with open(inp) as f:
            yield from _yield_seqs(f)
    elif isinstance(inp, (TextIOBase, abc.Iterable)) and not isinstance(inp, str):
        yield from _yield_seqs(inp)
    elif isinstance(inp, str):
        yield from _yield_seqs(StringIO(inp))
    else:
        raise TypeError(f'Unsupported input type {type(inp)}')


def write_fasta(inp: abc.Iterable[tuple[str, str]], out: Path | SupportsWrite) -> None:
    """
    Simple fasta writer.

    :param inp: Iterable over (header, _seq) pairs.
    :param out: Something that supports `.write` method.
    :return: Nothing.
    """
    data = '\n'.join(f'>{header}\n{seq}' for header, seq in inp)
    if isinstance(out, Path):
        out.write_text(data)
    elif isinstance(out, SupportsWrite):
        out.write(data)
    else:
        try:
            out.write(data)
        except Exception as e:
            raise TypeError(f'Unsupported output type {type(out)}') from e


def mafft_add(
    msa: abc.Iterable[tuple[str, str]] | Path,
    seqs: abc.Iterable[tuple[str, str]],
    *,
    mafft: str = 'mafft',
    thread: int = 1,
    keeplength: bool = True,
) -> abc.Iterator[tuple[str, str]]:
    """
    Add sequences to existing MSA using mafft.

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
    keep = '--keeplength' if keeplength else ''
    cmd = (
        f'{mafft} --add {add_handle.name} {keep} --anysymbol '
        f'--inputorder --thread {thread} {msa_path}'
    )

    return tail(len(seqs), read_fasta(StringIO(lxio.run_sp(cmd).stdout)))


def mafft_align(
    seqs: abc.Iterable[tuple[str, str]] | Path,
    *,
    mafft: str = 'mafft-linsi',
    thread: int = 1,
) -> t.Iterator[tuple[str, str]]:
    """
    Align an arbitrary number of sequences using mafft.

    :param seqs: An iterable over (header, _seq) pairs
        or path to file with sequences to align.
    :param thread: How many threads to dedicate for `mafft`.
    :param mafft: `mafft` executable (path or env variable).
    :return: An Iterator over aligned (header, _seq) pairs.
    """
    if isinstance(seqs, Path):
        cmd = f'{mafft} --anysymbol --thread {thread} --inputorder {seqs}'
        return read_fasta(StringIO(lxio.run_sp(cmd).stdout))
    else:
        with NamedTemporaryFile('w') as handle:
            write_fasta(seqs, handle)
            handle.seek(0)
            filename = handle.name
            cmd = f'{mafft} --anysymbol --thread {thread} --inputorder {filename}'
            return read_fasta(StringIO(lxio.run_sp(cmd).stdout))


def biotite_align(
    seqs: abc.Iterable[tuple[str, str]], **kwargs
) -> tuple[tuple[str, str], tuple[str, str]]:
    """
    Align two sequences using biotite `align_optimal` function.

    :param seqs: An iterable with exactly two sequences.
    :param kwargs: Additional arguments to `align_optimal`.
    :return: A pair of aligned sequences.
    """
    (h1, seq1), (h2, seq2) = take(2, seqs)

    if not isinstance(seq1, bseq.ProteinSequence):
        seq1 = bseq.ProteinSequence(seq1)
    if not isinstance(seq2, bseq.ProteinSequence):
        seq2 = bseq.ProteinSequence(seq2)

    alignments = balign.align_optimal(
        seq1, seq2, balign.SubstitutionMatrix.std_protein_matrix(), **kwargs
    )

    seq1a, seq2a = alignments[0].get_gapped_sequences()

    return (h1, seq1a), (h2, seq2a)


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

    arrays: np.ndarray = np.vstack([list(seq) for seq in seqs])

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
    Parse cd-hit cluster file into a (list of lists of) clusters with _seq ids.
    """
    with clstr_file.open() as f:
        return list(
            map(
                lambda cc: list(
                    map(lambda c: c.split('>')[1].split('...')[0], cc)  # type: ignore
                ),
                filter(bool, split_at(f, lambda x: x.startswith('>'))),
            )
        )


# def cluster_cdhit(
#         seqs: t.Iterable[SeqRec], ts: float,
#         cdhit_exec: t.Union[str, Path] = 'cd-hit'
# ) -> t.List[t.List[SeqRec]]:
#     """
#     Run cd-hit with params `-A 0.9 -g 1 -T 0 -d 0`.
#     :param seqs: Collection of _seq records.
#     :param ts: Threshold value (`c` parameter).
#     :param cdhit_exec: Path or name of the executable.
#     :return: clustered _seq record objects.
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
#     def ungap_seq(_seq: SeqRec):
#         return SeqRec(
#             _seq._seq.ungap(), id=_seq.id, name=_seq.name,
#             description=_seq.description)
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
#     lxio.run_sp(cmd)
#     LOGGER.debug(f'successfully executed {cmd}')
#     clusters = parse_cdhit(Path(f'{out_handle.name}.clstr'))
#     return [[seqs_map[x] for x in c] for c in clusters]


def map_pairs_numbering(
    s1: str,
    s1_numbering: abc.Iterable[int],
    s2: str,
    s2_numbering: abc.Iterable[int],
    align: bool = True,
    align_method: AlignMethod = mafft_align,
    empty: t.Any = None,
    **kwargs,
) -> abc.Generator[tuple[int | None, int | None], None, None]:
    """
    Map numbering between a pair of sequences.

    :param s1: The first sequence.
    :param s1_numbering: The first sequence's numbering.
    :param s2: The second sequence.
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

    def rm_dash(s: str) -> str:
        return s.replace('-', '')

    if align:
        s1_aln, s2_aln = map(
            op.itemgetter(1), align_method([('s1', s1), ('s2', s2)], **kwargs)
        )
    else:
        s1_aln, s2_aln = s1, s2

    s1_raw, s2_raw = map(rm_dash, [s1_aln, s2_aln])

    s1_num, s2_num = list(s1_numbering), list(s2_numbering)

    if len(s1_raw) > len(s1_num):
        raise LengthMismatch('s1 is larger than a corresponding numbering')
    if len(s2_raw) > len(s2_num):
        raise LengthMismatch('s2 is larger than a corresponding numbering')
    if len(s1_aln) != len(s2_aln):
        raise LengthMismatch(
            f'Lengths of aligned seqs must match; '
            f'(len(s1)={len(s1)} != len(s2)={len(s2)})'
        )

    s1_pool, s2_pool = iter(s1_num), iter(s2_num)

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


if __name__ == '__main__':
    raise RuntimeError
