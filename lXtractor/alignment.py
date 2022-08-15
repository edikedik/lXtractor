import logging
import typing as t
from io import StringIO
from itertools import tee
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from Bio import SeqIO
from more_itertools import partition, ilen
from toolz import curry

from lXtractor.base import Seq, SeqRec, AbstractResource, AmbiguousData, LengthMismatch
from lXtractor.utils import run_sp

_GAP = '-'
_Add_method = t.Callable[
    [t.Union[t.Iterable[SeqRec], Path], t.Iterable[SeqRec]],
    t.Tuple[t.Sequence[SeqRec], t.Sequence[SeqRec]]
]
_Align_method = t.Callable[
    [t.Iterable[SeqRec]], t.Sequence[SeqRec]
]
_Idx = t.Union[int, t.Tuple[int, ...]]
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
    Align a collection of sequences to a profile using `hmmalign`
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


def seq_identity(
        seq1: SeqRec, seq2: SeqRec, align: bool = True,
        align_method: _Align_method = mafft_align
) -> float:
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
        align_method: _Align_method = mafft_align):
    """
    Fraction of ``seq`` covered by ``cover``
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


class Alignment(AbstractResource):
    """
    A MSA resource. Internally it is a list of proteins sequences having the same length.
    It is also a container with sequences accessible via `id` property.

    >>> from Bio.Seq import Seq
    >>> from Bio.SeqRecord import SeqRecord
    >>> seqs = [SeqRecord(Seq('-AAAWWWAAA-'), id='1'),
    ...         SeqRecord(Seq('AAAAWWWAAA-'), id='1'),
    ...         SeqRecord(Seq('-AAAWWWAAAA'), id='2')]
    >>> aln = Alignment(seqs)
    >>> assert all(s in aln for s in ['AAAWWWAAA', 'AAAAWWWAAA', 'AAAWWWAAAA'])
    >>> assert aln.shape == (3, 11)
    >>> assert len(aln['1']) == 2
    >>> assert aln['2'].seq == seqs[-1].seq
    >>> all_seqs, addition = aln.add_sequences(
    ...    [SeqRecord(Seq('WWW'), id='3'),
    ...     SeqRecord(Seq('AWWWA'), id='4')],
    ...    overwrite=True)
    >>> assert len(all_seqs) == 5
    >>> assert len(addition) == 2
    >>> assert 'WWW' in aln
    >>> assert 'AWWWA' in aln
    """

    def __init__(self, seqs: t.Optional[t.Sequence[SeqRec]] = None,
                 resource_path: t.Optional[Path] = None,
                 resource_name: t.Optional[str] = None,
                 fmt: str = 'fasta',
                 gap: str = _GAP,
                 add_method: _Add_method = mafft_add,
                 align_method: _Align_method = mafft_align):
        """
        :param seqs: a list of :class:`Bio.SeqRecord.SeqRecord` objects
        :param resource_path: a path to an alignment biopython-readable
        :param resource_name: a resource's name for readability
        :param fmt: alignment format
        :param gap: gap-character
        :param add_method: a callable two arguments:
            (1) MSA (path or iterable with seqs) and (2) sequences to add.
            The method should return a tuple of
            (1) MSA after addition (sequence of sequence records), and
            (2) (a sequence of) added sequences, separately.
            By default, it is :func:`lXtractor.alignment.mafft_add`
        :param align_method: A callable accepting an arbitrary collection
            of sequences, and returning their MSA.
            By default, it is :func:`lXtractor.alignment.mafft_align`
        """
        super().__init__(resource_path, resource_name)
        self.fmt, self.gap = fmt, gap
        self.add_method = add_method
        self.align_method = align_method
        self.seqs: t.Optional[t.List[SeqRec]] = seqs
        if self.seqs is None:
            LOGGER.debug('Initializing from file')
            self.read()
            try:
                self.parse()
            except NotImplementedError:
                pass
        self._matrix = (
            None if self.seqs is None else
            np.array([list(str(s.seq)) for s in self.seqs]))
        self._handle: t.Optional[NamedTemporaryFile] = None
        self._verify(self.seqs)

    @property
    def shape(self):
        """
        :return: (a number of sequences, an alignment length)
        """
        num_rec = 0 if not self.seqs else len(self.seqs)
        aln_len = 0 if not self.seqs else len(self.seqs[0])
        return num_rec, aln_len

    def __repr__(self):
        return f'{self.name},shape={self.shape}'

    def __contains__(self, item: t.Union[SeqRec, str]):
        if isinstance(item, SeqRec):
            seq = str(item.seq)
        elif isinstance(item, str):
            seq = item
        else:
            return False
        return any(s == seq for s in self.ungapped_seqs)

    def __len__(self):
        return len(self.seqs)

    def __iter__(self):
        return iter(self.seqs)

    def __getitem__(self, seq_id: str):
        targets = [s for s in self.seqs if s.id == seq_id]
        if not targets:
            return None
        if len(targets) == 1:
            return targets.pop()
        return targets

    def index(
            self, i: t.Optional[_Idx], j: t.Optional[_Idx], return_type: str = 'array'
    ) -> t.Union[np.ndarray, t.List[SeqRec], t.List[str]]:
        if i is None:
            i = tuple(range(self.shape[0]))
        if j is None:
            j = tuple(range(self.shape[1]))
        sub = self._matrix[np.array(i)][:, np.array(j)]
        if return_type == 'seq_rec':
            idx = i if isinstance(i, t.Sequence) else [i]
            seqs = ["".join(x) for x in sub]
            recs = [rec for rec_i, rec in enumerate(self.seqs) if rec_i in idx]
            return [
                SeqRec(Seq(seq), id=rec.id, description=rec.description, name=rec.name)
                for seq, rec in zip(seqs, recs)]
        elif return_type == 'str':
            return ["".join(x) for x in sub]
        return sub

    def _verify(self, seqs: t.Iterable[SeqRec]):
        lengths = set(len(s) for s in seqs)
        if len(lengths) > 1:
            raise ValueError(
                f'Expected all seqs to have the same length, got {lengths}')
        for seq in seqs:
            for char in seq:
                if char.isalpha() or char == self.gap:
                    continue
                raise ValueError(
                    f'Invalid character {char} in sequence {seq}')

    @property
    def ids(self) -> t.Optional[t.List[str]]:
        """
        :return: A list of :attr:`Bio.SeqRecord.SeqRecord.id`
        """
        if self.seqs is None:
            return None
        return [s.id for s in self.seqs]

    @property
    def ungapped_seqs(self, upper: bool = True) -> t.Optional[t.Iterator[str]]:
        """
        :param upper: uppercase the sequences.
        :return: an iterator over ungapped alignment sequences.
            `None` if there are no :attr:`Alignment.seqs`.
        """
        if self.seqs is None:
            return None
        for s in self.seqs:
            seq = str(s.seq.ungap())
            if upper:
                yield seq.upper()
            yield seq

    def read(self) -> t.List[str]:
        """
        Reads an alignment and stores it into :attr:`Alignment.seqs`.

        :return: a list of sequence records.
        """
        seqs = list(SeqIO.parse(self.path, self.fmt))
        LOGGER.debug(f'Read the seqs with {len(seqs)}')
        self._verify(seqs)
        self.seqs = seqs
        return seqs

    def remove_gap_columns(
            self, max_fraction_of_gaps: float = 1.0,
            overwrite: bool = True
    ) -> t.List[SeqRec]:
        """
        Removes alignment columns, such that their gap fraction
        exceeds a given threshold.

        :param max_fraction_of_gaps: gap threshold.
        :param overwrite: overwrite :attr:`~seqs` with the results.
        :return: a list with the same number of sequences
            and (potentially) filtered columns.
        """
        LOGGER.debug(f'Filtering out the seqs columns with > '
                     f'{max_fraction_of_gaps * 100}% gaps')
        seqs, _ = remove_gap_columns(self.seqs, max_fraction_of_gaps)
        if overwrite:
            self.seqs = seqs
        return seqs

    def remove_gap_sequences(
            self, max_fraction_of_gaps: float = 0.8,
            overwrite: bool = True
    ) -> t.Tuple[t.List[SeqRec], t.List[SeqRec]]:
        """
        Removes sequences, such that their gap fraction exceeds
        a given threshold.

        :param max_fraction_of_gaps: gap threshold.
        :param overwrite: overwrite :attr:`~seqs` with the results.
        :return: a potentially filtered list of sequences
            with the same number of columns
        """
        LOGGER.debug(f'Filtering out the seqs sequences with > '
                     f'{max_fraction_of_gaps * 100}% gaps')
        below_seq, above_seqs = remove_gap_sequences(self.seqs, max_fraction_of_gaps)
        if overwrite:
            self.seqs = below_seq
        return below_seq, above_seqs

    def parse(self):
        """
        A method for subclasses to define additional manipulations
        invoked right after applying :meth:`read`.
        """
        raise NotImplementedError

    def dump(self, path: str) -> None:
        num_dumped = SeqIO.write(self.seqs, path, self.fmt)
        LOGGER.debug(f'Wrote {num_dumped} sequences to {path}')

    def add_sequences(
            self, seqs: t.Collection[SeqRec], overwrite: bool = False,
    ) -> t.Tuple[t.Sequence[SeqRec], t.Sequence[SeqRec]]:

        LOGGER.debug(
            f'Adding {len(seqs)} sequences using the method {self.add_method}')

        if self._handle is None:
            self._handle = NamedTemporaryFile('w')
            SeqIO.write(self.seqs, self._handle.name, 'fasta')
            self._handle.seek(0)
            LOGGER.debug(
                f'Created new temporary file with the MSA {self._handle.name}')

        alignment, added = self.add_method(
            Path(self._handle.name), seqs)

        if overwrite:
            self.seqs = alignment
        return alignment, added

    def remove_sequences(
            self, ids: t.Collection[str], overwrite: bool = True
    ) -> t.List[SeqRec]:
        LOGGER.debug(
            f'Removing {len(ids)} sequences with the following ids: {list(ids)}')
        alignment = [s for s in self.seqs if s.id not in ids]
        if overwrite:
            self.seqs = alignment
        return alignment

    def map_seq_numbering(
            self, seq: SeqRec,
            seq_numbering: t.Sequence[int]
    ) -> t.Dict[int, int]:
        """
        Map between a sequence numbering and the alignment columns.

        :param seq:
        :param seq_numbering:
        :return:
        """

        if not len(seq_numbering) == len(seq):
            raise AmbiguousData(
                f'Numbering length {len(seq_numbering)} does not match '
                f'the sequence length {len(seq)}')
        LOGGER.debug(
            f'Mapping between sequence {seq.id} (size {len(seq)}) '
            f'and the alignment columns numbering.')
        # add a sequence to the seqs and extract it right away
        _, aligned_msa = self.add_sequences([seq], False)
        aligned_msa = aligned_msa[-1]
        LOGGER.debug(
            f'Aligned and extracted sequence {len(aligned_msa.id)} '
            f'with size {len(aligned_msa)}')

        # align the extracted sequence and the original one
        # extract the (second time) aligned sequence
        aligned_ori = self.align_method([seq, aligned_msa])[-1]
        LOGGER.debug(f'Aligned MSA-extracted and original sequence '
                     f'to obtain a sequence {aligned_ori.id} '
                     f'with size {len(aligned_ori)}')

        # Obtain the filtered numbering with the sequence elements
        # really present in the MSA-aligned sequence
        ori2aligned = (i for i, c in zip(
            seq_numbering, aligned_ori) if c != '-')

        return {i: next(ori2aligned) for i, c in
                enumerate(aligned_msa, start=1) if c != '-'}


if __name__ == '__main__':
    raise RuntimeError
