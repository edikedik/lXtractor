import logging
import typing as t
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from Bio import SeqIO

from lXtractor.core.base import Seq, SeqRec, AbstractResource, AmbiguousData, _Add_method, _Align_method
from lXtractor.util.seq import mafft_add, mafft_align, remove_gap_columns, remove_gap_sequences

_GAP = '-'
_Idx = t.Union[int, t.Tuple[int, ...]]
LOGGER = logging.getLogger(__name__)


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
    def shape(self) -> t.Tuple[int, int]:
        """
        :return: (# sequences, # columns)
        """
        num_rec = 0 if not self.seqs else len(self.seqs)
        aln_len = 0 if not self.seqs else len(self.seqs[0])
        return num_rec, aln_len

    def __repr__(self) -> str:
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
            raise ValueError(f'Expected all seqs to have the same length, got {lengths}')

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
        Uses :func:`remove_gap_columns`.

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

    def fetch(self, url: str):
        """
        A method for subclasses to define fetch strategy to acquire resource.
        """
        raise NotImplementedError

    def dump(self, path: t.Union[str, Path]) -> int:
        """
        :param path: Path to save seqs.
        :return: The number of written seqs.
        """
        num_dumped = SeqIO.write(self.seqs, path, self.fmt)
        LOGGER.debug(f'Wrote {num_dumped} sequences to {path}')
        return num_dumped

    def add_sequences(
            self, seqs: t.Sequence[SeqRec], overwrite: bool = False,
    ) -> t.Tuple[t.Sequence[SeqRec], t.Sequence[SeqRec]]:
        """
        Add given sequences to the alignment.

        :param seqs: Sequences to add.
        :param overwrite: Overwrite existing :attr:`seqs` with the results.
        :return: A tuple with two elements:
            (1) a list of sequences after addition ``seqs``,
            (2) a list of added sequences extracted from the alignment after the addition.
        """

        LOGGER.debug(f'Adding {len(seqs)} sequences using the method {self.add_method}')

        if self._handle is None:
            self._handle = NamedTemporaryFile('w')
            SeqIO.write(self.seqs, self._handle.name, 'fasta')
            self._handle.seek(0)
            LOGGER.debug(f'Created new temporary file with the MSA {self._handle.name}')

        prev_shape = self.shape
        alignment, added = self.add_method(Path(self._handle.name), seqs)
        curr_shape = (len(alignment), len(alignment[0]))

        if prev_shape[1] != curr_shape[1]:
            raise ValueError(f'Expected to preserve the number of columns '
                             f'(shape_before={prev_shape},shape_now={curr_shape})')

        if overwrite:
            self.seqs = alignment

        return alignment, added

    def remove_sequences(
            self, ids: t.Collection[str], overwrite: bool = True
    ) -> t.List[SeqRec]:
        """
        Remove sequences with given ids from the alignment.

        :param ids: A collection of IDs to remove.
        :param overwrite: Overwrite existing :attr:`seqs` with the result.
        :return: A list of seqs without the excluded ones.
        """
        LOGGER.debug(f'Removing {len(ids)} sequences with the following ids: {list(ids)}')
        alignment = [s for s in self.seqs if s.id not in ids]
        if overwrite:
            self.seqs = alignment
        return alignment

    def map_seq_numbering(
            self, seq: SeqRec,
            seq_numbering: t.Sequence[int],
            seq_added: bool = False,
    ) -> t.Dict[int, int]:
        # TODO: support matching seq by ID and sequence to avoid unneeded alignments
        """
        Map between a sequence's numbering to the alignment columns.

        :param seq: A sequence whose numbering to map.
        :param seq_numbering: Optional numbering of the provided seq.
            Must be the same length as ``seq``.
        :param seq_added: Do not call :meth:`add_sequences`,
            assuming the sequence is already added.
        :return: A dictionary mapping ``seq_numbering`` to alignment's columns indices.
        """

        if not len(seq_numbering) == len(seq):
            raise AmbiguousData(
                f'Numbering length {len(seq_numbering)} does not match '
                f'the sequence length {len(seq)}')

        LOGGER.debug(
            f'Mapping between sequence {seq.id} (size {len(seq)}) '
            f'and the alignment columns numbering.')

        if seq_added:
            aligned_msa = seq
        else:
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
        ori2aligned = (i for i, c in zip(seq_numbering, aligned_ori) if c != '-')

        return {i: next(ori2aligned) for i, c in
                enumerate(aligned_msa, start=1) if c != '-'}


if __name__ == '__main__':
    raise RuntimeError
