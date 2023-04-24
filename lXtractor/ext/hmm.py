"""
Wrappers around PyHMMer for convenient annotation of domains and families.
"""
import logging
import typing as t
from collections import abc
from itertools import count
from pathlib import Path

from more_itertools import peekable
from pyhmmer.easel import (
    DigitalSequence,
    TextSequence,
    DigitalSequenceBlock,
    TextMSA,
)
from pyhmmer.plan7 import (
    HMM,
    HMMFile,
    Pipeline,
    TopHits,
    Alignment,
    Domain,
    TraceAligner,
)

from lXtractor.core.chain import ChainSequence, ChainStructure, Chain
from lXtractor.core.exceptions import MissingData

LOGGER = logging.getLogger(__name__)

HMM_DEFAULT_NAME = 'HMM'
_ChainT: t.TypeAlias = Chain | ChainStructure | ChainSequence
_SeqT: t.TypeAlias = _ChainT | str | tuple[str, str] | DigitalSequence
_HmmInpT: t.TypeAlias = HMM | HMMFile | Path | str
CT = t.TypeVar('CT', bound=t.Union[ChainSequence, ChainStructure, Chain])


# TODO: verify non-domain HMM types (e.g., Motif, Family) yield valid Domain hits


class PyHMMer:
    """
    A basis pyhmmer interface aimed at domain extraction.
    It works with a single hmm model and pipeline instance.

    `The original documentation <https://pyhmmer.readthedocs.io/en/stable/>`.
    """

    def __init__(self, hmm: _HmmInpT, **kwargs):
        """
        :param hmm: An :class:`HMMFile` handle or path as string or `Path`
            object to a file containing a single HMM model. In case of multiple
            models, only the first one will be taken
        :param kwargs: Passed to :class:`Pipeline`. The `alphabet` argument
            is derived from the supplied `hmm`.
        """
        try:
            #: HMM instance
            self.hmm = next(_iter_hmm(hmm))
        except (StopIteration, EOFError) as e:
            raise MissingData(f'Invalid input {hmm}') from e
        #: Pipeline to use for HMM searches
        self.pipeline: Pipeline = self.init_pipeline(**kwargs)
        #: Hits resulting from the most recent HMM search
        self.hits_: TopHits | None = None

    @classmethod
    def from_multiple(cls, hmm: _HmmInpT, **kwargs) -> abc.Generator['PyHMMer']:
        """
        Split HMM collection and initialize a :class:`PyHMMer` instance from
        each HMM model.

        :param hmm: A path to HMM file, opened HMMFile handle, or parsed HMM.
        :param kwargs: Passed to the class constructor.
        :return:
        """
        yield from (cls(inp, **kwargs) for inp in _iter_hmm(hmm))

    def init_pipeline(self, **kwargs) -> Pipeline:
        """
        :param kwargs: Passed to :class:`Pipeline` during initialization.
        :return: Initialized pipeline, also saved to :attr:`pipeline`.
        """
        self.pipeline = Pipeline(self.hmm.alphabet, **kwargs)
        return self.pipeline

    def convert_seq(self, obj: t.Any) -> DigitalSequence:
        """
        :param obj: A `Chain*`-type object or string or a tuple of (name, seq).
            A sequence of this object must be compatible with the alphabet of
            the HMM model.
        :return: A digitized sequence compatible with PyHMMer.
        """
        match obj:
            case DigitalSequence():
                return obj
            case str():
                _id = str(hash(obj))
                accession, name, text = _id, _id, obj
            case [str(), str()]:
                accession, name, text = obj[0], obj[0], obj[1]
            case ChainSequence():
                accession, name, text = obj.id, obj.name, obj.seq1
            case ChainStructure() | Chain():
                accession, name, text = obj.id, obj.id, obj.seq.seq1
            case _:
                raise TypeError(f'Unsupported sequence type {type(obj)}')
        return TextSequence(
            sequence=text,
            name=bytes(name, encoding='utf-8'),
            accession=bytes(accession, encoding='utf-8'),
        ).digitize(self.hmm.alphabet)

    def _convert_to_seq_block(self, seqs: abc.Iterable[_SeqT]) -> DigitalSequenceBlock:
        return DigitalSequenceBlock(self.hmm.alphabet, map(self.convert_seq, seqs))

    def search(self, seqs: abc.Iterable[_SeqT]) -> TopHits:
        """
        Run the :attr:`pipeline` to search for :attr:`hmm`.

        :param seqs: Iterable over digital sequences or objects accepted by
            :meth:`convert_seq`.
        :return: Top hits resulting from the search.
        """
        if self.pipeline is None:
            self.init_pipeline()

        seqs_block = self._convert_to_seq_block(seqs)
        self.hits_ = self.pipeline.search_hmm(self.hmm, seqs_block)
        return self.hits_

    def align(self, seqs: abc.Iterable[_SeqT]) -> TextMSA:
        """
        Align sequences to a profile.

        :param seqs: Sequences to align.
        :return: :class:`TextMSA` with aligned sequences.
        """
        block = self._convert_to_seq_block(seqs)
        aligner = TraceAligner()
        traces = aligner.compute_traces(self.hmm, block)
        msa = aligner.align_traces(self.hmm, block, traces)
        assert isinstance(msa, TextMSA), 'Unexpected MSA type returned'
        return msa

    def annotate(
        self,
        objs: abc.Iterable[_ChainT] | _ChainT,
        new_map_name: str | None = None,
        min_score: float | None = None,
        min_size: int | None = None,
        min_cov_hmm: float | None = None,
        min_cov_seq: float | None = None,
        domain_filter: abc.Callable[[Domain], bool] | None = None,
        **kwargs,
    ) -> abc.Generator[CT, None, None]:
        """
        Annotate provided objects by domain hits resulting from the HMM search.

        An annotation is the creation of a child object via :meth:`spawn_child`
        method (e.g., :meth:`lXtractor.core.chain.ChainSequence.spawn_child`).

        :param objs: A single one or an iterable over `Chain*`-type objects.
        :param new_map_name: A name for a child
            :class:`ChainSequence <lXtractor.core.chain.ChainSequence` to hold
            the mapping to the hmm numbering.
        :param min_score: Min hit score.
        :param min_size: Min hit size.
        :param min_cov_hmm: Min HMM model coverage -- a fraction of
            mapped / total nodes.
        :param min_cov_seq: Min coverage of a sequence by the HMM model
            nodes -- a fraction of mapped nodes to the sequence's length.
        :param domain_filter: A callable to filter domain hits.
        :param kwargs: Passed to the `spawn_child` method.
            **Hint:** if you don't want to keep spawned children,
            pass ``keep=False`` here.
        :return: A generator over spawned children yielded sequentially
            for each input object and valid domain hit.
        """

        def accept_domain(d: Domain, cov_hmm: float, cov_seq: float) -> bool:
            acc_cov_hmm = min_cov_hmm is None or cov_hmm >= min_cov_hmm
            acc_cov_seq = min_cov_seq is None or cov_seq >= min_cov_seq
            acc_score = min_score is None or d.score >= min_score
            acc_size = min_size is None or (
                d.alignment.target_to - d.alignment.target_from >= min_size
            )
            return all([acc_cov_hmm, acc_cov_seq, acc_score, acc_size])

        if isinstance(objs, _ChainT):
            objs = [objs]
        else:
            peeking = peekable(objs)
            fst = peeking.peek(False)
            if not fst:
                raise MissingData('No sequences provided')
            if not isinstance(fst, _ChainT):
                raise TypeError(f'Unsupported type {fst}')

        if new_map_name is None:
            try:
                new_map_name = self.hmm.accession.decode('utf-8').split('.')[0]
            except ValueError:
                try:
                    new_map_name = (
                        self.hmm.name.decode('utr-8')
                        .replace(' ', '_')
                        .replace('-', '_')
                    )
                except ValueError:
                    LOGGER.warning(
                        '`new_map_name` was not provided, and neither `accession` '
                        'nor `name` attribute exist: falling back to default '
                        f'name: {HMM_DEFAULT_NAME}'
                    )
                    new_map_name = HMM_DEFAULT_NAME

        objs_by_id: dict[str, CT] = {s.id: s for s in objs}

        self.search(objs_by_id.values())

        for hit in self.hits_:
            obj = objs_by_id[hit.accession.decode('utf-8')]

            for dom_i, dom in enumerate(hit.domains, start=1):
                aln = dom.alignment

                # compute coverages
                # HMM: - 1 2 3 - 4 - 5
                # SEQ: A - A A A A - A

                # HMM node numbers falling onto valid sequence elements
                # => - 2 3 - 4 5
                num = [hmm_i for seq_i, hmm_i in _enumerate_numbering(aln) if seq_i]

                # n = the number of valid HMM nodes covered by seq
                # => 2 3 4 5 => 4
                n = sum(1 for x in num if x is not None)

                # SEQ coverage 4 / 6
                cov_seq = n / len(num)
                # HMM coverage 4 / M
                cov_hmm = n / self.hmm.M

                if not accept_domain(dom, cov_hmm, cov_seq):
                    continue
                if domain_filter and not domain_filter(dom):
                    continue

                name = f'{new_map_name}_{dom_i}'
                sub = obj.spawn_child(aln.target_from, aln.target_to, name, **kwargs)

                seq = sub.seq if isinstance(obj, (Chain, ChainStructure)) else sub
                seq.add_seq(new_map_name, num)
                seq.meta[f'{new_map_name}_pvalue'] = dom.pvalue
                seq.meta[f'{new_map_name}_score'] = dom.score
                seq.meta[f'{new_map_name}_bias'] = dom.bias
                seq.meta[f'{new_map_name}_cov_seq'] = str(cov_seq)
                seq.meta[f'{new_map_name}_cov_hmm'] = str(cov_hmm)
                yield sub


def _iter_hmm(hmm: _HmmInpT) -> abc.Generator[HMM]:
    match hmm:
        case HMMFile():
            yield from hmm
        case HMM():
            yield hmm
        case _:
            with HMMFile(hmm) as f:
                yield from f


def _enumerate_numbering(
    a: Alignment,
) -> abc.Generator[tuple[int | None, int | None], None, None]:
    hmm_pool, seq_pool = count(a.hmm_from), count(a.target_from)
    for hmm_c, seq_c in zip(a.hmm_sequence, a.target_sequence):
        hmm_i = None if hmm_c == '.' else next(hmm_pool)
        seq_i = None if seq_c == '-' else next(seq_pool)
        yield seq_i, hmm_i


if __name__ == '__main__':
    raise RuntimeError
