import logging
import typing as t
from collections import abc
from itertools import count
from pathlib import Path

from more_itertools import peekable
from pyhmmer.easel import DigitalSequence, TextSequence
from pyhmmer.plan7 import HMMFile, Pipeline, TopHits, Alignment, Domain

from lXtractor.core.chain import ChainSequence, CT, ChainStructure, Chain
from lXtractor.core.exceptions import MissingData

LOGGER = logging.getLogger(__name__)
HMM_DEFAULT_NAME = 'HMM'


class PyHMMer:
    """
    A basis pyhmmer interface aimed at domain extraction.
    It works with a single hmm model and pipeline instance.
    """

    def __init__(self, hmm: HMMFile | Path | str, **kwargs):
        match hmm:
            case HMMFile():
                self.hmm = hmm
            case _:
                with HMMFile(hmm) as f:
                    self.hmm = next(f)
        self.pipeline: Pipeline = self.init_pipeline(**kwargs)
        self.hits: TopHits | None = None

    def init_pipeline(self, **kwargs) -> Pipeline:
        self.pipeline = Pipeline(self.hmm.alphabet, **kwargs)
        return self.pipeline

    def convert_seq(self, obj: CT | str) -> DigitalSequence:
        match obj:
            case str():
                accession, name, text = hash(obj), None, obj
            case ChainSequence():
                accession, name, text = obj.id, obj.name, obj.seq1
            case ChainStructure() | Chain():
                accession, name, text = obj.id, obj.id, obj.seq.seq1
            case _:
                raise TypeError(f'Unsupported sequence type {type(obj)}')
        return TextSequence(
            sequence=text, name=bytes(name, encoding='utf-8'),
            accession=bytes(accession, encoding='utf-8')
        ).digitize(self.hmm.alphabet)

    def search(
            self, seqs: abc.Iterable[str | CT | DigitalSequence]
    ) -> TopHits:
        if self.pipeline is None:
            self.init_pipeline()
        seqs = map(
            lambda s: self.convert_seq(s) if not isinstance(s, DigitalSequence) else s,
            seqs)
        self.hits = self.pipeline.search_hmm(self.hmm, seqs)
        return self.hits

    def annotate(
            self, objs: abc.Iterable[CT] | CT,
            new_map_name: t.Optional[str] = None,
            min_score: float | None = None,
            min_size: int | None = None,
            min_cov: float | None = None,
            domain_filter: abc.Callable[[Domain], bool] | None = None,
            **kwargs
    ) -> abc.Generator[CT, None, None]:

        def accept_domain(d: Domain, cov: float) -> bool:
            acc_cov = min_cov is None or cov >= min_cov
            acc_score = min_score is None or d.score >= min_score
            acc_size = min_size is None or (
                    d.alignment.target_to - d.alignment.target_from >= min_size)
            return all([acc_cov, acc_score, acc_size])

        if not isinstance(objs, abc.Iterable):
            objs = [objs]
        else:
            peeking = peekable(objs)
            if not peeking.peek(False):
                raise MissingData('No sequences provided')

        if new_map_name is None:
            try:
                new_map_name = self.hmm.accession.decode('utf-8')
            except ValueError:
                try:
                    new_map_name = self.hmm.name.decode(
                        'utr-8').replace(' ', '_').replace('-', '_')
                except ValueError:
                    LOGGER.warning(
                        'new_map_name was not provided, and neither `accession` nor `name` '
                        f'attribute exist: falling back to default name: {HMM_DEFAULT_NAME}')
                    new_map_name = HMM_DEFAULT_NAME

        objs_by_id: dict[str, CT] = {s.id: s for s in objs}

        self.search(objs_by_id.values())

        for hit in self.hits:
            obj = objs_by_id[hit.accession.decode('utf-8')]

            for dom_i, dom in enumerate(hit.domains, start=1):
                aln = dom.alignment
                num = [hmm_i for seq_i, hmm_i in enumerate_numbering(aln) if seq_i]
                coverage = sum(1 for x in num if x is not None) / len(num)

                if not accept_domain(dom, coverage):
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
                seq.meta[f'{new_map_name}_cov'] = coverage
                yield sub


def enumerate_numbering(a: Alignment) -> abc.Generator[tuple[int | None, int | None], None, None]:
    hmm_pool, seq_pool = count(a.hmm_from), count(a.target_from)
    for hmm_c, seq_c in zip(a.hmm_sequence, a.target_sequence):
        hmm_i = None if hmm_c == '.' else next(hmm_pool)
        seq_i = None if seq_c == '-' else next(seq_pool)
        yield seq_i, hmm_i


if __name__ == '__main__':
    raise RuntimeError
