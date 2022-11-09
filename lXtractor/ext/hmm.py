import typing as t
from collections import abc
from itertools import count
from pathlib import Path

from pyhmmer.easel import DigitalSequence, TextSequence
from pyhmmer.plan7 import HMMFile, Pipeline, TopHits, Alignment, Domain

from lXtractor.core.chain import ChainSequence, CT, ChainStructure, Chain


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
            domain_filter: abc.Callable[[Domain], bool] | None = None,
            **kwargs
    ) -> abc.Generator[CT, None, None]:

        def accept_domain(d: Domain) -> bool:
            if min_score:
                return d.score >= min_score
            if min_size:
                size = d.alignment.target_to - d.alignment.target_from
                return size >= min_size
            return True

        if not isinstance(objs, abc.Iterable):
            objs = [objs]

        if new_map_name is None:
            new_map_name = self.hmm.accession.decode('utf-8')

        objs_by_id: dict[str, CT] = {s.id: s for s in objs}

        # if self.hits is None:
        self.search(objs_by_id.values())

        for hit in self.hits:
            obj = objs_by_id[hit.accession.decode('utf-8')]

            for dom_i, dom in enumerate(hit.domains, start=1):
                if not accept_domain(dom):
                    continue
                if domain_filter and not domain_filter(dom):
                    continue
                aln = dom.alignment
                name = f'{new_map_name}_{dom_i}'
                sub = obj.spawn_child(aln.target_from, aln.target_to, name, **kwargs)
                num = [hmm_i for seq_i, hmm_i in enumerate_numbering(aln) if seq_i]
                seq = sub.seq if isinstance(obj, (Chain, ChainStructure)) else sub
                seq.add_seq(new_map_name, num)
                seq.meta[f'{new_map_name}_pvalue'] = dom.pvalue
                seq.meta[f'{new_map_name}_score'] = dom.score
                seq.meta[f'{new_map_name}_bias'] = dom.bias
                yield sub


def enumerate_numbering(a: Alignment) -> abc.Generator[tuple[int | None, int | None], None, None]:
    hmm_pool, seq_pool = count(a.hmm_from), count(a.target_from)
    for hmm_c, seq_c in zip(a.hmm_sequence, a.target_sequence):
        hmm_i = None if hmm_c == '.' else next(hmm_pool)
        seq_i = None if seq_c == '-' else next(seq_pool)
        yield seq_i, hmm_i


if __name__ == '__main__':
    raise RuntimeError
