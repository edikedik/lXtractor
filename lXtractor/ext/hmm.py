import typing as t
from collections import abc
from itertools import count
from pathlib import Path

from pyhmmer.easel import DigitalSequence, TextSequence
from pyhmmer.plan7 import HMMFile, Pipeline, TopHits, Alignment

from lXtractor.core.chain import ChainSequence


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

    def convert_seq(self, seq: ChainSequence | str) -> DigitalSequence:
        match seq:
            case str():
                accession, name, text = hash(seq), None, seq
            case ChainSequence():
                accession, name, text = seq.id, seq.name, seq.seq1
            case _:
                raise TypeError(f'Unsupported sequence type {type(seq)}')
        return TextSequence(
            sequence=text, name=bytes(name, encoding='utf-8'),
            accession=bytes(accession, encoding='utf-8')
        ).digitize(self.hmm.alphabet)

    def search(
            self, seqs: abc.Iterable[str | ChainSequence | DigitalSequence]
    ) -> TopHits:
        if self.pipeline is None:
            self.init_pipeline()
        seqs = map(
            lambda s: self.convert_seq(s) if not isinstance(s, DigitalSequence) else s,
            seqs)
        self.hits = self.pipeline.search_hmm(self.hmm, seqs)
        return self.hits

    def annotate(
            self, seqs: abc.Iterable[ChainSequence] | ChainSequence,
            map_name: t.Optional[str] = None, **kwargs
    ) -> abc.Generator[ChainSequence, None, None]:

        if isinstance(seqs, ChainSequence):
            seqs = [seqs]
        if map_name is None:
            map_name = self.hmm.accession.decode('utf-8')
        seqs_by_id: dict[str, ChainSequence] = {s.id: s for i, s in enumerate(seqs, start=1)}

        if self.hits is None:
            self.search(seqs_by_id.values())

        for hit in self.hits:
            seq = seqs_by_id[hit.accession.decode('utf-8')]
            for dom in hit.domains:
                aln = dom.alignment
                sub = seq.spawn_child(aln.target_from, aln.target_to, **kwargs)
                num = [hmm_i for seq_i, hmm_i in enumerate_numbering(aln) if seq_i]
                sub.add_seq(map_name, num)
                sub.name = map_name
                sub.meta[f'{map_name}_pvalue'] = dom.pvalue
                sub.meta[f'{map_name}_score'] = dom.score
                sub.meta[f'{map_name}_bias'] = dom.bias
                yield sub


def enumerate_numbering(a: Alignment) -> abc.Generator[tuple[int | None, int | None], None, None]:
    hmm_pool, seq_pool = count(a.hmm_from), count(a.target_from)
    for hmm_c, seq_c in zip(a.hmm_sequence, a.target_sequence):
        hmm_i = None if hmm_c == '.' else next(hmm_pool)
        seq_i = None if seq_c == '-' else next(seq_pool)
        yield seq_i, hmm_i


if __name__ == '__main__':
    raise RuntimeError
