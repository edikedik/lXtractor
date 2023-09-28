import typing as t
from copy import deepcopy

from lXtractor.core.chain import Chain, ChainStructure, ChainSequence
from lXtractor.core.config import SeqNames
from lXtractor.core.structure import GenericStructure


def basic_chain_seq(start=1, end=10, fill='A', name='chain') -> ChainSequence:
    return ChainSequence(
        start, end, name=name, seqs={SeqNames.seq1: fill * (end - start + 1)}
    )


def basic_chain(start=1, end=10, fill='A', name='chain') -> Chain:
    return Chain.from_seq(basic_chain_seq(start, end, fill, name))


def sample_chain(prefix: str = 'c', structure: t.Optional[ChainStructure] = None):
    structures = [structure] if structure else None
    return Chain(
        basic_chain_seq(fill='K', name=f'{prefix}_root'),
        children=[
            Chain(
                basic_chain_seq(name=f'{prefix}1', end=5),
                children=[
                    Chain(basic_chain_seq(name=f'{prefix}1_1')),
                    Chain(
                        basic_chain_seq(name=f'{prefix}1_2', start=5),
                        children=[
                            Chain(
                                basic_chain_seq(name=f'{prefix}1_2_1', start=8),
                                structures=structures,
                            )
                        ],
                    ),
                ],
            ),
            Chain(
                basic_chain_seq(name=f'{prefix}2', end=8),
                structures=structures,
                children=[
                    Chain(basic_chain_seq(name=f'{prefix}2_1', start=2, end=8)),
                    Chain(basic_chain_seq(name=f'{prefix}2_2', start=8, end=9)),
                ],
            ),
        ],
    )


def get_fst_chain(s: GenericStructure) -> ChainStructure:
    return ChainStructure(next(s.split_chains()), s.name)


def mark_meta(s: ChainStructure) -> ChainStructure:
    seq = deepcopy(s.seq)
    seq.meta['X'] = 'x'
    return ChainStructure(s.structure, seq=seq)


if __name__ == '__main__':
    raise RuntimeError
