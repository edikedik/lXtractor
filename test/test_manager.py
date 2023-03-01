import pandas as pd

from lXtractor.core.chain import ChainSequence, ChainStructure, ChainList
from lXtractor.core.structure import GenericStructure
from lXtractor.variables.calculator import GenericCalculator
from lXtractor.variables.manager import Manager
from lXtractor.variables.sequential import SeqEl


def test_variables_modifications(sample_chain_list, simple_chain_variables):
    cl = sample_chain_list
    vs = simple_chain_variables
    manager = Manager()

    manager.assign(vs, cl.iter_sequences())
    assert all(c.seq.variables for c in cl)
    assert all(not c.seq.variables for c in cl.collapse_children())
    assert all(all(not s.variables for s in c.structures) for c in cl)

    structures = list(cl.collapse_children().iter_structures())
    manager.assign(vs, structures)
    assert all(len(s.variables) == 2 for s in structures)

    manager.remove(structures)
    assert all(len(s.variables) == 0 for s in structures)
    manager.reset(cl.iter_sequences())
    assert all(c.seq.variables for c in cl)
    assert all((x is None for x in c.seq.variables.values()) for c in cl)


def test_aggregate_from_chains(sample_chain_list, simple_chain_variables):
    manager = Manager()
    manager.assign(simple_chain_variables, sample_chain_list.iter_sequences())
    df = manager.aggregate_from_chains(sample_chain_list.iter_sequences())
    assert isinstance(df, pd.DataFrame) and len(df) == 2


def test_staging(sample_chain_list, simple_chain_variables):
    cl = sample_chain_list
    vs = simple_chain_variables
    manager = Manager()

    staged = list(manager.stage(cl.iter_sequences(), vs))
    assert len(staged) == len(cl)  # number of seq vars x number of seqs
    assert all(
        all(
            [
                isinstance(x[0], ChainSequence),
                isinstance(x[1], str),
                isinstance(x[2], list),
                x[3] is None,
            ]
        )
        for x in staged
    )
    structures = ChainList(cl.collapse_children().iter_structures())
    staged = list(manager.stage(structures, vs))
    assert len(staged) == len(structures)
    assert all(
        all(
            [
                isinstance(x[0], ChainStructure),
                isinstance(x[1], GenericStructure),
                isinstance(x[2], list),
                x[3] is None,
            ]
        )
        for x in staged
    )


def test_calculate(sample_chain_list, simple_chain_variables):
    cl = sample_chain_list
    vs = simple_chain_variables
    manager = Manager()
    calculator = GenericCalculator()
    results = list(manager.calculate(cl.iter_sequences(), vs, calculator))
    assert len(results) == 2

    structures = ChainList(cl.collapse_children().iter_structures())
    manager.assign(vs, structures)
    results = list(manager.calculate(structures, None, calculator))
    assert len(results) == len(structures) * 2


def test_aggregate_from_it(simple_chain_seq):
    cseq1 = ChainSequence.from_string('ACDEG', name='seq1')
    cseq2 = ChainSequence.from_string('GEDCA', name='seq2')
    manager = Manager()
    res = list(
        manager.calculate(
            [cseq1, cseq2], [SeqEl(1), SeqEl(2), SeqEl(6)], GenericCalculator()
        )
    )
    assert len(res) == 6

    df = manager.aggregate_from_it(res)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 4)
    assert len(df.dropna()) == 0

    df = manager.aggregate_from_it(res, replace_errors=False)
    assert len(df.dropna()) == 2

    df = manager.aggregate_from_it(res, vs_to_cols=False)
    assert df.shape == (6, 4)

    res = list(
        manager.calculate(
            [cseq1, cseq1], [SeqEl(1), SeqEl(2), SeqEl(6)], GenericCalculator()
        )
    )
    # sequences aren't unique -- conversions to df fails
    df = manager.aggregate_from_it(res)
    assert not isinstance(df, pd.DataFrame)
    df = manager.aggregate_from_it(res, vs_to_cols=False)
    assert isinstance(df, pd.DataFrame) and df.shape == (6, 4)
