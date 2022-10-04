from more_itertools import take

from lXtractor.variables.calculator import SimpleCalculator
from lXtractor.variables.manager import Manager
from lXtractor.variables.sequential import SeqEl


def test_variables_modifications(sample_chain_list, simple_chain_variables):
    cl = sample_chain_list
    vs = simple_chain_variables
    manager = Manager()

    manager.assign(vs, cl, level=0)
    assert all(c.seq.variables for c in cl)
    assert all(not c.seq.variables for c in cl.collapse_children())

    manager.assign(vs, cl, level=2)
    l1, l2 = take(2, cl.iter_children())
    assert all([c.seq.variables for c in l2])
    assert all(len(c.seq.variables) == 1 for c in l2)
    assert all(not s.seq.variables for s in l1)

    manager.assign(vs, cl, level=1, obj_type='str')
    assert all(not s.seq.variables for s in l1)
    assert all(s.variables for s in l1.iter_structures())
    assert all(len(s.variables) == 2 for s in l1.iter_structures())

    manager.remove(cl, level=1, obj_type='str')
    assert all(not s.seq.variables for s in l1)
    assert all([len(s.variables) == 0 for s in l1.iter_structures()])

    cl[0].seq.variables[SeqEl(1)] = 'A'
    manager.reset(cl, level=0, obj_type='seq')
    assert cl[0].seq.variables[SeqEl(1)] is None


def test_aggregate(sample_chain_list, simple_chain_variables):
    cl = sample_chain_list
    vs = simple_chain_variables
    manager = Manager()
    manager.assign(vs, cl, level=0)
    df = manager.aggregate(cl, id_contains='k')
    assert len(df) == 1


def test_staging(sample_chain_list, simple_chain_variables):
    cl = sample_chain_list
    vs = simple_chain_variables
    manager = Manager()

    manager.assign(vs, cl, level=1, obj_type=['str', 'str_seq'])
    staged_seq, staged_str = map(list, manager.stage_calculations(cl))
    assert len(staged_seq) == 4
    assert len(staged_str) == 4


def test_calculate(sample_chain_list, simple_chain_variables):
    cl = sample_chain_list
    vs = simple_chain_variables
    manager = Manager()
    calculator = SimpleCalculator()

    manager.assign(vs, cl, level=1, obj_type=['str', 'str_seq'])
    successes, failures = manager.calculate(cl, calculator)
    assert successes
    assert not failures
    # four seq elements, four dihedrals, four distances
    assert len(successes) == 12
    assert len(manager.aggregate(cl)) == 12
