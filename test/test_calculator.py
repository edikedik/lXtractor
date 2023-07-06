import pytest
from toolz import curry

from lXtractor.core.chain import ChainStructure
from lXtractor.variables.base import SequenceVariable
from lXtractor.variables.calculator import GenericCalculator
from lXtractor.variables.sequential import SeqEl, make_str, PFP
from lXtractor.variables.structural import Dist, AggDist


@curry
def count_char(seq: str, c: str):
    return sum(1 for x in seq if x == c)


@pytest.fixture
def chain_structure(four_chain_structure):
    return ChainStructure(next(four_chain_structure.split_chains()))


@pytest.fixture
def chain_structure_seq(chain_structure):
    seq = chain_structure.seq
    mapping = dict(zip(seq.numbering, range(1, len(seq) + 1)))
    return seq, mapping


SeqEl191 = SeqEl(191, 'str')
SeqEl191Res = (True, 'K')
TEST_SINGLES = (
    # Sequential
    (SeqEl191, SeqEl191Res),
    (SeqEl(1000, 'str'), (False, 'Missing 1000 in mapping')),
    (PFP(191, 1), (True, -4.99)),
    (
        make_str(reduce=count_char(c='E'), rtype=str)(start=178),
        (True, 5),
    ),
    (
        make_str(reduce=count_char(c='E'), rtype=str)(start=178, stop=189),
        (True, 4),
    ),
    # Structural
    (Dist(28, 191, 'CB', 'CB'), (True, 23.513)),
    (AggDist(28, 191, 'min'), (True, 19.607)),
)
TEST_ITERABLES = (
    ([True], [[SeqEl191, (PFP(191, 1))]], [[SeqEl191Res, (True, -4.99)]]),
    (
        [True, False],
        [[SeqEl191], [(Dist(28, 191, 'CB', 'CB'))]],
        [[SeqEl191Res], [(True, 23.513)]],
    ),
    (  # check vs broadcasting
        [True, True],
        [SeqEl191, SeqEl191],
        [[SeqEl191Res, SeqEl191Res], [SeqEl191Res, SeqEl191Res]],
    ),
)
TEST_MAP = (([SeqEl191, SeqEl191], [SeqEl191Res, SeqEl191Res]),)
EPS = 1e-3


def comp(actual, res, is_float):
    if is_float:
        return actual[0] == res[0] and abs(actual[1] - res[1]) <= EPS
    else:
        return actual == res


def get_obj(is_seq_str, chain_structure_seq, chain_structure):
    if is_seq_str:
        o, m = chain_structure_seq
        o = o.seq1
    else:
        o, m = chain_structure.structure, None
    return o, m


@pytest.mark.parametrize('v,res', TEST_SINGLES)
@pytest.mark.parametrize('num_proc', [None, 2])
def test_call_singles(v, res, chain_structure, chain_structure_seq, num_proc):
    calc = GenericCalculator(num_proc=num_proc)
    o, m = get_obj(
        isinstance(v, SequenceVariable), chain_structure_seq, chain_structure
    )
    actual = calc(o, v, m)
    is_float = v.rtype == float
    assert comp(actual, res, is_float)


@pytest.mark.parametrize('is_seq,v,res', TEST_ITERABLES)
@pytest.mark.parametrize('num_proc', [None, 2])
def test_call_iterables(is_seq, v, res, num_proc, chain_structure_seq, chain_structure):
    calc = GenericCalculator(num_proc=num_proc)
    inputs = [get_obj(x, chain_structure_seq, chain_structure) for x in is_seq]
    o = (x[0] for x in inputs)
    m = (x[1] for x in inputs)
    calc_res = calc(o, v, m)
    for xs, rr in zip(calc_res, res, strict=True):
        for actual, r in zip(xs, rr, strict=True):
            is_float = isinstance(r[1], float)
            assert comp(actual, r, is_float)


@pytest.mark.parametrize('v,res', TEST_MAP)
@pytest.mark.parametrize('num_proc', [None, 2])
def test_map(v, res, num_proc, chain_structure_seq):
    calc = GenericCalculator(num_proc=num_proc)
    o, m = chain_structure_seq
    calc_res = list(calc.map(o.seq1, v, m))
    assert calc_res == res


@pytest.mark.parametrize('num_proc', [None, 2])
def test_vmap(num_proc, chain_structure_seq):
    calc = GenericCalculator(num_proc=num_proc)
    o, m = chain_structure_seq
    s = o.seq1
    calc_res = list(calc.vmap([s, s], SeqEl191, m))
    assert calc_res == [SeqEl191Res, SeqEl191Res]
