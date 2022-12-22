import pytest
from more_itertools import unzip
from toolz import curry

from lXtractor.core.chain import ChainStructure
from lXtractor.variables.base import SequenceVariable
from lXtractor.variables.calculator import SimpleCalculator, ParallelCalculator
from lXtractor.variables.sequential import SeqEl, make_str, PFP
from lXtractor.variables.structural import Dist, AggDist


@curry
def count_char(seq: str, c: str):
    return sum(1 for x in seq if x == c)


TEST_VARIABLES = (
    # Sequential
    (SeqEl(191), (True, 'K')),
    (SeqEl(1000), (False, 'Missing 1000 in mapping')),
    (PFP(191, 1), (True, -4.99)),
    (
        make_str(reduce=count_char(c='E'), rtype=str)(start=178),
        (True, 5),
    ),
    (
        make_str(reduce=count_char(c='E'), rtype=str)(start=178, stop=189),
        (True, 4),
    ),
    (Dist(28, 191, 'CB', 'CB'), 23.513),
    (AggDist(28, 191, 'min'), 19.607),
)
TEST_SEQ_VS = list(filter(lambda x: isinstance(x[0], SequenceVariable), TEST_VARIABLES))
TEST_STR_VS = list(
    filter(lambda x: not isinstance(x[0], SequenceVariable), TEST_VARIABLES)
)
EPS = 1e-3


@pytest.fixture(scope='module')
def simple_calc():
    return SimpleCalculator()


@pytest.fixture(scope='module')
def parallel_calc():
    return ParallelCalculator(num_proc=2)


@pytest.fixture
def chain_structure(four_chain_structure):
    return ChainStructure.from_structure(next(four_chain_structure.split_chains()))


@pytest.fixture
def chain_structure_seq(chain_structure):
    seq = chain_structure.seq
    mapping = dict(zip(seq.numbering, range(1, len(seq) + 1)))
    return seq, mapping


@pytest.mark.parametrize('v,res', TEST_VARIABLES)
def test_call(v, res, simple_calc, chain_structure, chain_structure_seq):
    if isinstance(v, SequenceVariable):
        seq, mapping = chain_structure_seq
        assert simple_calc(seq.seq1, v, mapping) == res
    else:
        if v.rtype == float:
            assert abs(simple_calc(chain_structure.array, v, None)[1] - res) <= EPS
        else:
            assert simple_calc(chain_structure.array, v, None) == res


@pytest.mark.parametrize('v,res', TEST_VARIABLES)
@pytest.mark.skip('Not possible to pickle Range variables yet')
def test_call_parallel(v, res, parallel_calc, chain_structure, chain_structure_seq):
    if isinstance(v, SequenceVariable):
        seq, mapping = chain_structure_seq
        assert next(parallel_calc([seq.seq1], [(v,)], mapping))[0] == res
    else:
        if v.rtype == float:
            assert (
                abs(next(parallel_calc([chain_structure.array], [(v,)], None))[0][1])
                - res
                <= EPS
            )
        else:
            assert next(parallel_calc([chain_structure.array], [(v,)], None)) == res


# @pytest.mark.parametrize('vs', [TEST_SEQ_VS, TEST_STR_VS])
# def test_map(vs, calc, chain_structure, chain_structure_seq):
#     if isinstance(vs[0], SequenceVariable):
#         seq, mapping = chain_structure_seq
#         vs, results = unzip(vs)
#         results = calc.map()
#         assert simple_calc(seq.seq1, v, mapping) == res
#     else:
#         if v.rtype == float:
#             assert abs(simple_calc(chain_structure.array, v, None)[1] - res) <= EPS
