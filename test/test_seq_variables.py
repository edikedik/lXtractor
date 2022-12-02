import pytest

from lXtractor.variables.sequential import SeqEl, PFP
from lXtractor.core.exceptions import FailedCalculation


def test_seq_el(simple_chain_seq):
    _, s = simple_chain_seq
    s.add_seq('I', [1, 2, 3, 4, 5])

    v1, v2, v3 = SeqEl(1), SeqEl(1, 'I'), SeqEl(10, 'I')

    r1, r2, r3 = (
        v1.calculate(s.seq1),
        v2.calculate(s['I']),
        v3.calculate(s['I'], {10: 5})
    )
    assert (r1, r2, r3) == ('A', 1, 5)

    with pytest.raises(FailedCalculation):
        SeqEl(10).calculate(s.seq1)


def test_prot_fp(simple_chain_seq):
    _, s = simple_chain_seq
    assert (
        PFP(1, 1).calculate(s.seq1), PFP(5, 5).calculate(s.seq1) ==
        -0.1, -2.14
    )

    with pytest.raises(FailedCalculation):
        PFP(1, 6).calculate(s.seq1)

    with pytest.raises(FailedCalculation):
        PFP(6, 1).calculate(s.seq1)

    with pytest.raises(FailedCalculation):
        s.add_seq('I', [1, 2, 3, 4, 5])
        PFP(6, 1).calculate(s['I'])
