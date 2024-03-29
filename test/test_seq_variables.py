import pytest
from toolz import curry

from lXtractor.variables.sequential import SeqEl, PFP, make_str
from lXtractor.core.exceptions import FailedCalculation


def test_seq_el(simple_chain_seq):
    s = simple_chain_seq
    s.add_seq("I", [1, 2, 3, 4, 5])

    v1, v2, v3 = SeqEl(1), SeqEl(1, "I"), SeqEl(10, "I")

    r1, r2, r3 = (
        v1.calculate(s.seq1),
        v2.calculate(s["I"]),
        v3.calculate(s["I"], {10: 5}),
    )
    assert (r1, r2, r3) == ("A", 1, 5)

    with pytest.raises(FailedCalculation):
        SeqEl(10).calculate(s.seq1)


def test_prot_fp(simple_chain_seq):
    s = simple_chain_seq
    # 1st component of A and 5th component of E
    assert PFP(1, 1).calculate(s.seq1), PFP(5, 5).calculate(s.seq1) == (-0.1, -2.14)

    with pytest.raises(FailedCalculation):
        PFP(1, 6).calculate(s.seq1)

    with pytest.raises(FailedCalculation):
        PFP(6, 1).calculate(s.seq1)

    with pytest.raises(FailedCalculation):
        s.add_seq("I", [1, 2, 3, 4, 5])
        PFP(6, 1).calculate(s["I"])


def test_range_map_reduce_factory():
    v = make_str(sum, float)()
    assert v.id == "SliceSum(start=None,stop=None,step=None,seq_name='seq1')"
    assert v.rtype is float

    v = make_str(
        sum,
        int,
        transform=lambda _: "whatever",
        reduce_name="Summer",
        transform_name="whatever",
    )
    assert v.__name__ == "SliceWhateverSummer"


@curry
def count_char(seq: str, c: str):
    return sum(1 for x in seq if x == c)


def test_range_map_reduce(simple_chain_seq):
    s = simple_chain_seq
    s.add_seq("X", [1, 2, 3, 2, 1])
    v = make_str(sum, float)(start=1, stop=2)
    assert v.calculate(s["X"]) == 3
    v = make_str(count_char(c="A"), int, reduce_name="CounterA")()
    assert v.calculate(s.seq1) == 5
    v = make_str(
        count_char(c="e"),
        int,
        transform=lambda x: "".join(x).lower(),
        reduce_name="counter",
        transform_name="lower",
    )()
    assert v.calculate(s.seq1) == 0
