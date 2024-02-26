import pytest

from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.structural import (
    Dist,
    Dihedral,
    PseudoDihedral,
    Phi,
    Psi,
    Omega,
)
from lXtractor.variables.sequential import SeqEl, make_str
from lXtractor.variables.parser import parse_var

EPS = 1e-3

# SEQ_VS = (SeqEl(191), make_str())
# STR_VS = ()


@pytest.mark.skip(reason="Need to refactor variable parsing first")
def test_parse_variable():
    # one variable, one position
    vs, ss, ds = parse_var("1")
    assert len(vs) == len(ss) == len(ds) == 1
    assert isinstance(vs[0], SeqEl)
    assert ss == ds == [None]

    vs, _, _ = parse_var("1_Phi")
    assert isinstance(vs[0], Phi)
    vs, _, _ = parse_var("1_Psi")
    assert isinstance(vs[0], Psi)
    vs, _, _ = parse_var("1_Omega")
    assert isinstance(vs[0], Omega)

    # one variable, two positions
    vs, _, _ = parse_var("1-2")
    assert isinstance(vs[0], Dist)
    assert vs[0].p1 == 1
    assert vs[0].p2 == 2
    assert vs[0].com is True
    vs, _, _ = parse_var("1:CB-2:CB")
    assert vs[0].a1 == "CB"
    assert vs[0].a2 == "CB"

    # one variable, four positions
    vs, _, _ = parse_var("1-2-3-4")
    assert isinstance(vs[0], PseudoDihedral)
    (
        vs,
        _,
        _,
    ) = parse_var("1:N-2:CA-3:C-4:N")
    assert isinstance(vs[0], Dihedral)

    # several variables
    (
        vs,
        _,
        _,
    ) = parse_var("1-2-3-4,1,2:CB-3:CB,2_Phi")
    assert len(vs) == 4
    assert isinstance(vs[0], PseudoDihedral)
    assert isinstance(vs[1], SeqEl)
    assert isinstance(vs[2], Dist)
    assert isinstance(vs[3], Phi)

    # several variables, several proteins
    vs, ss, _ = parse_var("1,2--ABCD:A,BLABLA23")
    assert ss == ["ABCD:A", "BLABLA23"]

    # several variables, several proteins and domains
    vs, ss, ds = parse_var("1,2--ABCD:A,BLABLA23::SH2,SH3")
    assert ss == ["ABCD:A", "BLABLA23"]
    assert ds == ["SH2", "SH3"]


def test_dist(simple_structure):
    v = Dist(1, 40, "CB", "CB")
    r = v.calculate(simple_structure)
    assert abs(round(r, 2) - 4.56) < EPS
    v = Dist(-10, 1000, "CB", "CB")
    with pytest.raises(FailedCalculation):
        v.calculate(simple_structure)
    r = v.calculate(simple_structure, mapping={-10: 1, 1000: 40})
    assert abs(round(r, 2) - 4.56) < EPS
