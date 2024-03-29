import pytest

import lXtractor.variables.structural as str_vs
from lXtractor.core.exceptions import FailedCalculation

EPS = 0.01
TEST_VARIABLES = (
    (str_vs.AggDist(412, 449), 3.66),
    (str_vs.AggDist(412, 449, "max"), 11.69),
    (str_vs.Dist(412, 449, "CB", "OH"), 3.66),
    (str_vs.Dist(412, 449, "CB", "CE1"), 3.91),
    (str_vs.PseudoDihedral(414, 413, 412, 411), 0.24),
    (str_vs.Chi1(413), -1.02),
    (str_vs.Chi2(388), 2.67),
    (str_vs.Psi(413), 0.12),
    (str_vs.Phi(413), -2.16),
    (str_vs.Omega(413), 3.08),
    (str_vs.ClosestLigandContactsCount(413, "OH"), 1),
    (str_vs.ClosestLigandContactsCount(413, "CB"), 0),
    (str_vs.ClosestLigandContactsCount(413), 6),
    (str_vs.ClosestLigandNames(413, "OH"), "66K"),
    (str_vs.ClosestLigandNames(413, "CB"), ""),
    (str_vs.ClosestLigandNames(413), "66K"),
    (str_vs.ClosestLigandDist(413), 3.827),
    (str_vs.ClosestLigandDist(413, "OH"), 3.827),
    (str_vs.ClosestLigandDist(413, agg_res="max"), 7.915),
    (str_vs.ClosestLigandDist(413, agg_lig="max", agg_res="max"), 23.135),
)


@pytest.mark.parametrize("v,res", TEST_VARIABLES)
def test_variables(v, res, human_abl_str):
    if res is None:
        with pytest.raises(FailedCalculation):
            v.calculate(human_abl_str)
    else:
        res_actual = v.calculate(human_abl_str)
        if isinstance(res, str):
            assert res_actual == res
        else:
            assert abs(res - res_actual) <= EPS


def test_contacts(human_abl_str):
    # All residue contacting any 225 atoms
    v = str_vs.Contacts(225)
    res = v.calculate(human_abl_str)
    assert res == "226,227,228,229,231,717,848,856"

    # Only mapped residues
    v = str_vs.Contacts(1)
    res = v.calculate(human_abl_str, {1: 225, 2: 226, 3: 227})
    assert res == "2,3"

    # No residues -- empty string
    res = v.calculate(human_abl_str, {1: 225, 10: 500})
    assert res == ""
