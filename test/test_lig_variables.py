import pytest

import lXtractor.variables.ligand as lig_vs
from lXtractor.core.exceptions import FailedCalculation

EPS = 0.01
TEST_VARIABLES = (
    (lig_vs.LigandDist(412, "CB"), 1, 3.76),
    (lig_vs.LigandDist(412, "CB", "CAB"), 1, 5.59),
)


@pytest.mark.parametrize("v,lig_idx,res", TEST_VARIABLES)
def test_lig_vs(v, lig_idx, res, human_abl_str):
    lig = human_abl_str.ligands[lig_idx]

    if res is None:
        with pytest.raises(FailedCalculation):
            v.calculate(lig)
    else:
        res_actual = v.calculate(lig)
        if isinstance(res, str):
            assert res_actual == res
        else:
            assert abs(res - res_actual) <= EPS
