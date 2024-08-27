import shutil

import pandas as pd
import pytest

from lXtractor.ext.dssp import dssp_run, DSSP_COLUMNS
from lXtractor.util import load_structure
from test.common import ALL_STRUCTURES


@pytest.mark.skipif(shutil.which("dssp") is None, reason="Missing dssp")
@pytest.mark.parametrize("str_path", ALL_STRUCTURES)
@pytest.mark.parametrize("annotate", [True, False])
def test_dssp(str_path, annotate):
    a = load_structure(str_path)
    res = dssp_run(a, set_ss_annotation=annotate)
    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == list(DSSP_COLUMNS)
    if annotate:
        assert {"ss3", "ss8"}.issubset(set(a.get_annotation_categories()))
