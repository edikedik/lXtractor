import pandas as pd
import pytest

from lXtractor.ext import fetch_orthologs_info


@pytest.mark.parametrize('inp_ids,match_expected',
                         [(['P12931', 'P23458'], True), (['Nonsense'], False)])
def test_fetch_orthologs_info(inp_ids, match_expected):
    res = fetch_orthologs_info(inp_ids)
    if match_expected:
        assert isinstance(res, pd.DataFrame)
        assert set(inp_ids) == set(res['id'])
    else:
        assert isinstance(res, dict)
