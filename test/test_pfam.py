from pathlib import Path

import pandas as pd
import pytest

from lXtractor.ext.hmm import Pfam
from lXtractor.util import get_dirs
from test.common import DATA


@pytest.fixture(scope="module")
def small_pfam(tmp_path_factory) -> Pfam:
    return Pfam(DATA / "Pfam")


@pytest.mark.skip(reason="Takes too long")
def test_fetch():
    res = Pfam().fetch()
    assert len(res) == 2
    assert isinstance(res[0], Path)
    assert isinstance(res[1], Path)


@pytest.mark.parametrize("accessions", [("PF10417",), ("Missing",)])
@pytest.mark.parametrize("categories", [("Domain", "Family"), ("Missing",)])
def test_parse_read_clean(small_pfam, accessions, categories):
    # Parse
    res = small_pfam.parse(dump=True, rm_raw=False)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    res_sub = res[res["Accession"].isin(accessions) & res["Category"].isin(categories)]

    # Read
    res_read = small_pfam.read(accessions=accessions, categories=categories)
    assert isinstance(res_read, pd.DataFrame)
    assert set(res_sub.Accession) == set(res_read.Accession)

    # Clean
    small_pfam.clean(raw=False, parsed=True)
    dirs = get_dirs(small_pfam.path)
    assert "parsed" not in dirs
