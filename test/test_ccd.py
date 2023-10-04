from pathlib import Path

import pandas as pd
import pytest

from lXtractor.ext import CCD
from lXtractor.ext.ccd import Field


@pytest.fixture
def ccd_data() -> Path:
    return Path("./data/CCD/components.cif.gz")


@pytest.mark.skip("Takes too long")
def test_fetch():
    res = CCD(read_entries=False).fetch(overwrite=False)
    assert isinstance(res, Path)


def test_parse_dump_load(ccd_data, tmp_path):
    resource_path = tmp_path / "CCD.msgpack"
    ccd = CCD(resource_path=resource_path, read_entries=False)
    res = ccd.parse(ccd_data, store_to_resources=True, rm_raw=False)
    assert isinstance(res, dict)
    assert resource_path.exists() and resource_path.is_file()
    res_read = ccd.read()
    assert len(res) == len(res_read)

    # access some data
    entry = ccd["001"]
    data = entry["_chem_comp"]
    print(data)
    assert isinstance(data, Field)
    df = data.as_df()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns)[:3] == ['id', 'name', 'type']