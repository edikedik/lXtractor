from pathlib import Path

import biotite.structure as bst
import pandas as pd
import pytest

from lXtractor.ext import CCD
from lXtractor.ext.ccd import Field


DATA = Path(__file__).parent / "data"


@pytest.fixture
def ccd_data() -> Path:
    return DATA / "CCD" / "components.cif.gz"


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
    assert isinstance(data, Field)
    df = data.as_df()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns)[:3] == ["id", "name", "type"]

    # create an atom array
    a = ccd["001"]["_chem_comp_atom"].as_atom_array()
    assert isinstance(a, bst.AtomArray)
    a = ccd["001"]["_chem_comp_atom"].as_atom_array(ideal_coord=True)
    assert isinstance(a, bst.AtomArray)
    with pytest.raises(KeyError):
        ccd["001"]["_chem_comp"].as_atom_array()
