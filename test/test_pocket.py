import pytest

from lXtractor.core import GenericStructure
from lXtractor.core.exceptions import FormatError
from lXtractor.core.pocket import Pocket
from test.common import STRUCTURES


@pytest.mark.parametrize(
    "path,definition,is_valid,mapping,expected",
    [
        (
            STRUCTURES / "mmtf.gz" / "4hvd.mmtf.gz",
            "c:853:CB == 1 & daa:903,904,905:CA <= 6",
            True,
            None,
            {("933", True), ("PHU", False)},
        ),
        (
            STRUCTURES / "mmtf.gz" / "4hvd.mmtf.gz",
            "daa:1:any < 20 | da:2:CD < 9",
            True,
            {1: 1038, 2: 871},
            {("933", True), ("PHU", True)},
        ),
        (
            STRUCTURES / "mmtf.gz" / "4hvd.mmtf.gz",
            "1:any < 20 | da:2:CD < 9",
            False,
            {1: 1038, 2: 871},
            {("933", True), ("PHU", True)},
        ),
        (
            STRUCTURES / "mmtf.gz" / "4hvd.mmtf.gz",
            "cs:904,905,906:any >= 4",
            True,
            None,
            {("933", False), ("PHU", False)},
        ),
    ],
)
def test_pocket(path, definition, is_valid, mapping, expected):
    s = GenericStructure.read(path)
    p = Pocket(definition)
    if is_valid:
        res = {(lig.res_name, p.is_connected(lig)) for lig in s.ligands}
        assert res == expected
    else:
        with pytest.raises(FormatError):
            p.is_connected(s.ligands[0])
