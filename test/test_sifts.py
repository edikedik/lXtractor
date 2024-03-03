from collections import abc
from pathlib import Path

import pandas as pd
import pytest

from lXtractor.ext import SIFTS, PDB


@pytest.fixture()
def sifts() -> SIFTS:
    return SIFTS(load_id_mapping=True, load_segments=False)


def test_fetch(sifts):
    path = sifts.fetch(overwrite=True)
    assert isinstance(path, Path) and path.exists()


def test_parse(sifts):
    df, m = sifts.parse(rm_raw=False, overwrite=True)
    assert isinstance(df, pd.DataFrame) and isinstance(m, abc.Mapping)


@pytest.mark.parametrize(
    "up_ids,pdb_ids,base_path,pdb_method",
    [
        (["P12931"], None, None, None),
        ({"P12931": ("id", "fakeseq")}, None, Path(__file__), None),
        ({"P12931": ("id", "fakeseq")}, ["2SRC:A"], Path(__file__), "X-ray"),
    ],
)
def test_prepare_mapping(sifts, up_ids, pdb_ids, base_path, pdb_method):
    m = sifts.prepare_mapping(
        up_ids,
        pdb_ids,
        pdb_method=pdb_method,
        pdb_base=base_path,
        pdb_method_filter_kwargs=dict(pdb=PDB(num_threads=3), dir_=None),
    )
    assert len(m) == len(up_ids)
    pdb_ids = next(iter(m.values()))
    orig_key = list(up_ids).pop()
    key = next(iter(m.keys()))
    assert all(isinstance(x[1], list) for x in pdb_ids)

    if base_path:
        assert all(isinstance(x[0], Path) for x in pdb_ids)

    if isinstance(up_ids, abc.Mapping):
        assert isinstance(key, tuple)

    assert len(pdb_ids) > 0
    assert len(pdb_ids) < len(sifts[orig_key])
