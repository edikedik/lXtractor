from lXtractor.collection import Collection
import pytest


GET_TABLE_NAMES = """SELECT name FROM sqlite_master WHERE type='table';"""
TABLE_NAMES = (
    "chain_types",
    "var_types",
    "var_rtypes",
    "chains",
    "parents",
    "variables",
    "sqlite_sequence"
)


@pytest.mark.parametrize("loc", [":memory:"])
def test_setup(loc):
    col = Collection(loc)
    with col._db as cur:
        names = cur.execute(GET_TABLE_NAMES)
        assert set(x[0] for x in names) == set(TABLE_NAMES)
