import pytest

from lXtractor.ext.alphafold import AlphaFold


@pytest.mark.parametrize(
    "ids,fmt,output",
    [
        (["P12931", "P00523"], "cif", ({("P12931", "cif"), ("P00523", "cif")}, set())),
        (["XXX", "P12931"], "pdb", ({("P12931", "pdb")}, {("XXX", "pdb")})),
    ],
)
def test_fetching_structures(ids, fmt, output):
    af = AlphaFold(max_trials=2)
    fetched, remaining = af.fetch_structures(ids, fmt=fmt, dir_=None)
    ids_fetched = {x[0] for x in fetched}
    ids_remaining = set(remaining)
    assert (ids_fetched, ids_remaining) == output


@pytest.mark.parametrize(
    "ids,output",
    [
        (["P12931", "P00523"], ({"P12931", "P00523"}, set())),
        (["XXX", "P12931"], ({"P12931"}, {"XXX"})),
    ],
)
def test_fetching_pae(ids, output):
    af = AlphaFold(max_trials=2)
    fetched, remaining = af.fetch_info("pae", ids, dir_=None)
    ids_fetched = {x[0] for x in fetched}
    ids_remaining = set(remaining)
    assert (ids_fetched, ids_remaining) == output

    if fetched:
        _, info = fetched.pop()
        assert isinstance(info, (dict, list))
