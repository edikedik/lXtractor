from io import StringIO

import pandas as pd
import pytest
from more_itertools import unzip

from lXtractor.chain import ChainSequence
from lXtractor.ext.uniprot import fetch_uniprot, UniProt
from lXtractor.util.seq import read_fasta


def test_fetch_uniprot():
    ids = ["P00523", "P12931"]
    results = fetch_uniprot(ids)
    seqs = list(read_fasta(StringIO(results)))
    assert len(seqs) == 2
    assert {x[0].split("|")[1] for x in seqs} == {"P00523", "P12931"}


@pytest.mark.parametrize(
    "inp",
    [[("P00523", True), ("P12931", True)], [("P00523", True), ("P129310", False)]],
)
@pytest.mark.parametrize(
    "callback,rtype", [(ChainSequence.from_tuple, ChainSequence), (None, None)]
)
@pytest.mark.parametrize("chunk_size", [1, 10])
def test_fetch_sequences(inp, callback, rtype, chunk_size):
    ids, is_valid = map(list, unzip(inp))
    num_valid = sum(is_valid)
    if chunk_size > len(ids) > num_valid:
        num_valid = 0
    uni = UniProt(chunk_size=chunk_size)
    res = list(uni.fetch_sequences(ids, callback=callback))
    assert len(res) == num_valid
    for elem in res:
        if callback is not None:
            assert isinstance(elem, rtype)
        else:
            assert isinstance(elem, tuple) and len(elem) == 2


@pytest.mark.parametrize(
    "inp",
    [["P00523", "P12931"]],
)
@pytest.mark.parametrize("chunk_size", [1, 10])
def test_fetch_sequences_with_dir(inp, chunk_size, tmp_path):
    uni = UniProt(chunk_size=chunk_size)
    res = list(uni.fetch_sequences(inp, tmp_path))
    assert len(res) == len(inp)
    fetched_names = {x[0].split("|")[1] for x in res}
    written_names = {p.stem for p in tmp_path.glob("*.fasta")}
    assert set(inp) == written_names
    assert set(inp) == fetched_names

    # calling again should not fetch anything since the files exist
    res = list(uni.fetch_sequences(inp, tmp_path))
    assert len(res) == 0


@pytest.mark.parametrize("ids", [["P00523", "P12931"]])
@pytest.mark.parametrize("fields", ["accession,id", None])
@pytest.mark.parametrize("chunk_size", [1, 10])
@pytest.mark.parametrize("as_df", [True, False])
def test_fetch_info(ids, fields, chunk_size, as_df):
    uni = UniProt(chunk_size=chunk_size)
    res = uni.fetch_info(ids, fields, as_df)
    if as_df:
        assert isinstance(res, pd.DataFrame)
        if fields is not None:
            assert len(res.columns) == len(fields.split(","))
    else:
        assert isinstance(res, list)
