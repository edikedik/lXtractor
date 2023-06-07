from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lXtractor.core import GenericStructure
from lXtractor.ext.pdb_ import PDB
from lXtractor.util import get_files

PDB_IDS = [("2src", "xxxx")]


@pytest.mark.parametrize("ids", PDB_IDS)
@pytest.mark.parametrize("fmt", ["cif"])
@pytest.mark.parametrize("dir_", [True, False])
@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("callback", [GenericStructure.read, None])
def test_fetch(ids, fmt, dir_, num_threads, callback):
    pdb = PDB(num_threads=num_threads)
    if dir:
        with TemporaryDirectory() as dir_:
            dir_ = Path(dir_)

            fetched, missed = pdb.fetch_structures(ids, dir_=dir_, fmt=fmt)

            # existing => skip
            _fetched, _missed = pdb.fetch_structures(ids, dir_=dir_, fmt=fmt)
            assert len(_fetched) == 0
            assert len(_missed) == 1
    else:
        fetched, missed = pdb.fetch_structures(
            ids, dir_=None, fmt=fmt, callback=callback
        )

    assert len(fetched) == len(missed) == 1

    inp, res = fetched.pop()

    if dir:
        assert isinstance(res, Path)
    else:
        if callback is not None:
            assert isinstance(res, GenericStructure)
        else:
            assert isinstance(res, str)


@pytest.mark.parametrize("ids", [(PDB_IDS[0][0],)])
@pytest.mark.parametrize("fmt", ["pdb", "cif", "mmtf", "pdb.gz"])
def test_fetch_fmts(ids, fmt):
    with TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        pdb = PDB()
        fetched, missed = pdb.fetch_structures(ids, dir_=dir_, fmt=fmt)
        assert len(fetched) == 1
        assert len(missed) == 0

        files = get_files(dir_)

        for inp_id in ids:
            if fmt == "mmtf":
                fmt += ".gz"
            assert f"{inp_id}.{fmt}" in files


@pytest.mark.parametrize("ids", PDB_IDS)
@pytest.mark.parametrize("use_dir", [True, False])
@pytest.mark.parametrize("service", ["entry", "pubmed"])
def test_fetch_info(ids, use_dir, service):
    pdb = PDB()
    if use_dir:
        with TemporaryDirectory() as tmp:
            fetched, missed = pdb.fetch_info(service, ids, Path(tmp))
        item_type = Path
    else:
        fetched, missed = pdb.fetch_info(service, ids, None)
        item_type = dict

    assert len(fetched) == len(missed) == 1
    assert isinstance(fetched[0][1], item_type)
    assert missed[0] == "xxxx"


def test_fetch_obsolete():
    pdb = PDB()
    obsolete = pdb.fetch_obsolete()
    assert isinstance(obsolete, dict)
    assert all(len(k) == 4 for k in obsolete)
