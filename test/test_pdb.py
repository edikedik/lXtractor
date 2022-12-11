from pathlib import Path
from tempfile import TemporaryDirectory

from lXtractor.ext.pdb_ import PDB


def test_fetch():
    with TemporaryDirectory() as tmpdir:
        pdb = PDB()
        ids = ['2oiq', '3i6x']
        fetched, missed = pdb.fetch(ids, pdb_dir=Path(tmpdir), fmt='cif')
        assert len(fetched) == 2
        assert len(missed) == 0
        fetched, missed = pdb.fetch(ids, Path(tmpdir), overwrite=False)
        assert len(fetched) == len(missed) == 0


def test_get_info():
    pdb = PDB()
    res = pdb.get_info(('2src', ))
    assert len(res) == 1
    assert res.pop()['entry']['id'] == '2SRC'