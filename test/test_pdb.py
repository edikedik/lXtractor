from pathlib import Path
from tempfile import TemporaryDirectory

from lXtractor.ext.pdb_ import PDB


def test_fetch():
    with TemporaryDirectory() as tmpdir:
        pdb = PDB()
        ids = ['2oiq', '3i6x']
        fetched, missed = pdb.fetch_files(ids, pdb_dir=Path(tmpdir), fmt='cif')
        assert len(fetched) == 2
        assert len(missed) == 0
        fetched, missed = pdb.fetch_files(ids, Path(tmpdir), overwrite=False)
        assert len(fetched) == len(missed) == 0


def test_get_info():
    pdb = PDB()
    fetched, remaining = pdb.get_info(pdb.url_getters['entry'], [('2src', )])
    assert len(remaining) == 0 and len(fetched) == 1
    args, results = fetched.pop()
    assert args == ('2src', )
    assert results['entry']['id'] == '2SRC'
