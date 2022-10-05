from pathlib import Path

from lXtractor.ext.pdb import PDB


def test_fetch():
    pdb = PDB(pdb_dir=Path('test/data'), fmt='cif')
    ids = ['2oiq', '3i6x']
    fetched, missed = pdb.fetch(ids)
    assert len(fetched) == len(missed) == 0
    fetched, missed = pdb.fetch(ids, overwrite=True)
    assert len(fetched) == 2
    assert len(missed) == 0
