from pathlib import Path
from tempfile import TemporaryDirectory

from lXtractor.ext.pdb_ import PDB


def test_fetch():
    with TemporaryDirectory() as tmpdir:
        pdb = PDB()
        ids = ['2oiq', '3i6x']

        # Not fetched => save to dir
        fetched, missed = pdb.fetch_structures(ids, dir_=Path(tmpdir), fmt='cif')
        assert len(fetched) == 2 and len(missed) == 0

        # Already fetched => skip
        fetched, missed = pdb.fetch_structures(ids, dir_=Path(tmpdir), overwrite=False)
        assert len(fetched) == len(missed) == 0

        # No dir => results are strings
        ids.append('xxxx')
        fetched, missed = pdb.fetch_structures(ids, dir_=None)
        assert len(missed) == 1 and len(fetched) == 2
        (id1, res1), (id2, res2) = fetched
        assert {id1, id2} == {'2oiq', '3i6x'}
        assert isinstance(res1, str) and isinstance(res2, str)
        assert missed.pop() == 'xxxx'

        # Fetch in parallel
        pdb = PDB(num_threads=3)
        fetched, missed = pdb.fetch_structures(ids, dir_=None)
        assert len(missed) == 1 and len(fetched) == 2


def test_get_info():
    pdb = PDB()
    fetched, remaining = pdb.fetch_info('entry', [('2src',)], dir_=None)
    assert len(remaining) == 0 and len(fetched) == 1
    args, results = fetched.pop()
    assert args == ('2src',)
    assert results['entry']['id'] == '2SRC'

    fetched, remaining = pdb.fetch_info('entry', [('xxxx',)], dir_=None)
    assert len(remaining) == 1 and len(fetched) == 0

    # Parallel
    pdb = PDB(num_threads=3)
    fetched, remaining = pdb.fetch_info(
        'entry', [('2src',), ('2oiq',), ('xxxx',)], dir_=None)
    assert len(remaining) == 1 and len(fetched) == 2
