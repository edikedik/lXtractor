from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lXtractor.ext.pdb_ import PDB

PDB_IDS = [('2src', 'xxxx')]


def test_fetch():
    with TemporaryDirectory() as tmpdir:
        pdb = PDB()
        ids = ['2oiq', '3i6x']

        # Not fetched => save to dir
        fetched, missed = pdb.fetch_structures(ids, dir_=Path(tmpdir))

        assert len(fetched) == 2 and len(missed) == 0

        # Already fetched => skip
        fetched, missed = pdb.fetch_structures(ids, dir_=Path(tmpdir), overwrite=False)
        assert len(fetched) == len(missed) == 0

        # No dir => results are strings
        ids.append('xxxx')
        fetched, missed = pdb.fetch_structures(ids, dir_=None)
        assert len(missed) == 1 and len(fetched) == 2
        (args1, res1), (args2, res2) = fetched
        assert {args1, args2} == {('2oiq', 'cif'), ('3i6x', 'cif')}
        assert isinstance(res1, str) and isinstance(res2, str)
        assert missed.pop() == ('xxxx', 'cif')

        # Fetch in parallel
        pdb = PDB(num_threads=3)
        fetched, missed = pdb.fetch_structures(ids, dir_=None)
        assert len(missed) == 1 and len(fetched) == 2


@pytest.mark.parametrize('ids', PDB_IDS)
@pytest.mark.parametrize('use_dir', [True, False])
@pytest.mark.parametrize('service', ['entry', 'pubmed'])
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
    assert missed[0] == 'xxxx'
