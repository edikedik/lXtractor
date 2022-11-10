import pytest

from lXtractor.core.exceptions import LengthMismatch, NoOverlap
from lXtractor.core.segment import Segment, resolve_overlaps


# TODO: Test inheritance and deepcopying explicitly


def test_init():
    assert len(Segment(1, 2)) == 2
    assert len(Segment(1, 1)) == 1
    s = Segment(1, 5, 'A', parent=Segment(2, 3, 'B'))
    assert s.parent.id == 'B:2-3'
    assert s.id == 'A:1-5<-B:2-3'

    with pytest.raises(ValueError):
        Segment(1, 0)

    with pytest.raises(LengthMismatch):
        Segment(0, 1, seqs={'a': [1, 2, 3]})

    with pytest.raises(LengthMismatch):
        Segment(0, 1, seqs={'a': [1, 2], 'b': [1]})


def test_iter():
    s = Segment(1, 3)
    assert list(iter(s)) == [1, 2, 3]
    s = Segment(1, 3, seqs={'a': ['one', 'two', 'three'], 'b': '123'})
    print(s.item_type._fields)
    assert list(map(tuple, iter(s))) == [(1, 'one', '1'), (2, 'two', '2'), (3, 'three', '3')]


def test_slice():
    s = Segment(1, 3, seqs={'a': [1, 2, 3], 'b': '123'})
    assert list(map(tuple, s[:1])) == [(1, 1, '1')]
    assert list(map(tuple, s[3:])) == [(3, 3, '3')]
    assert list(map(tuple, s[4:])) == []
    assert list(map(tuple, s[-100:])) == [(1, 1, '1'), (2, 2, '2'), (3, 3, '3')]
    assert tuple(s[1]) == (1, 1, '1')
    assert s['a'] == [1, 2, 3]


def test_bounds():
    s = Segment(1, 3)
    assert s.bounds(Segment(1, 3))
    assert s.bounds(Segment(1, 2))
    assert s.bounded_by(Segment(1, 3))
    assert s.bounded_by(Segment(0, 5))


def test_overlap():
    s1 = Segment(1, 5, seqs={'a': '12345'}, meta={'reason': 'none'})

    with pytest.raises(NoOverlap):
        s1 & Segment(6, 7)
    with pytest.raises(NoOverlap):
        Segment(-1, 0) & s1

    #
    so = s1.overlap(0, 3)
    assert (so.start, so.end) == (1, 3)
    assert 'a' in so._seqs
    assert 'reason' in so.meta
    assert so._seqs['a'] == '123'

    # s2 is the subset
    s2 = Segment(2, 3, seqs={'b': ['three', 'four']})
    so = s1.overlap_with(s2, handle_mode='merge')
    assert so.start == 2
    assert so.end == 3
    assert 'a' in so._seqs
    assert 'b' in so._seqs
    assert 'reason' in so.meta
    assert so._seqs['a'] == '23'
    assert so._seqs['b'] == ['three', 'four']
    so = s1.sub_by(s2)
    assert so.start == 2
    assert so.end == 3

    # partially overlapping
    s2 = Segment(5, 6, seqs={'b': '56'})
    so = s1.overlap_with(s2)
    assert len(so) == 1
    assert so._seqs['b'] == '5'

    with pytest.raises(NoOverlap):
        s1.sub_by(s2)

    with pytest.raises(NoOverlap):
        s1.sub(5, 6)


def test_resolving_overlaps():
    segments = [
        Segment(1, 3, 'x', meta={'s': 1}),
        Segment(2, 5, 'y', meta={'s': 3}),
        Segment(4, 6, 'z', meta={'s': 1}),
        Segment(8, 9, 'q', meta={'s': 1})
    ]
    filtered = list(resolve_overlaps(segments))
    assert len(filtered) == 3
    assert set(x.name for x in filtered) == {'x', 'z', 'q'}

    filtered = list(resolve_overlaps(segments, value_fn=lambda x: x.meta['s']))
    assert len(filtered) == 2
    assert set(x.name for x in filtered) == {'y', 'q'}
