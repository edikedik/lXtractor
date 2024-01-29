from copy import deepcopy
from itertools import product

import pytest

from lXtractor.core.exceptions import LengthMismatch, NoOverlap
from lXtractor.core.segment import Segment, resolve_overlaps

# TODO: Test inheritance and deepcopying explicitly


Segments = (
    Segment(0, 0, 'S'),
    Segment(4, 7, 'S47', meta={'codename': '47'}),
    Segment(4, 7, 'S47', meta={'codename': '47'}, seqs={'A': 'ABCD'}),
    Segment(1, 5, 'S5', seqs={'S': list(range(5)), 'A': 'ABCDE'}),
    Segment(1, 1, 'S1', seqs={'A': ['A']}),
)


def test_init():
    assert len(Segment(1, 2)) == 2
    assert len(Segment(1, 1)) == 1
    s = Segment(1, 5, 'A', parent=Segment(2, 3, 'B'))
    assert s.parent.id == 'B|2-3'
    assert s.id == 'A|1-5<-(B|2-3)'

    with pytest.raises(ValueError):
        Segment(0, 1)
    with pytest.raises(ValueError):
        Segment(1, 0)
    with pytest.raises(ValueError):
        Segment(2, 1)

    with pytest.raises(LengthMismatch):
        Segment(1, 2, seqs={'a': [1, 2, 3]})

    with pytest.raises(LengthMismatch):
        Segment(1, 2, seqs={'a': [1, 2], 'b': [1]})

    s = Segment(0, 0)
    assert len(s) == 0


def test_iter():
    s = Segment(1, 3)
    assert list(map(tuple, iter(s))) == [(1,), (2,), (3,)]
    s = Segment(1, 3, seqs={'a': ['one', 'two', 'three'], 'b': '123'})
    assert list(map(tuple, iter(s))) == [
        (1, 'one', '1'),
        (2, 'two', '2'),
        (3, 'three', '3'),
    ]


@pytest.mark.parametrize('s', Segments)
def test_setters(s):
    if s.is_empty:
        for x in [-1, 0, 1]:
            with pytest.raises(IndexError):
                s.start = x
            with pytest.raises(IndexError):
                s.end = x
    else:
        s_new = s >> 1
        assert s_new.start == (s.start + 1)
        assert s_new.end == (s.end + 1)
        s_new = s_new << 1
        assert s_new.start == s.start and s_new.end == s.end
        assert not (s.meta or s_new.meta) or (s.meta != s_new.meta)
        domain = list(range(s.start, s.end + 1))
        s_new = deepcopy(s)
        for start in domain:
            s_new.start = start
            assert len(s_new) == s.end - start + 1
        for end in domain:
            s_new = deepcopy(s)
            s_new.end = end
            assert len(s_new) == end - s.start + 1


@pytest.mark.parametrize('s', Segments)
def test_slice(s):
    if s.is_empty:
        domain = [None, 0, 1]
        for start, stop, step in product(domain, domain, domain):
            with pytest.raises(IndexError):
                _ = s[slice(start, stop, step)]
            if start is not None:
                with pytest.raises(IndexError):
                    _ = s[start]
    else:
        for s_new in [s[:], s[s.start :], s[: s.end], s[s.start : s.end]]:
            assert id(s_new) != id(s) and s_new == s

        for start, stop, step in product([None, s.start], [None, s.end], [0, 1, 2]):
            with pytest.raises(IndexError):
                _ = s[slice(start, stop, step)]

        start_element = s[s.start]
        assert isinstance(start_element, tuple)

        start = s.start
        stop = min(s.end, s.start + 1)
        s_new = s[start:stop]

        if s.is_singleton:
            assert s.is_singleton
        else:
            assert len(s_new) == 2

        for name in s.seq_names:
            assert name in s_new

        assert not s_new.meta

    with pytest.raises(IndexError):
        _ = s[0]
    with pytest.raises(IndexError):
        _ = s[0:]
    with pytest.raises(IndexError):
        _ = s[:0]


def test_bounds():
    s = Segment(1, 3)
    assert s.bounds(Segment(1, 3))
    assert s.bounds(Segment(1, 2))
    assert s.bounded_by(Segment(1, 3))
    assert s.bounded_by(Segment(1, 5))


def test_overlap():
    s1 = Segment(1, 5, seqs={'a': '12345'}, meta={'reason': 'none'})
    e = Segment(0, 0)

    with pytest.raises(NoOverlap):
        s1 & Segment(6, 7)

    # basic algebra
    assert s1 & e == e
    assert e & s1 == e
    assert e & e == e
    assert s1 & s1 == s1

    assert s1.overlaps(e)
    assert e.overlaps(s1)
    assert e.bounded_by(s1)
    assert s1.bounds(e)

    assert s1.overlap_with(e) == e
    assert e.overlap_with(s1) == e
    assert e.overlap_with(e) == e
    assert s1.overlap_with(s1) == s1
    with pytest.raises(NoOverlap):
        e.sub_by(s1)
    with pytest.raises(NoOverlap):
        e.sub(1, 1)
    assert e.sub_by(e) == e
    assert e.sub(0, 0) == e

    #
    so = s1.overlap(3, 7)
    assert (so.start, so.end) == (3, 5)
    assert 'a' in so._seqs
    assert 'reason' in so.meta
    assert so._seqs['a'] == '345'

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
        Segment(8, 9, 'q', meta={'s': 1}),
    ]
    filtered = list(resolve_overlaps(segments))
    assert len(filtered) == 3
    assert set(x.name for x in filtered) == {'x', 'z', 'q'}

    filtered = list(resolve_overlaps(segments, value_fn=lambda x: x.meta['s']))
    assert len(filtered) == 2
    assert set(x.name for x in filtered) == {'y', 'q'}


def test_remove():
    s = Segment(1, 3, 'X', seqs={'A': 'AAA'})
    s.remove_seq('B')
    assert 'A' in s
    s.remove_seq('A')
    assert 'A' not in s