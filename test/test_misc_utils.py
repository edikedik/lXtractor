import pytest

from lXtractor.util.misc import valgroup


@pytest.mark.parametrize(
    "inp,exp",
    [
        ({}, {}),
        ({"X": ["A:B", "A:C"]}, {"X": [("A", ["B", "C"])]}),
        ({"X": ["A:B:B", "A:C"]}, {"X": [("A", ["B:B", "C"])]}),
        ({"X": ["A-B", "A-C"]}, {'X': [('A-B', ['']), ('A-C', [''])]}),
    ],
)
def test_valgroup(inp, exp):
    assert valgroup(inp) == exp
