"""
Miscellaneous utilities that couldn't be properly categorized.
"""
from collections import UserDict, namedtuple
from itertools import groupby

import pandas as pd
from more_itertools import take

from lXtractor.core.exceptions import FormatError


class SizedDict(UserDict):
    """
    Dict with limited number of keys. In case of exceeding the max number
    of elements during the set item operation, removes the first elements
    to abide the size constraints.
    """

    def __init__(self, max_items: int, *args, **kwargs):
        self.max_items = max_items
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        diff = len(self) - self.max_items
        if diff > 0:
            for k in take(diff, iter(self.keys())):
                super().__delitem__(k)
        super().__setitem__(key, value)


def split_validate(inp: str, sep: str, parts: int) -> list[str]:
    """
    :param inp: Arbitrary string.
    :param sep: Separator.
    :param parts: How many parts to expect.
    :return: Split data iff the number of parts exactly matches
        the expected one.
    :raise FormatError: If the number of parts doesn't match the expected one.
    """
    split = inp.split(sep)
    if len(split) != parts:
        raise FormatError(
            f'Expected {parts} "{sep}" separators, ' f'got {len(split) - 1} in {inp}'
        )
    return split


def col2col(df: pd.DataFrame, col_fr: str, col_to: str):
    """
    :param df: Some DataFrame.
    :param col_fr: A column name to map from.
    :param col_to: A column name to map to.
    :return: Mapping between values of a pair of columns.
    """
    sub = df[[col_fr, col_to]].drop_duplicates().sort_values([col_fr, col_to])
    groups = groupby(zip(sub[col_fr], sub[col_to]), key=lambda x: x[0])
    return {k: [x[1] for x in group] for k, group in groups}


def is_valid_field_name(s: str) -> bool:
    """
    :param s: Some string.
    :return: ``True`` if ``s` is a valid field name for ``__getattr__ ``
        operations else ``False``.
    """
    try:
        namedtuple('x', [s])
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    raise RuntimeError
