import typing as t
from collections import UserDict, namedtuple
from itertools import groupby
from pathlib import Path

import pandas as pd
from more_itertools import take

from lXtractor.core.config import Sep
from lXtractor.core.exceptions import FormatError


class SizedDict(UserDict):

    def __init__(self, max_items: int, *args, **kwargs):
        self.max_items = max_items
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        diff = len(self) - self.max_items
        if diff > 0:
            for k in take(diff, iter(self.keys())):
                super().__delitem__(k)
        super().__setitem__(key, value)


def split_validate(inp: str, sep: str, parts: int):
    split = inp.split(sep)
    if len(split) != parts:
        raise FormatError(
            f'Expected {parts} "{sep}" separators, '
            f'got {len(split) - 1} in {inp}')
    return split


def col2col(df: pd.DataFrame, col_fr: str, col_to: str):
    sub = df[[col_fr, col_to]].drop_duplicates().sort_values([col_fr, col_to])
    groups = groupby(zip(sub[col_fr], sub[col_to]), key=lambda x: x[0])
    return {k: [x[1] for x in group] for k, group in groups}


def parse_protein_path(path: Path) -> t.Tuple[str, str, str]:
    uni_id, pdb = path.name.split(Sep.uni_pdb)
    pdb_id, pdb_chain = pdb.split(Sep.chain)
    return uni_id, pdb_id, pdb_chain


def is_valid_field_name(s: str) -> bool:
    try:
        namedtuple('x', [s])
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    raise RuntimeError
