"""
Miscellaneous utilities that couldn't be properly categorized.
"""
from __future__ import annotations

import typing as t
from collections import namedtuple, abc
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby

import pandas as pd
from tqdm.auto import tqdm

from lXtractor.core.exceptions import FormatError

__all__ = ("is_valid_field_name", "apply")

T = t.TypeVar("T")
R = t.TypeVar("R")


# class SizedDict(UserDict):
#     """
#     Dict with limited number of keys. In case of exceeding the max number
#     of elements during the set item operation, removes the first elements
#     to abide the size constraints.
#     """
#
#     def __init__(self, max_items: int, *args, **kwargs):
#         self.max_items = max_items
#         super().__init__(*args, **kwargs)
#
#     def __setitem__(self, key, value):
#         diff = len(self) - self.max_items
#         if diff > 0:
#             for k in take(diff, iter(self.keys())):
#                 super().__delitem__(k)
#         super().__setitem__(key, value)


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
            f'Expected {parts} "{sep}" separators, ' f"got {len(split) - 1} in {inp}"
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
    groups = groupby(
        zip(sub[col_fr], sub[col_to]), key=lambda x: x[0]  # type: ignore  # No Any
    )
    return {k: [x[1] for x in group] for k, group in groups}


def is_valid_field_name(s: str) -> bool:
    """
    :param s: Some string.
    :return: ``True`` if ``s` is a valid field name for ``__getattr__ ``
        operations else ``False``.
    """
    try:
        namedtuple("x", [s])
        return True
    except ValueError:
        return False


def _apply_sequentially(fn, it, verbose, desc, total):
    total = total or (len(it) if isinstance(it, abc.Sized) else None)
    if verbose:
        it = tqdm(it, desc=desc, total=total)
    yield from map(fn, it)


def _apply_parallel(fn, it, verbose, desc, num_proc, total, **kwargs):
    assert num_proc > 1, "More than 1 CPU requested"
    total = total or (len(it) if isinstance(it, abc.Sized) else None)
    with ProcessPoolExecutor(num_proc) as executor:
        if verbose:
            yield from tqdm(
                executor.map(fn, it, **kwargs),
                desc=desc, total=total
            )
        else:
            yield from executor.map(fn, it)


def apply(
    fn: abc.Callable[[T], R],
    it: abc.Iterable[T],
    verbose: bool,
    desc: str,
    num_proc: int,
    total: int | None = None,
    **kwargs
) -> abc.Iterator[R]:
    """
    :param fn: A one-argument function.
    :param it: An iterable over some objects.
    :param verbose: Display progress bar.
    :param desc: Progress bar description.
    :param num_proc: The number of processes to use. Anything below ``1``
        indicates sequential processing. Otherwise, will apply ``fn``
        in parallel using ``ProcessPoolExecutor``.
    :param total: The total number of elements. Used for the progress bar.
    :return: Passed to :meth:`ProcessPoolExecutor.map()`.
    """
    if num_proc > 1:
        yield from _apply_parallel(fn, it, verbose, desc, num_proc, total, **kwargs)
    else:
        yield from _apply_sequentially(fn, it, verbose, desc, total)


if __name__ == "__main__":
    raise RuntimeError
