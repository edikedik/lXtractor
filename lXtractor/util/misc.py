"""
Miscellaneous utilities that couldn't be properly categorized.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import typing as t
from collections import namedtuple, abc
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from os import PathLike
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rustworkx as rx
from toolz import valmap, compose_left
from tqdm.auto import tqdm

from lXtractor.core.exceptions import FormatError

__all__ = (
    "is_valid_field_name",
    "apply",
    "is_empty",
    "col2col",
    "valgroup",
    "all_logging_disabled",
    "json_to_molgraph",
    "graph_reindex_nodes",
    "get_cpu_count"
)

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


def is_empty(x: t.Any) -> bool:
    if isinstance(x, float):
        return np.isnan(x)
    elif isinstance(x, str):
        return x == ""
    return False


def _apply_sequentially(fn, it, verbose, desc, total):
    total = total or (len(it) if isinstance(it, abc.Sized) else None)
    if verbose:
        it = tqdm(it, desc=desc, total=total)
    yield from map(fn, it)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress
    bar given as argument
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _apply_parallel_joblib(fn, it, verbose, desc, num_proc, total, **kwargs):
    assert num_proc > 1, "More than 1 CPU requested"
    if total is None and isinstance(it, abc.Sized):
        total = len(it)
    if verbose:
        with tqdm_joblib(tqdm(desc=desc, total=total)):
            with joblib.Parallel(n_jobs=num_proc, **kwargs) as executor:
                yield from executor(joblib.delayed(fn)(x) for x in it)
    else:
        with joblib.Parallel(n_jobs=num_proc) as executor:
            yield from executor(joblib.delayed(fn)(x) for x in it)

    # # Alternative version, without the context manager:
    # with joblib.Parallel(n_jobs=num_proc) as executor:
    #     # yield from executor(delayed(fn)(x) for x in it)
    #     if verbose:
    #         yield from executor(
    #             joblib.delayed(fn)(x) for x in tqdm(it, desc=desc, total=total)
    #         )
    #     else:
    #         yield from executor(joblib.delayed(fn)(x) for x in it)


def _apply_parallel(fn, it, verbose, desc, num_proc, total, **kwargs):
    total = total or (len(it) if isinstance(it, abc.Sized) else None)
    with ProcessPoolExecutor(num_proc) as executor:
        if verbose:
            yield from tqdm(executor.map(fn, it, **kwargs), desc=desc, total=total)
        else:
            yield from executor.map(fn, it)


def apply(
    fn: abc.Callable[[T], R],
    it: abc.Iterable[T],
    verbose: bool,
    desc: str,
    num_proc: int,
    total: int | None = None,
    use_joblib: bool = False,
    **kwargs,
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
    :param use_joblib: Use :class:`joblib.Parallel` for parallel application.
    :return: Passed to :meth:`ProcessPoolExecutor.map()` or
        :class:`joblib.Parallel`.
    """
    if num_proc > 1:
        f = _apply_parallel_joblib if use_joblib else _apply_parallel
        yield from f(fn, it, verbose, desc, num_proc, total, **kwargs)
    else:
        yield from _apply_sequentially(fn, it, verbose, desc, total)


def valgroup(m: abc.Mapping[str, list[str]], sep: str = ":"):
    """
    Reformat a mapping from the format::

        X => [Y{sep}Z, ...]

    To a format::

        X => [(Y, [Z, ...]), ...]

    >>> mapping = {'X': ['C:A', 'C:B', 'Y:Z']}
    >>> valgroup(mapping)
    {'X': [('X', ['A', 'B']), ('Y', ['Z'])]}

    .. hint::
        This method is useful for converting the sequence-to-structure mapping
        outputted by :class:`lXtractor.ext.sifts.SIFTS` to a format accepted by
        the :method:`lXtractor.core.chain.initializer.ChainInitializer.from_mapping`
        to initialize :class:`lXtractor.core.chain.Chain` objects

    :param m: A mapping from strings to a list of strings.
    :param sep: A separator of each mapped string in the list.
    :return: A reformatted mapping.
    """

    def group(xs):
        for k, vs in groupby(sorted(xs), key=lambda x: x.split(":")[0]):
            try:
                yield k, [":".join(x.split(":")[1:]) for x in vs]
            except IndexError as e:
                raise FormatError(
                    f"Separator {sep} is absent in item with key {k}"
                ) from e

    return valmap(compose_left(group, list), m)


@contextlib.contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    The function was borrowed from
    `this gist <https://gist.github.com/simon-weber/7853144>`_

    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def json_to_molgraph(inp: dict | PathLike) -> rx.PyGraph:
    """
    Converts a JSON-formatted molecular graph into a PyGraph object.
    This graph is a dictionary with two keys: "num_nodes" and "edges".
    The former indicates the number of atoms in a structure, whereas the latter
    is a list of edge tuples.

    :param inp: A dictionary or a path to a JSON file produced using
        `rustworkx.node_link_json`.
    :return: A graph with nodes and edges initialized in order given in `inp`.
        Any associated data will be omitted.
    """
    if not isinstance(inp, dict):
        with open(inp) as f:
            inp: dict = json.load(f)
    g = rx.PyGraph()
    g.add_nodes_from(range(inp["num_nodes"]))
    g.add_edges_from_no_data(list(map(tuple, inp["edges"])))
    return g


@t.overload
def molgraph_to_json(g: rx.PyGraph, path: PathLike) -> Path:
    ...


@t.overload
def molgraph_to_json(g: rx.PyGraph, path: None) -> dict:
    ...


def molgraph_to_json(g: rx.PyGraph, path: PathLike | None = None) -> dict | Path:
    d = {"num_nodes": len(g), "edges": list(g.edge_list())}
    if path is not None:
        with open(path, "w") as f:
            json.dump(d, f)
        return Path(path)
    return d


def graph_reindex_nodes(g: rx.PyGraph) -> rx.PyGraph:
    """
    Reindex the graph nodes so that node data equals to node indices.

    :param g: An arbitrary PyGraph.
    :return: A PyGraph of the same size and having the same edges but with
        reindexed nodes.
    """
    gg = rx.PyGraph()
    gg.add_nodes_from(g.node_indexes())
    gg.add_edges_from_no_data(g.edge_list())
    return gg


def get_cpu_count(c: int):
    mc = os.cpu_count()
    if c == -1:
        return mc
    elif 0 < c <= mc:
        return c
    else:
        raise ValueError(
            f"Invalid requested CPU count {c}. Must be between 1 and the maximum "
            f"number of available cores {mc}."
        )


if __name__ == "__main__":
    raise RuntimeError
