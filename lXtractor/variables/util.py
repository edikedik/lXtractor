from __future__ import annotations

from collections import abc

import numpy as np
from biotite import structure as bst

from toolz import curry

from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.base import T, V, MappingT, AggFn, AggFns


@curry
def _try_map(p: T, m: abc.Mapping[T, V] | None) -> V | T:
    try:
        if m is not None:
            return m[p]
        return p
    except KeyError as e:
        raise FailedCalculation(f"Missing {p} in mapping") from e


@curry
def residue_mask(pos: int, a: bst.AtomArray, m: MappingT | None = None) -> np.ndarray:
    """
    Get a boolean mask for specific position in an atom array.

    :param pos: Position.
    :param a: Atom array.
    :param m: Optional mapping to map `pos` onto structure's numbering.
    :return: A boolean mask ``True`` for indicated `pos`.
    """
    pos = _try_map(pos, m)

    mask: np.ndarray = np.equal(a.res_id, pos)

    residue_atoms = a[mask]
    if residue_atoms.array_length() == 0:
        raise FailedCalculation(f"Missing position {pos}")

    num_starts = bst.get_residue_starts(residue_atoms)
    if len(num_starts) > 1:
        raise FailedCalculation(
            f"Position {pos} points to {len(num_starts)}>1 residues"
        )

    return mask


def atom_mask(
    pos: int, atom_name: str, a: bst.AtomArray, m: MappingT | None
) -> np.ndarray:
    """
    Get a boolean mask for certain atom at some position.

    :param pos: Position number.
    :param atom_name: The name of the atom within `pos`.
    :param a: Atom array.
    :param m: Optional mapping to map `pos` onto structure's numbering.
    :return: A boolean mask with a single ``True`` pointing to the desired atom.
    """
    r_mask = residue_mask(pos, a, m)
    a_mask = r_mask & (a.atom_name == atom_name)
    if a[a_mask].array_length() == 0:
        raise FailedCalculation(
            f"Missing atom {atom_name} at position {pos} (unmapped)"
        )
    if a[a_mask].array_length() > 1:
        raise FailedCalculation(
            f"More than one atom {atom_name} at position {pos} (unmapped)"
        )
    return a_mask


@curry
def _get_residue(
    pos: int, array: bst.AtomArray, mapping: MappingT | None = None
) -> bst.AtomArray:
    mask = residue_mask(pos, array, mapping)
    return array[mask]


@curry
def _get_atom(array: bst.AtomArray, name: str) -> bst.Atom:
    atom = array[array.atom_name == name]

    size = atom.array_length()

    if not size:
        raise FailedCalculation(f"Missing atom {name}")

    if size > 1:
        raise FailedCalculation(f"Non-unique atom with name {name}")

    return atom[0]


@curry
def _get_coord(
    residue: bst.AtomArray, atom_name: str | None, agg_fn: AggFn = AggFns["mean"]
) -> np.ndarray:
    if atom_name is None:
        coord = agg_fn(residue.coord, axis=0)
    else:
        coord = _get_atom(residue, atom_name).coord
    assert isinstance(coord, np.ndarray) and coord.shape == (3,)
    return coord


def _agg_dist(r1: bst.AtomArray, r2: bst.AtomArray, agg_fn: AggFn) -> float:
    """
    Calculate the aggregated distance between two residues
    """
    res = agg_fn(np.linalg.norm(r1.coord[:, np.newaxis] - r2.coord, axis=2))
    if not isinstance(res, (float, np.floating)):
        raise TypeError(
            f"Expected float-type return when aggregating distances; "
            f"Got res {res} of type {type(res)}"
        )

    return res


if __name__ == "__main__":
    raise RuntimeError
