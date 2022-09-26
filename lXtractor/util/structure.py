from __future__ import annotations

import logging
from pathlib import Path

import biotite.structure as bst
import fastpdb
import numpy as np

LOGGER = logging.getLogger(__name__)


def read_fast_pdb(path: Path, model: int = 1) -> bst.AtomArray:
    file = fastpdb.PDBFile.read(str(path))
    return file.get_structure(model=model)


def calculate_dihedral(
        atom1: np.ndarray, atom2: np.ndarray,
        atom3: np.ndarray, atom4: np.ndarray
) -> float:
    """
    Calculate angle between planes formed by
    [a1, a2, atom3] and [a2, atom3, atom4].

    Each atom is an array of shape (3, ) with XYZ coordinates.

    Calculation method inspired by
    https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    """

    v1, v2, v3 = map(
        lambda v: v / np.linalg.norm(v),
        [atom2 - atom1, atom3 - atom2, atom4 - atom3])

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    x = n1.dot(n2)
    y = np.cross(n1, n2).dot(v2)

    return np.arctan2(y, x)


def check_bond_continuity(
        array: bst.AtomArray | np.ndarray,
        min_len: float = 1.0, max_len: float = 2.0
):
    if isinstance(array, bst.AtomArray):
        array = array.coord

    dist = np.linalg.norm(np.diff(array, axis=0), axis=1)
    mask = (dist < min_len) | (dist > max_len)

    return np.where(mask)[0] + 1


def check_backbone_bond_continuity(
        array: bst.AtomArray,
        min_len: float = 1.0, max_len: float = 2.0
):
    backbone_idx = np.where(bst.filter_backbone(array))[0]
    discont_idx = check_bond_continuity(array, min_len, max_len)

    return discont_idx[np.isin(discont_idx, backbone_idx)]


if __name__ == '__main__':
    raise RuntimeError
