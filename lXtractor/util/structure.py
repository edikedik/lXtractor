from __future__ import annotations

import logging
from collections import abc
from itertools import repeat, starmap

import biotite.structure as bst
import numpy as np
from more_itertools import zip_equal, unzip

from lXtractor.core.exceptions import LengthMismatch

LOGGER = logging.getLogger(__name__)

_BASIC_COMPARISON_ATOMS = {'N', 'CA', 'C', 'CB'}


# def read_fast_pdb(path: Path, model: int = 1) -> bst.AtomArray:
#     file = fastpdb.PDBFile.read(str(path))
#     return file.get_structure(model=model)


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


def filter_selection(
        array: bst.AtomArray, res_id: abc.Sequence[int] | None,
        atom_names: abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None = None
) -> np.ndarray:

    if res_id is None:
        res_id, _ = bst.get_residues(array)

    if atom_names is None:
        staged = zip(res_id, repeat(None))
    else:
        if isinstance(atom_names, abc.Sequence):
            atom_names = repeat(atom_names, len(res_id))

        staged = zip(res_id, atom_names, strict=True)

    mask = np.zeros_like(array, bool)

    for r_id, a_names in staged:
        mask_local = array.res_id == r_id
        if a_names is not None:
            mask_local &= np.isin(array.atom_name, a_names)
        mask |= mask_local
    return mask


def filter_to_common_atoms(
        a1: bst.AtomArray, a2: bst.AtomArray,
        allow_residue_mismatch: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter to atoms common between residues of atom arrays a1 and a2.

    :param a1: Arbitrary atom array.
    :param a2: Arbitrary atom array.
    :param allow_residue_mismatch: If ``True``, when residue names mismatch, the common atoms
        are derived from the intersection ``a1.atoms & a2.atoms & {"C", "N", "CA", "CB"}``.
    :return: A pair of masks for a1 and a2, ``True`` if the atoms match.
    :raises ValueError: (1) If `a1` and `a2` have different number of residues.
        (2) If the selection for some residue produces different number of atoms.

    .. note::
        If residues match, :func:`biotite.filter_intersection` is used.

    """

    def preprocess_array(a: bst.AtomArray):
        num_res = bst.get_residue_count(a)
        r_it = bst.residue_iter(a)
        return num_res, r_it

    def process_pair(r1: bst.AtomArray, r2: bst.AtomArray) -> tuple[np.ndarray, np.ndarray]:
        r1_name, r2_name = r1.res_name[0], r2.res_name[0]
        if r1_name != r2_name:
            if not allow_residue_mismatch:
                raise ValueError(
                    f'Residue names must match. Got {r1_name} from the first array and {r2_name} '
                    f'from the second one. Use `allow_residue_mismatch` to allow name mismatches.'
                )
            atom_names = set(r1.atom_name) & set(r2.atom_name) & _BASIC_COMPARISON_ATOMS
            m1, m2 = map(lambda r: np.isin(r.atom_name, list(atom_names)), [r1, r2])
            if m1.sum() != m2.sum():
                raise ValueError(
                    f'Obtained different sets of atoms {atom_names}. '
                    f'Residue 1: {r1[m1]}. Residue 2: {r2[m2]}'
                )
        else:
            m1 = bst.filter_intersection(r1, r2)
            m2 = bst.filter_intersection(r2, r1)

        return m1, m2

    (a1_l, a1_it), (a2_l, a2_it) = map(preprocess_array, [a1, a2])

    if a1_l != a2_l:
        raise LengthMismatch(
            'The number of residues must match between structures. '
            f'Got {a1_l} for `a1` and {a2_l} for `a2`.'
        )

    mask1, mask2 = map(
        lambda x: np.concatenate(list(x)),
        unzip(starmap(process_pair, zip(a1_it, a2_it, strict=True)))
    )

    return mask1, mask2


if __name__ == '__main__':
    raise RuntimeError
