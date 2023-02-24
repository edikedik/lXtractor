"""
Low-level utilities to work with structures.
"""
from __future__ import annotations

import logging
import operator as op
from collections import abc
from functools import reduce, partial
from itertools import repeat, starmap

import biotite.structure as bst
import biotite.structure.info as bstinfo
import numpy as np
from more_itertools import unzip

from lXtractor.core.base import SOLVENTS
from lXtractor.core.exceptions import LengthMismatch, MissingData
from lXtractor.util.typing import is_sequence_of

LOGGER = logging.getLogger(__name__)

_BASIC_COMPARISON_ATOMS = {'N', 'CA', 'C', 'CB'}


# def read_fast_pdb(path: Path, model: int = 1) -> bst.AtomArray:
#     file = fastpdb.PDBFile.read(str(path))
#     return file.get_structure(model=model)


def calculate_dihedral(
    atom1: np.ndarray, atom2: np.ndarray, atom3: np.ndarray, atom4: np.ndarray
) -> float:
    """
    Calculate angle between planes formed by
    [a1, a2, atom3] and [a2, atom3, atom4].

    Each atom is an array of shape (3, ) with XYZ coordinates.

    Calculation method inspired by
    https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-
    dihedral-angle-given-cartesian-coordinates
    """

    def vnorm(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)  # type: ignore  # thinks is "Any"

    v1, v2, v3 = map(vnorm, [atom2 - atom1, atom3 - atom2, atom4 - atom3])

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    x = n1.dot(n2)
    y = np.cross(n1, n2).dot(v2)

    res: float = np.arctan2(y, x)
    return res


def filter_selection(
    array: bst.AtomArray,
    res_id: abc.Sequence[int] | None,
    atom_names: abc.Sequence[abc.Sequence[str]] | abc.Sequence[str] | None = None,
) -> np.ndarray:
    """
    Filter :class:`AtomArray` by residue numbers and atom names.

    :param array: Arbitrary structure.
    :param res_id: A sequence of residue numbers.
    :param atom_names: A sequence of atom names (broadcasted to each position
        in `res_id`) or an iterable over such sequences for each position in
        `res_id`.
    :return: A binary mask that is ``True`` for filtered atoms.
    """

    if res_id is None:
        res_id, _ = bst.get_residues(array)

    _atom_names: abc.Iterable[None] | abc.Iterable[abc.Sequence[str]]

    if atom_names is None:
        _atom_names = repeat(None, len(res_id))
    elif is_sequence_of(atom_names, str):
        _atom_names = repeat(atom_names, len(res_id))
    else:
        _atom_names = atom_names

    mask = np.zeros_like(array, bool)

    for r_id, a_names in zip(res_id, _atom_names, strict=True):
        mask_local = array.res_id == r_id
        if a_names is not None:
            mask_local &= np.isin(array.atom_name, a_names)
        mask |= mask_local
    return mask


def filter_to_common_atoms(
    a1: bst.AtomArray, a2: bst.AtomArray, allow_residue_mismatch: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter to atoms common between residues of atom arrays a1 and a2.

    :param a1: Arbitrary atom array.
    :param a2: Arbitrary atom array.
    :param allow_residue_mismatch: If ``True``, when residue names mismatch,
        the common atoms are derived from the intersection
        ``a1.atoms & a2.atoms & {"C", "N", "CA", "CB"}``.
    :return: A pair of masks for a1 and a2, ``True`` for matching atoms.
    :raises ValueError:

        1. If `a1` and `a2` have different number of residues.
        2. If the selection for some residue produces different number
            of atoms.
    """

    def preprocess_array(a: bst.AtomArray) -> tuple[int, bst.AtomArray]:
        num_res = bst.get_residue_count(a)
        r_it = bst.residue_iter(a)
        return num_res, r_it

    def process_pair(
        r1: bst.AtomArray, r2: bst.AtomArray
    ) -> tuple[np.ndarray, np.ndarray]:
        r1_name, r2_name = r1.res_name[0], r2.res_name[0]
        atom_names = set(r1.atom_name) & set(r2.atom_name)

        if r1_name != r2_name:
            if not allow_residue_mismatch:
                raise ValueError(
                    f'Residue names must match. Got {r1_name} from the first array '
                    f'and {r2_name} from the second one. Use `allow_residue_mismatch` '
                    'to allow name mismatches.'
                )
            atom_names &= _BASIC_COMPARISON_ATOMS

        m1, m2 = map(lambda r: np.isin(r.atom_name, list(atom_names)), [r1, r2])
        if m1.sum() != m2.sum():
            raise ValueError(
                f'Obtained different sets of atoms {atom_names}. '
                f'Residue 1: {r1[m1]}. Residue 2: {r2[m2]}'
            )

        return m1, m2

    (a1_l, a1_it), (a2_l, a2_it) = map(preprocess_array, [a1, a2])

    if a1_l != a2_l:
        raise LengthMismatch(
            'The number of residues must match between structures. '
            f'Got {a1_l} for `a1` and {a2_l} for `a2`.'
        )

    mask1, mask2 = map(
        # numpy + mypy = confusion
        lambda x: np.concatenate(list(x)),  # type: ignore
        unzip(starmap(process_pair, zip(a1_it, a2_it, strict=True))),
    )

    return mask1, mask2


def _is_polymer(array, min_size, pol_type):
    if pol_type.startswith('p'):
        filt_fn = bst.filter_amino_acids
    elif pol_type.startswith('n'):
        filt_fn = bst.filter_nucleotides
    elif pol_type.startswith('c'):
        filt_fn = bst.filter_carbohydrates
    else:
        raise ValueError(f'Unsupported polymer type {pol_type}')

    mask = filt_fn(array)
    return bst.get_residue_count(array[mask]) >= min_size


def filter_polymer(array, min_size=2, pol_type='peptide'):
    """
    Filter for atoms that are a part of a consecutive standard macromolecular
    polymer entity.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to filter.
    min_size : int
        The minimum number of monomers.
    pol_type : str
        The polymer type, either ``"peptide"``, ``"nucleotide"``, or ``"carbohydrate"``.
        Abbreviations are supported: ``"p"``, ``"pep"``, ``"n"``, etc.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where atoms belong to
        consecutive polymer entity having at least `min_size` monomers.

    """

    split_idx = np.sort(
        np.unique(
            np.concatenate(
                [
                    bst.check_res_id_continuity(array),
                    bst.check_backbone_continuity(array),
                ]
            )
        )
    )
    # print(
    #     *[bst.check_res_id_continuity(array), bst.check_backbone_continuity(array)],
    #     sep='\n',
    # )

    check_pol = partial(_is_polymer, min_size=min_size, pol_type=pol_type)
    bool_idx = map(
        lambda a: np.full(len(a), check_pol(bst.array(a)), dtype=bool),
        np.split(array, split_idx),
    )
    return np.concatenate(list(bool_idx))


def filter_any_polymer(a: bst.AtomArray, min_size: int = 2) -> np.ndarray:
    """
    Get a mask indicating atoms being a part of a macromolecular polymer:
    peptide, nucleotide, or carbohydrate.

    :param a: Array of atoms.
    :param min_size: Min number of polymer monomers.
    :return: A boolean mask ``True`` for polymers' atoms.
    """
    return reduce(
        op.or_,
        (filter_polymer(a, min_size, pol_type) for pol_type in ['p', 'n', 'c']),
    )


def filter_solvent_extended(a: bst.AtomArray) -> np.ndarray:
    """
    Filter for solvent atoms using a curated solvent list including non-water
    molecules typically being a part of a crystallization solution.

    :param a: Atom array.
    :return: A boolean mask ``True`` for solvent atoms.
    """
    return np.isin(a.res_name, SOLVENTS)


def filter_ligand(a: bst.AtomArray) -> np.ndarray:
    """
    Filter for ligand atoms -- non-polymer and non-solvent atoms.

    ..note ::
        No contact-based verification is performed here.

    :param a: Atom array.
    :return: A boolean mask ``True`` for ligand atoms.
    """
    is_polymer = filter_any_polymer(a)
    is_solvent = filter_solvent_extended(a) | (np.vectorize(len)(a.res_name) != 3)
    return ~(is_polymer | is_solvent)


def iter_residue_masks(a: bst.AtomArray) -> abc.Generator[np.ndarray, None, None]:
    """
    Iterate over residue masks.

    :param a: Atom array.
    :return: A generator over boolean masks for each residue in `a`.
    """
    starts = bst.get_residue_starts(a, add_exclusive_stop=True)
    arange = np.arange(len(a))
    for i in range(len(starts) - 1):
        yield (arange >= starts[i]) & (arange < starts[i + 1])


def iter_canonical(a: bst.AtomArray) -> abc.Generator[bst.AtomArray | None, None, None]:
    """
    :param a: Arbitrary atom array.
    :return: Generator of canonical versions of residues in `a` or ``None``
        if no such residue found in CCD.
    """
    for name in bst.get_residues(a)[1]:
        try:
            r_can = bstinfo.residue(name)
            yield r_can
        except KeyError:
            yield None


def _exclude(a, names, elements):
    if names:
        a = a[~np.isin(a.atom_name, names)]
    if elements:
        a = a[~np.isin(a.element, elements)]
    return a


def get_missing_atoms(
    a: bst.AtomArray,
    excluding_names: abc.Sequence[str] | None = ('OXT',),
    excluding_elements: abc.Sequence[str] | None = ('H',),
) -> abc.Generator[list[str | None] | None, None, None]:
    """
    For each residue, compare with the one stored in CCD, and find missing
    atoms.

    :param a: Non-empty atom array.
    :param excluding_names: A sequence of atom names to exclude
        for calculation.
    :param excluding_elements: A sequence of element names to exclude
        for calculation.
    :return: A generator of lists of missing atoms (excluding hydrogens)
        per residue in `a` or ``None`` if not such residue was found in CCD.
    """
    if len(a) == 0:
        raise MissingData('Array is empty')
    for r_can, r_obs in zip(iter_canonical(a), bst.residue_iter(a)):
        if r_can is None:
            yield None
        else:
            r_can = _exclude(r_can, excluding_names, excluding_elements)
            m_can, _ = filter_to_common_atoms(r_can, r_obs)
            yield list(r_can[~m_can].atom_name)


def get_observed_atoms_frac(
    a: bst.AtomArray,
    excluding_names: abc.Sequence[str] | None = ('OXT',),
    excluding_elements: abc.Sequence[str] | None = ('H',),
) -> abc.Generator[list[str | None] | None, None, None]:
    """
    Find fractions of observed atoms compared to canonical residue versions
    stored in CCD.

    :param a: Non-empty atom array.
    :param excluding_names: A sequence of atom names to exclude
        for calculation.
    :param excluding_elements: A sequence of element names to exclude
        for calculation.
    :return: A generator of observed atom fractions per residue in
        `a` or ``None`` if a residue was not found in CCD.
    """
    if len(a) == 0:
        raise MissingData('Array is empty')
    for r_can, r_obs in zip(iter_canonical(a), bst.residue_iter(a)):
        if r_can is None:
            yield None
        else:
            r_can = _exclude(r_can, excluding_names, excluding_elements)
            _, m_obs = filter_to_common_atoms(r_can, r_obs)
            yield m_obs.sum() / len(r_can)


if __name__ == '__main__':
    raise RuntimeError
