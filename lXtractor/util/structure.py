"""
Low-level utilities to work with structures.
"""
from __future__ import annotations

import gzip
import logging
from collections import abc
from io import IOBase, StringIO, BytesIO, BufferedReader
from itertools import repeat, starmap, chain, combinations
from pathlib import Path

import biotite.structure.info as bstinfo
import biotite.structure.io as bstio
import numpy as np
import rustworkx as rx
from biotite import structure as bst
from more_itertools import unzip, windowed, unique_everseen
from numpy import typing as npt
from scipy.spatial import KDTree
from toolz import compose_left

from lXtractor.core.config import (
    DefaultConfig,
    STRUCTURE_FMT,
    EMPTY_ALTLOC,
    POL_MARKS,
)
from lXtractor.core.exceptions import LengthMismatch, MissingData, FormatError
from lXtractor.util.io import parse_suffix
from lXtractor.util.typing import is_sequence_of

__all__ = (
    "calculate_dihedral",
    "compare_coord",
    "compare_arrays",
    "filter_selection",
    "filter_ligand",
    "filter_polymer",
    "filter_any_polymer",
    "filter_solvent_extended",
    "filter_to_common_atoms",
    "find_primary_polymer_type",
    "find_contacts",
    "find_first_polymer_type",
    "iter_canonical",
    "extend_residue_mask",
    "iter_residue_masks",
    "get_missing_atoms",
    "get_observed_atoms_frac",
    "load_structure",
    "mark_polymer_type",
    "save_structure",
    "to_graph",
)
LOGGER = logging.getLogger(__name__)

_BASIC_COMPARISON_ATOMS = {"N", "CA", "C", "CB"}


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


def compare_coord(a: bst.AtomArray, b: bst.AtomArray, eps: float = 1e-3):
    """
    Compare coordinates between atoms of two atom arrays.

    :param a: The first atom array.
    :param b: The second atom array.
    :param eps: Comparison tolerance.
    :return: ``True`` if the two arrays are of the same length and the absolute
        difference between coordinates of the corresponding atom pairs is
        within `eps`.
    """
    return len(a) == len(b) and compare_arrays(a.coord, b.coord, eps)


def compare_arrays(
    a: npt.NDArray[float | int], b: npt.NDArray[float | int], eps: float = 1e-3
):
    """
    Compare two numerical arrays.

    :param a: The first array.
    :param b: The second array.
    :param eps: Comparison tolerance.
    :return: ``True`` if the absolute difference between the two arrays is
        within `eps`.
    :raises LengthMismatch: If the two arrays are not of the same shape.
    """
    if a.shape != b.shape:
        raise LengthMismatch(
            f"Different shapes of two arrays ({a.shape} vs. {b.shape})."
        )
    return np.all(np.abs(a - b) <= eps)


def check_het_continuity(array: bst.AtomArray):
    """
    Find indices of atoms delineating sequences of hetero and non-hetero atoms.

    :param array: Atom aray.
    :return: Indices of atoms such that:
        (1) if the atom is non-hetero, the previous atom is hetero, and
        (2) if the atom is hetero, the previous atom is non-hetero.
    """
    diff = np.diff(array.hetero)
    return np.where(diff != 0)[0] + 1


def check_polymer_continuity(array: bst.AtomArray, pol_type: str = "p"):
    if pol_type.startswith("p"):
        filt_fn = bst.filter_amino_acids
    elif pol_type.startswith("n"):
        filt_fn = bst.filter_nucleotides
    elif pol_type.startswith("c"):
        filt_fn = bst.filter_carbohydrates
    else:
        raise ValueError(f"Unsupported polymer type {pol_type}")

    mask = filt_fn(array)
    diff = np.diff(mask)
    return np.where(diff != 0)[0] + 1


def get_chains(a: bst.AtomArray) -> list[str]:
    """
    :param a: Atom Array.
    :return: A list of unique chain IDs in order or appearance.
    """
    return list(unique_everseen(a.chain_id))


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
        return get_residue_count(a), bst.residue_iter(a)

    def process_pair(
        r1: bst.AtomArray, r2: bst.AtomArray
    ) -> tuple[np.ndarray, np.ndarray]:
        r1_name, r2_name = r1.res_name[0], r2.res_name[0]
        atom_names = set(r1.atom_name) & set(r2.atom_name)

        if r1_name != r2_name:
            if not allow_residue_mismatch:
                raise ValueError(
                    f"Residue names must match. Got {r1_name} from the first array "
                    f"and {r2_name} from the second one. Use `allow_residue_mismatch` "
                    "to allow residue name mismatches."
                )
            atom_names &= _BASIC_COMPARISON_ATOMS

        m1, m2 = map(lambda r: np.isin(r.atom_name, list(atom_names)), [r1, r2])
        if m1.sum() != m2.sum():
            raise ValueError(
                f"Obtained different sets of atoms {atom_names}. "
                f"Residue 1: {r1[m1]}. Residue 2: {r2[m2]}"
            )

        return m1, m2

    (a1_l, a1_it), (a2_l, a2_it) = map(preprocess_array, [a1, a2])

    if a1_l != a2_l:
        raise LengthMismatch(
            "The number of residues must match between structures. "
            f"Got {a1_l} for `a1` and {a2_l} for `a2`."
        )

    mask1, mask2 = map(
        # numpy + mypy = confusion
        lambda x: np.concatenate(list(x)),  # type: ignore
        unzip(starmap(process_pair, zip(a1_it, a2_it, strict=True))),
    )

    return mask1, mask2


def _is_polymer(array, min_size, pol_type):
    if pol_type.startswith("p"):
        filt_fn = bst.filter_amino_acids
    elif pol_type.startswith("n"):
        filt_fn = bst.filter_nucleotides
    elif pol_type.startswith("c"):
        filt_fn = bst.filter_carbohydrates
    else:
        raise ValueError(f"Unsupported polymer type {pol_type}")

    mask = filt_fn(array)
    return get_residue_count(array[mask]) >= min_size


def _split_array(a: bst.AtomArray, idx: abc.Iterable[int]):
    idx = chain([0], idx, [len(a)])
    for i, j in windowed(idx, 2):
        yield a[i:j]


def _find_breaks(a: bst.AtomArray) -> npt.NDArray[np.int_]:
    return compose_left(np.concatenate, np.sort, np.unique)(
        [
            bst.check_res_id_continuity(a),
            bst.check_backbone_continuity(a),
            *(check_polymer_continuity(a, p) for p in ["n", "p", "c"]),
            check_het_continuity(a)[-1:],
        ]
    )


def find_first_polymer_type(
    a: bst.AtomArray | npt.NDArray[int],
    min_size: int = 2,
    order: tuple[str, str, str] = ("p", "n", "c"),
) -> str:
    """
    Determines polymer type of the supplied atom array or an array of atom marks.

    Probe polymer types in a sequence in a given order.
    If a polymer with at least `min_size` atoms of the probed type is found,
    it will be returned.

    .. hint::
        The function serves as a good quick-check when a single polymer type is
        expected, which should always be true when `a` is an array of atom marks.

    :param a: An arbitrary array of atoms.
    :param min_size: A minimum number of monomers in a polymer.
    :param order: An order of the polymers to probe.
    :return: The first polymer type to accommodate `min_size` requirement.
    """
    if isinstance(a, bst.AtomArray):
        for p in order:
            if _is_polymer(a, min_size, p):
                return p
        return "x"
    pol_marks = dict(POL_MARKS)
    for p in order:
        if np.sum(a == pol_marks[p]) >= min_size:
            return p
    return "x"


def find_primary_polymer_type(
    a: bst.AtomArray, min_size: int = 2, residues: bool = False
) -> tuple[np.ndarray, str]:
    """
    Find the major polymer type, i.e., the one with the largest number of
    atoms or monomers.

    :param a: An arbitrary atom array.
    :param min_size: Minimum number of monomers for a polymer.
    :param residues: ``True`` if the dominant polymer should be picked according
        to the number of residues. Otherwise, the number of atoms will be used.
    :return: A binary mask pointing at the polymer atoms in `a` and the polymer
        type -- "c" (carbohydrate), "n" (nucleotide), or "p" (peptide). If no
        polymer atoms were found, polymer type will be designated as "x".
    """
    pol_types = mark_polymer_type(a, min_size)
    is_nuc, is_pep, is_carb = (pol_types == p for p in ["n", "p", "c"])
    key = (lambda x: get_residue_count(a[x[0]])) if residues else (lambda x: x[0].sum())
    is_pol, pol_type = max(
        [(is_carb, "c"), (is_nuc, "n"), (is_pep, "p")],
        key=key,
    )
    if np.any(is_pol):
        return is_pol, pol_type
    return is_pol, "x"


def mark_polymer_type(a: bst.AtomArray, min_size: int = 2) -> npt.NDArray[np.str_]:
    """
    Denote polymer type in an atom array.

    It will find the breakpoints in `a` and split it into segments.
    Each segment will be checked separately to determine its polymer type.
    The results are then concatenated into a single array and returned.

    :param a: Any atom array.
    :param min_size: Minimum number of consecutive monomers in a polymer.
    :return: An array where each atom of `a` is marked by a character: ``"n"``,
        ``"p"``, or ``"c"`` for nucleotide, peptide, and carbohydrate.
        Non-polymer atoms are marked by "x".
    """

    if len(a) < min_size:
        return np.full_like(a, "x", dtype=np.str_)

    chain_ids = get_chains(a)
    if len(chain_ids) > 1:
        pol_types = np.full_like(a, "x")
        for c in chain_ids:
            chain_mask = a.chain_id == c
            for alt_mask in iter_altloc_masks(a):
                m = chain_mask & alt_mask
                pol_types[m] = mark_polymer_type(a[m], min_size)
        return pol_types

    chunks = _split_array(a, _find_breaks(a))
    return np.concatenate(
        [np.full_like(x, find_first_polymer_type(x, min_size)) for x in chunks]
    )


def filter_polymer(
    a: bst.AtomArray, min_size=2, pol_type="peptide"
) -> npt.NDArray[np.bool_]:
    """
    Filter for atoms that are a part of a consecutive standard macromolecular
    polymer entity.

    :param a: The array to filter.
    :param min_size: The minimum number of monomers.
    :param pol_type: The polymer type, either ``"peptide"``, ``"nucleotide"``,
        or ``"carbohydrate"``. Abbreviations are supported: ``"p"``, ``"pep"``,
        ``"n"``, etc.
    :return: This array is `True` for all indices in `array`, where atoms
        belong to consecutive polymer entity having at least `min_size` monomers.
    """
    pol_types = mark_polymer_type(a, min_size)
    return np.equal(pol_types, pol_type)


def filter_any_polymer(a: bst.AtomArray, min_size: int = 2) -> np.ndarray:
    """
    Get a mask indicating atoms being a part of a macromolecular polymer:
    peptide, nucleotide, or carbohydrate.

    :param a: Array of atoms.
    :param min_size: Min number of polymer monomers.
    :return: A boolean mask ``True`` for polymers' atoms.
    """
    pol_types = mark_polymer_type(a, min_size)
    return np.not_equal(pol_types, "x")


def filter_solvent_extended(
    a: bst.AtomArray,
) -> np.ndarray:
    """
    Filter for solvent atoms using a curated solvent list including non-water
    molecules typically being a part of a crystallization solution.

    :param a: Atom array.
    :return: A boolean mask ``True`` for solvent atoms.
    """
    if len(a) == 0:
        return np.empty(shape=0, dtype=np.bool_)
    return (
        np.isin(a.res_name, DefaultConfig["residues"]["solvents"])
        | (np.vectorize(len)(a.res_name) != 3)
        & ~np.isin(a.res_name, DefaultConfig["residues"]["nucleotides"])
        # | (np.vectorize(len)(a.res_name) == 1)
    )


def filter_ligand(a: bst.AtomArray) -> np.ndarray:
    """
    Filter for ligand atoms -- non-polymer and non-solvent hetero atoms.

    ..note ::
        No contact-based verification is performed here.

    :param a: Atom array.
    :return: A boolean mask ``True`` for ligand atoms.
    """
    # return ~(is_polymer | is_solvent) & a.hetero
    return ~(filter_any_polymer(a) | filter_solvent_extended(a))


def find_contacts(
    a: bst.AtomArray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find contacts between a subset of atoms within the structure and the rest
    of the structure. An atom is considered to be in contact with another atom
    if the distance between them is below the threshold for the non-covalent
    bond specified in config (``DefaultConfig["bonds"]["NC-NC"][1]``).

    :param a: Atom array.
    :param mask: A boolean mask ``True`` for atoms for which to find contacts.
    :return: A tuple with three arrays of size equal to the `a`'s number of atoms:

        #. Contact mask: ``True`` for ``a[~mask]`` atoms in contact with
            ``a[mask]``.
        #. Distances: for ``a[mask]`` atoms to the closest ``a[~mask]`` atom.
        #. Indices: of these closest ``a[~mask]`` atoms within the `mask`.

        Suppose that ``mask`` specifies a ligand. Then, for ``i``-th atom in `a`,
        ``contacts[i]``, ``distances[i]``, ``indices[i]`` indicate whether
        ``a[i]`` has a contact, the precise distance from ``a[i]`` atom to the
        closest ligand atom, and an index of this ligand atom, respectively.

    """

    # An MxL matrix where L is the number of atoms in the structure and M is the
    # number of atoms in the ligand residue
    coord = np.round(a.coord, decimals=2)
    # d = np.linalg.norm(a[mask].coord[:, np.newaxis] - a.coord, axis=-1)
    d = np.linalg.norm(coord[mask, np.newaxis] - coord, axis=-1)
    d_min = np.min(d, axis=0)  # min distance from sub atoms to the rest
    d_argmin = np.argmin(d, axis=0)  # sub atom indices contacting structure

    bonds = DefaultConfig["bonds"]

    contacts = np.zeros_like(d_min, dtype=bool)
    nc_upper = bonds["NC-NC"][1]
    contacts[d_min < nc_upper] = True

    return contacts, d_min, d_argmin


def extend_residue_mask(a: bst.AtomArray, idx: list[int]) -> npt.NDArray[np.bool_]:
    """
    Extend a residue mask for given atoms.

    :param a: An arbitrary atom array.
    :param idx: Indices pointing to atoms at which to extend the mask.
    :return: The extended mask, where ``True`` indicates that the atom belongs
        to the same residue as indicated by `idx`.
    """
    res_ids = np.unique(a[idx].res_id)
    chain_ids = np.unique(a[idx].chain_id)
    return np.isin(a.res_id, res_ids) & np.isin(a.chain_id, chain_ids)


def iter_residue_masks(
    a: bst.AtomArray,
) -> abc.Generator[npt.NDArray[np.bool_], None, None]:
    """
    Iterate over residue masks.

    :param a: Atom array.
    :return: A generator over boolean masks for each residue in `a`.
    """
    starts = bst.get_residue_starts(a, add_exclusive_stop=True)
    arange = np.arange(len(a))
    for i in range(len(starts) - 1):
        yield (arange >= starts[i]) & (arange < starts[i + 1])


def iter_altloc_masks(
    a: bst.AtomArray,
) -> abc.Generator[npt.NDArray[np.bool_], None, None]:
    if not hasattr(a, "altloc_id"):
        yield np.full_like(a, True, dtype=np.bool_)
    else:
        aids = a.altloc_id
        aids_unique = sorted(unique_everseen(aids))
        if len(aids_unique) == 0:
            raise RuntimeError("...")
        elif len(aids_unique) == 1:
            yield np.full_like(a, True, dtype=np.bool_)
        else:
            empty_ = np.isin(aids, EMPTY_ALTLOC)
            for aid in aids_unique[1:]:
                yield (aids == aid) | empty_


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


def get_residue_count(a: bst.AtomArray):
    """
    A simple wrapper around the original biotite `get_residue_count`
    that handles an empty `a` properly.

    :param a: An Atom Array.
    :return: The number of distinct residues in `a`.
    """
    if len(a) == 0:
        return 0
    return bst.get_residue_count(a)


def get_missing_atoms(
    a: bst.AtomArray,
    excluding_names: abc.Sequence[str] | None = ("OXT",),
    excluding_elements: abc.Sequence[str] | None = ("H",),
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
        raise MissingData("Array is empty")
    for r_can, r_obs in zip(iter_canonical(a), bst.residue_iter(a)):
        if r_can is None:
            yield None
        else:
            r_can = _exclude(r_can, excluding_names, excluding_elements)
            m_can, _ = filter_to_common_atoms(r_can, r_obs)
            yield list(r_can[~m_can].atom_name)


def get_observed_atoms_frac(
    a: bst.AtomArray,
    excluding_names: abc.Sequence[str] | None = ("OXT",),
    excluding_elements: abc.Sequence[str] | None = ("H",),
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
        raise MissingData("Array is empty")
    for r_can, r_obs in zip(iter_canonical(a), bst.residue_iter(a)):
        if r_can is None:
            yield None
        else:
            r_can = _exclude(r_can, excluding_names, excluding_elements)
            _, m_obs = filter_to_common_atoms(r_can, r_obs)
            yield m_obs.sum() / len(r_can)


def _check_fmt(fmt: str):
    if not fmt:
        raise FormatError(
            "The format must be specified explicitly when not using a Path-like input."
        )
    if fmt not in STRUCTURE_FMT:
        raise FormatError(
            f"Unsupported structure format {fmt}. "
            f"Supported formats are: {STRUCTURE_FMT}."
        )


def load_structure(
    inp: IOBase | Path | str | bytes, fmt: str = "", *, gz: bool = False, **kwargs
) -> bst.AtomArray:
    """
    This is a simplified version of a ``biotite.io.general.load_structure``
    extending the supported input types. Namely, it allows using paths,
    strings, bytes or gzipped files. On the other hand, there are less supported
    formats: pdb, cif, and mmtf.

    :param inp: Input to load from. It can be a path to a file, an opened file
        handle, a string or bytes of file contents. Gzipped bytes and files are
        supported.
    :param fmt: If ``inp`` is a ``Path``-like object, it must be of the form
        "name.fmt" or "name.fmt.gz". In this case, ``fmt`` is ignored.
        Otherwise, it is used to determine the parser type and must be provided.
    :param gz: If ``inp`` is gzipped ``bytes``, this flag must be ``True``.
    :param kwargs: Passed to ``get_structure``: either a method or a separate
        function used by ``biotite`` to convert the input into an ``AtomArray``.
    :return:
    """
    if isinstance(inp, Path):
        suffix = parse_suffix(inp)
        if suffix.endswith(".gz"):
            fmt = inp.suffixes[-2]
        else:
            fmt = suffix
        fmt = fmt.removeprefix(".")

    _check_fmt(fmt)

    io_type = BytesIO if fmt == "mmtf" else StringIO
    read_mode = "rb" if fmt == "mmtf" else "r"

    if isinstance(inp, Path):
        if inp.name.endswith(".gz"):
            with gzip.open(inp) as f:
                content = f.read()
                if io_type is StringIO:
                    content = content.decode("utf-8")
                handle = io_type(content)
        else:
            handle = inp.open(read_mode)
    elif isinstance(inp, str):
        handle = StringIO(inp)
    elif isinstance(inp, bytes):
        content = gzip.decompress(inp) if gz else inp
        if io_type is StringIO:
            content = content.decode("utf-8")
        handle = io_type(content)
    elif isinstance(inp, BufferedReader) and gz:
        return load_structure(inp.read(), fmt, gz=gz)
    else:
        handle = inp

    if fmt == "pdb":
        from biotite.structure.io.pdb import PDBFile

        file = PDBFile.read(handle)
        array = file.get_structure(**kwargs)
    elif fmt in ["cif"]:
        from biotite.structure.io.pdbx import CIFFile, get_structure

        file = CIFFile.read(handle)
        array = get_structure(file, **kwargs)
    elif fmt == "mmtf":
        from biotite.structure.io.mmtf import MMTFFile, get_structure

        file = MMTFFile.read(handle)
        array = get_structure(file, **kwargs)
    else:
        raise FormatError(f"Unsupported format {fmt}.")

    handle.close()

    if isinstance(array, bst.AtomArrayStack) and array.stack_depth() == 1:
        # Stack containing only one model -> return as atom array
        array = array[0]

    return array


def save_structure(array: bst.AtomArray, path: Path, **kwargs):
    """
    This is a simplified version of a ``biotite.io.general.save_structure``.
    On the one hand, it can conveniently compress the data using ``gzip``.
    On the other hand, the number of supported formats is fewer: pdb, cif, and
    mmtf.

    :param array: An ``AtomArray`` to write.
    :param path: A path with correct extension, e.g.,
        ``Path("data/structure.pdb")``, or ``Path("data/structure.pdb.gz")``.
    :param kwargs: If compressing is not required, the original ``save_structure``
        from biotite is used with these ``kwargs``. Otherwise, ``kwargs`` are
        ignored.
    :return: If the file was successfully written, returns the original `path`.
    """
    suffix = parse_suffix(path)
    if suffix.endswith(".gz"):
        fmt = path.suffixes[-2]
        gz = True
    else:
        fmt = suffix
        gz = False

    if not gz:
        bstio.save_structure(path, array, **kwargs)
        return path

    fmt = fmt.removeprefix(".")
    _check_fmt(fmt)

    if fmt == "mmtf":
        from biotite.structure.io.mmtf import MMTFFile, set_structure

        io, file = BytesIO(), MMTFFile()
        set_structure(file, array)
        file.write(io)
    elif fmt == "pdb":
        from biotite.structure.io.pdb import PDBFile

        io, file = StringIO(), PDBFile()
        file.set_structure(array)
        file.write(io)
    elif fmt == "cif":
        from biotite.structure.io.pdbx import PDBxFile, set_structure

        io, file = StringIO(), PDBxFile()
        set_structure(file, array, data_block="STRUCTURE")
        file.write(io)
    else:
        raise FormatError(f"Unsupported format {fmt}.")

    io.seek(0)

    content = io.read()
    if fmt != "mmtf":
        content = str.encode(content)

    with gzip.open(path, "wb") as f:
        f.write(content)

    io.close()

    return path


def _get_element_combs(a: bst.AtomArray) -> abc.Iterator[str]:
    """
    Get valid combinations of elements that can form covalent bonds in the
    provided atom array.
    """
    #
    elements = sorted(unique_everseen(a.element))
    el_combs = chain(((el, el) for el in elements), combinations(elements, 2))
    el_combs = map(lambda x: "-".join(x), el_combs)
    el_combs = filter(lambda x: x in DefaultConfig["bonds"], el_combs)
    return el_combs


def _is_valid_altloc(a1: bst.Atom, a2: bst.Atom) -> bool:
    if not (hasattr(a1, "altloc_id") or hasattr(a2, "altloc_id")):
        return True
    a1_alt, a2_alt = a1.altloc_id, a2.altloc_id
    return a1_alt in EMPTY_ALTLOC or a2_alt in EMPTY_ALTLOC or a1_alt == a2_alt


def to_graph(a: bst.AtomArray, split_chains: bool = False) -> rx.PyGraph:
    """
    Create a molecular connectivity graph from an atom array.

    Molecular graph is a undirected graph without multiedges, where nodes are
    indices to atoms. Thus, node indices point directly to atoms in the provided
    atom array, and the number of nodes equals the number of atoms. A pair of
    nodes has an edge between them, if they form a covalent bond. The edges
    are constructed according to atom-depended bond thresholds defined by the
    global config. These distances are stored as edge values. See the docs of
    `rustworkx` on how to manipulate the resulting graph object.

    :param a: Atom array to guild a graph from.
    :param split_chains: Edges between atoms from different chains are forbidden.
    :return: A graph object where nodes are atom indices and edges represent
        covalent bonds.
    """
    # Create initial graph object and populate nodes.
    g = rx.PyGraph(multigraph=False)
    g.add_nodes_from(range(len(a)))

    # Make bool masks for chains
    if split_chains:
        chain_ids = np.unique(a.chain_id)
        chain_masks = (a.chain_id == c for c in chain_ids)
    else:
        chain_masks = [np.full_like(a, True, dtype=np.bool_)]

    for chain_mask in chain_masks:
        el_combs = _get_element_combs(a[chain_mask])
        for pair in el_combs:
            # Subset to atoms from an element pair.
            el1, el2 = pair.split("-")
            a_idx = np.where(np.isin(a.element, [el1, el2]) & chain_mask)[0]
            if len(a_idx) == 0:
                continue

            # Build a KDTree object to speed up neighbor searches
            kdtree = KDTree(a[a_idx].coord)
            lower, upper = DefaultConfig["bonds"][pair]
            # Query for pairs under distance threshold for non-covalent interaction
            pairs = kdtree.query_pairs(r=upper)
            if not pairs:
                continue
            valid_pairs = [(el1, el2), (el2, el1)]
            edges = (
                (
                    a_idx[i],  # Original atom indices
                    a_idx[j],
                    np.linalg.norm(a.coord[a_idx[i]] - a.coord[a_idx[j]]),  # Dist
                )
                for i, j in pairs
                # Pair is valid if element combinations are valid and if atom
                # altlocs are compatible.
                if (a[a_idx[i]].element, a[a_idx[j]].element) in valid_pairs
                and _is_valid_altloc(a[a_idx[i]], a[a_idx[j]])
            )
            # Batch-add edges to the graph
            edges = list(filter(lambda x: upper > x[-1] > lower, edges))
            g.add_edges_from(edges)

    return g


if __name__ == "__main__":
    raise RuntimeError
