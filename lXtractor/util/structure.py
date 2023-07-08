"""
Low-level utilities to work with structures.
"""
from __future__ import annotations

import gzip
import logging
import operator as op
from collections import abc
from functools import reduce, partial
from io import IOBase, StringIO, BytesIO, BufferedReader
from itertools import repeat, starmap, chain
from pathlib import Path

import biotite.structure as bst
import biotite.structure.info as bstinfo
import biotite.structure.io as bstio
import numpy as np
from more_itertools import unzip, windowed

from lXtractor.core.config import SOLVENTS, BondThresholds, STRUCTURE_FMT
from lXtractor.core.exceptions import LengthMismatch, MissingData, FormatError
from lXtractor.util.io import parse_suffix
from lXtractor.util.typing import is_sequence_of

__all__ = (
    "calculate_dihedral",
    "filter_selection",
    "filter_ligand",
    "filter_polymer",
    "filter_any_polymer",
    "filter_solvent_extended",
    "filter_to_common_atoms",
    "find_contacts",
    "iter_canonical",
    "iter_residue_masks",
    "get_missing_atoms",
    "get_observed_atoms_frac",
    "load_structure",
    "save_structure",
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
        return bst.get_residue_count(a), bst.residue_iter(a)

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
    return bst.get_residue_count(array[mask]) >= min_size


def _split_array(a: bst.AtomArray, idx: abc.Iterable[int]):
    idx = chain([0], idx, [len(a)])
    for i, j in windowed(idx, 2):
        yield a[i:j]


def filter_polymer(a, min_size=2, pol_type="peptide"):
    # TODO: make a PR
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

    res_breaks = bst.check_res_id_continuity(a)
    bb_breaks = bst.check_backbone_continuity(a)
    het_breaks = check_het_continuity(a)
    pep_breaks = check_polymer_continuity(a, "p")
    nuc_breaks = check_polymer_continuity(a, "n")
    car_breaks = check_polymer_continuity(a, "c")
    if len(het_breaks) != 0:
        # take only het tail to avoid including modified residues
        het_breaks = het_breaks[-1:]

    breaks = np.concatenate(
        [res_breaks, bb_breaks, het_breaks, pep_breaks, nuc_breaks, car_breaks]
    )
    split_idx = np.sort(np.unique(breaks))

    check_pol = partial(_is_polymer, min_size=min_size, pol_type=pol_type)
    bool_idx = map(
        lambda x: np.full(len(x), check_pol(x), dtype=bool),
        _split_array(a, split_idx),
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
        op.or_, (filter_polymer(a, min_size, pol_type) for pol_type in ["p", "n", "c"])
    )


def filter_solvent_extended(a: bst.AtomArray) -> np.ndarray:
    """
    Filter for solvent atoms using a curated solvent list including non-water
    molecules typically being a part of a crystallization solution.

    :param a: Atom array.
    :return: A boolean mask ``True`` for solvent atoms.
    """
    # TODO: should use size threshold to automatically exclude small ligands
    if len(a) == 0:
        return np.empty(shape=0, dtype=bool)
    return (
        np.isin(a.res_name, SOLVENTS)
        | (np.vectorize(len)(a.res_name) != 3)
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
    a: bst.AtomArray, mask: np.ndarray, ts: BondThresholds = BondThresholds()
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find contacts between a subset of atoms within the structure and the rest
    of the structure.

    :param a: Atom array.
    :param mask: A boolean mask ``True`` for atoms for which to find contacts.
    :param ts: Bond thresholds.
    :return: A tuple with three arrays of size equal to the `a`'s number of atoms:

        #. Contacts.
        #. Distances.
        #. Index of an atom within ``a[mask]`` closest the structure's atom.

        In the first array, ``0`` indicate the lack of contact, ``1`` indicate
        a non-covalent contact, and ``2`` indicate a covalent contact.

        For ``i``-th atom in `a`, ``contacts[i]``, ``distances[i]``,
        ``indices[i]`` indicate whether ``a[i]`` has a contact, the distance
        from this atom to the ``a[mask]`` atom whose index is specified by
        ``a[mask][distances[i]]``.
    """

    # An MxL matrix where L is the number of atoms in the structure and M is the
    # number of atoms in the ligand residue
    d = np.linalg.norm(a[mask].coord[:, np.newaxis] - a.coord, axis=-1)
    d_min = np.min(d, axis=0)  # min distance from sub atoms to the rest
    d_argmin = np.argmin(d, axis=0)  # sub atom indices contacting structure

    contacts = np.zeros_like(d_min, dtype=int)
    contacts[(d_min >= ts.non_covalent.lower) & (d_min <= ts.non_covalent.upper)] = 1
    contacts[(d_min >= ts.covalent.lower) & (d_min <= ts.covalent.upper)] = 2
    contacts[mask] = 0

    return contacts, d_min, d_argmin


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
    elif fmt in ["cif", "pdbx"]:
        from biotite.structure.io.pdbx import PDBxFile, get_structure

        file = PDBxFile.read(handle)
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


if __name__ == "__main__":
    raise RuntimeError
