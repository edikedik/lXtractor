"""
Module defining variables calculated on structures.
"""
from __future__ import annotations

import logging
import typing as t
from abc import abstractmethod
from collections import abc
from itertools import starmap

import biotite.structure as bst
import numpy as np
from more_itertools import unique_justseen
from scipy.spatial import KDTree
from toolz import pipe

from lXtractor.core.config import DefaultConfig
from lXtractor.core.exceptions import FailedCalculation, InitError
from lXtractor.util.structure import calculate_dihedral
from lXtractor.variables.base import (
    StructureVariable,
    AggFns,
    MappingT,
    RT,
)
from lXtractor.variables.util import (
    residue_mask,
    atom_mask,
    _get_residue,
    _get_coord,
    _agg_dist,
)

if t.TYPE_CHECKING:
    from lXtractor.core.structure import GenericStructure

__all__ = (
    "Dist",
    "AggDist",
    "Dihedral",
    "PseudoDihedral",
    "Phi",
    "Psi",
    "Omega",
    "Chi1",
    "Chi2",
    "SASA",
    "ClosestLigandContactsCount",
    "ClosestLigandNames",
    "ClosestLigandDist",
    "Contacts",
)

LOGGER = logging.getLogger(__name__)


def _verify_consecutive(positions: abc.Iterable[int]) -> None:
    """
    Verify whether positions in a given iterable are consecutive.
    """
    # Unduplicate consecutively duplicated elements
    positions = list(unique_justseen(positions))

    # Check consecutive positions
    for i in range(1, len(positions)):
        current, previous = positions[i], positions[i - 1]
        if current != previous + 1:
            raise FailedCalculation(
                f"Positions {previous} and {current} are not consecutive "
                f"in a given list {positions}"
            )


class Dist(StructureVariable):
    """
    A distance between two atoms.
    """

    __slots__ = ("p1", "p2", "a1", "a2", "com")

    def __init__(
        self,
        p1: int,
        p2: int,
        a1: str | None = None,
        a2: str | None = None,
        com: bool = False,
    ):
        #: Position 1.
        self.p1: int = p1
        #: Position 2.
        self.p2: int = p2
        #: Atom name 1.
        self.a1: str | None = a1
        #: Atom name 2.
        self.a2: str | None = a2
        #: Use center of mass instead of concrete atoms.
        self.com: bool = com

        if any((not com and a1 is None, not com and a2 is None)):
            raise ValueError(
                'No atom name specified and "center of mass" flag is down. '
                "Therefore, not possible to calculate distance."
            )

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
        self, obj: GenericStructure, mapping: MappingT | None = None
    ) -> float:
        def get_coord(p: int, a: str | None) -> np.ndarray:
            return _get_coord(_get_residue(p, obj.array, mapping), a)  # type: ignore

        xyz1, xyz2 = get_coord(self.p1, self.a1), get_coord(self.p2, self.a2)

        return np.linalg.norm(xyz2 - xyz1)


class AggDist(StructureVariable):
    """
    Aggregated distance between two residues.

    It will return ``agg_fn(pdist)`` where ``pdist`` is an array of
    all pairwise distances between atoms of `p1` and `p2`.
    """

    __slots__ = ("p1", "p2", "key")

    def __init__(self, p1: int, p2: int, key: str = "min"):
        """
        :param p1: Position 1.
        :param p2:  Position 2.
        :param key: Agg function name.

        Available aggregator functions are:

        >>> print(list(AggFns))
        ['min', 'max', 'mean', 'median']

        """
        if key not in AggFns:
            raise InitError(
                f"Wrong key {key}. " f"Available aggregators: {list(AggFns)}"
            )
        #: Agg function name.
        self.key = key
        #: Position 1.
        self.p1 = p1
        #: Position 2.
        self.p2 = p2

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
        self, obj: GenericStructure, mapping: MappingT | None = None
    ) -> float:
        res1, res2 = map(
            # no type info for biotite
            lambda p: _get_residue(p, obj.array, mapping),  # type: ignore
            [self.p1, self.p2],
        )
        return _agg_dist(res1, res2, AggFns[self.key])


# class AllDist(StructureVariable):
#     __slots__ = ('key',)
#
#     def __init__(self, key: str = 'min'):
#         if key not in AggFns:
#             raise ValueError(
#                 f'Wrong key {key}. '
#                 f'Available aggregators: {list(AggFns)}')
#         self.key = key
#
#     @property
#     def rtype(self) -> t.Type[float]:
#         return float
#
#     def calculate(
#             self, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
#     ) -> t.List[t.Tuple[int, int, float]]:
#         raise NotImplementedError
#
#         residues = array.get_residues()
#         if mapping:
#             residues = filter(
#                 lambda r: r.get_id()[1] in mapping.values(),
#                 residues)
#         cs = combinations(residues, 2)
#         ds = starmap(
#             lambda r1, r2: (
#                 r1.get_id()[1],
#                 r2.get_id()[1],
#                 agg_dist(r1, r2, AggFns[self.key])),
#             cs)
#         if mapping:
#             m_rev = {v: k for k, v in mapping.items()}
#             ds = starmap(
#                 lambda r1_id, r2_id, d: (
#                     m_rev[r1_id], m_rev[r2_id], d),
#                 ds)
#         return list(ds)


class Dihedral(StructureVariable):
    """
    Dihedral angle involving four different atoms.
    """

    __slots__ = ("p1", "p2", "p3", "p4", "a1", "a2", "a3", "a4", "name")

    def __init__(
        self,
        p1: int,
        p2: int,
        p3: int,
        p4: int,
        a1: str,
        a2: str,
        a3: str,
        a4: str,
        name: str = "GenericDihedral",
    ):
        #: Used to designate special kinds of dihedrals.
        self.name: str = name
        #: Position.
        self.p1, self.p2, self.p3, self.p4 = p1, p2, p3, p4
        #: Atom name.
        self.a1, self.a2, self.a3, self.a4 = a1, a2, a3, a4

    @property
    def rtype(self) -> t.Type[float]:
        return float

    @property
    def positions(self) -> list[int]:
        """
        :return: A list of positions `p1-p4`.
        """
        return [self.p1, self.p2, self.p3, self.p4]

    @property
    def atoms(self) -> list[str]:
        """
        :return: A list of atoms `a1-a4`.
        """
        return [self.a1, self.a2, self.a3, self.a4]

    def calculate(
        self, obj: GenericStructure, mapping: MappingT | None = None
    ) -> float:
        # Map positions to the PDB numbering
        coordinates = starmap(
            lambda p, a: pipe(  # type: ignore  # no type info for biotite
                _get_residue(p, obj.array, mapping),  # type: ignore
                _get_coord(atom_name=a),  # pylint: disable=no-value-for-parameter
            ),
            zip(self.positions, self.atoms),
        )
        return calculate_dihedral(*coordinates)


class PseudoDihedral(Dihedral):
    """
    Pseudo-dihedral angle -
    "the torsion angle between planes defined by 4 consecutive
    alpha-carbon atoms."

    """

    __slots__ = ()

    def __init__(self, p1: int, p2: int, p3: int, p4: int):
        super().__init__(p1, p2, p3, p4, "CA", "CA", "CA", "CA", name="PseudoDihedral")


class Phi(Dihedral):
    """
    Phi dihedral angle.
    """

    __slots__ = ("p",)

    def __init__(self, p: int):
        self.p = p
        super().__init__(p - 1, p, p, p, "C", "N", "CA", "C", name="Phi")


class Psi(Dihedral):
    """
    Psi dihedral angle.
    """

    __slots__ = ("p",)

    def __init__(self, p: int):
        self.p = p
        super().__init__(p, p, p, p + 1, "N", "CA", "C", "N", name="Psi")


class Omega(Dihedral):
    """
    Omega dihedral angle.
    """

    __slots__ = ("p",)

    def __init__(self, p: int):
        self.p = p
        super().__init__(p, p, p + 1, p + 1, "CA", "C", "N", "CA", name="Omega")


class CompositeDihedral(StructureVariable):
    """
    An abstract class that defines a dihedral angle s.t. the atoms are within
    a single residue but the atom names may vary depending on the residue type.
    """

    __slots__ = ("p",)

    def __init__(self, p: int):
        #: Position
        self.p = p

    @property
    def rtype(self) -> t.Type[float]:
        return float

    @staticmethod
    @abstractmethod
    def get_dihedrals(pos: int) -> abc.Iterable[Dihedral]:
        """
        Implemented by child classes.

        :param pos: Position to create :class:`Dihedral` instances.
        :return: An iterable over :class:`Dihedral`'s.
            The :meth:`calculate` will try calculating dihedrals in the
            provided order until the first successful calculation. If no
            calculations were successful, will raise :class:`FailedCalculation`
            error.
        """
        raise NotImplementedError("Must be implemented by the subclass")

    def calculate(
        self, obj: GenericStructure, mapping: MappingT | None = None
    ) -> float:
        res = None
        dihedrals = self.get_dihedrals(self.p)
        for d in dihedrals:
            try:
                res = d.calculate(obj, mapping)
                break
            except FailedCalculation:
                pass
        if res is None:
            raise FailedCalculation(
                f"Couldn't calculate any of dihedrals " f"{[d.id for d in dihedrals]}"
            )
        return res


class Chi1(CompositeDihedral):
    """
    Chi1-dihedral angle.
    """

    __slots__ = ()

    @staticmethod
    def get_dihedrals(pos) -> list[Dihedral]:
        return [
            Dihedral(pos, pos, pos, pos, "N", "CA", "CB", "CG", "Chi1_CG"),
            Dihedral(pos, pos, pos, pos, "N", "CA", "CB", "CG1", "Chi1_CG1"),
            Dihedral(pos, pos, pos, pos, "N", "CA", "CB", "OG", "Chi1_OG"),
            Dihedral(pos, pos, pos, pos, "N", "CA", "CB", "OG1", "Chi1_OG1"),
            Dihedral(pos, pos, pos, pos, "N", "CA", "CB", "SG", "Chi1_SG"),
        ]


class Chi2(CompositeDihedral):
    """
    Chi2-dihedral angle,
    """

    __slots__ = ()

    @staticmethod
    def get_dihedrals(pos) -> list[Dihedral]:
        return [
            Dihedral(pos, pos, pos, pos, "CA", "CB", "CG", "CD", "Chi2_CG-CD"),
            Dihedral(pos, pos, pos, pos, "CA", "CB", "CG", "OD1", "Chi2_CG-OD1"),
            Dihedral(pos, pos, pos, pos, "CA", "CB", "CG", "ND1", "Chi2_CG-ND1"),
            Dihedral(pos, pos, pos, pos, "CA", "CB", "CG1", "CD", "Chi2_CG1-CD"),
            Dihedral(pos, pos, pos, pos, "CA", "CB", "CG", "CD1", "Chi2_CG-CD1"),
            Dihedral(pos, pos, pos, pos, "CA", "CB", "CG", "SD", "Chi2_CG-SD"),
        ]


class SASA(StructureVariable):
    """
    Solvent-accessible surface area of a residue or a specific atom.

    The SASA is calculated for the whole array, and subset to all or a single
    atoms of a residue (so atoms are taken into account for calculation).
    """

    __slots__ = ("p", "a")

    def __init__(self, p: int, a: str | None = None):
        self.p = p
        self.a = a

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
        self, obj: GenericStructure, mapping: MappingT | None = None
    ) -> float | None:
        m = residue_mask(self.p, obj.array, mapping)

        if self.a is not None:
            m &= obj.array.atom_name == self.a

        if m.sum() == 0:
            raise FailedCalculation("Empty selection")
        try:
            sasa = bst.sasa(obj.array, atom_filter=m)
        except Exception as e:
            raise FailedCalculation(f"Failed to calculate {self} on {obj}") from e
        return float(np.sum(sasa[~np.isnan(sasa)]))


class ClosestLigandContactsCount(StructureVariable):
    """
    The number of atoms involved in contacting ligands.
    """

    __slots__ = ("p", "a")

    def __init__(self, p: int, a: str | None = None):
        #: Residue position.
        self.p = p

        #: Atom name. If not provided, sum contacts across all residue atoms.
        self.a = a

    @property
    def rtype(self) -> t.Type[int]:
        return int

    def calculate(
        self, obj: GenericStructure, mapping: MappingT | None = None
    ) -> float:
        mask = (
            atom_mask(self.p, self.a, obj.array, mapping)
            if self.a is not None
            else residue_mask(self.p, obj.array, mapping)
        )
        n_contacts = 0
        for lig in obj.ligands:
            try:
                cm = lig.contact_mask[mask]
            except IndexError as e:
                raise FailedCalculation(
                    f"Failed to apply obtained position mask derived from {obj.name} "
                    f"to a ligand {lig.res_name} ({lig.parent.name}) parent contacts."
                ) from e
            n_contacts += np.sum(cm != 0)
        return n_contacts


class ClosestLigandNames(StructureVariable):
    """
    ``","``-separated contacting ligand (residue) names.
    """

    __slots__ = ("p", "a")

    def __init__(self, p: int, a: str | None = None):
        #: Residue position.
        self.p = p

        #: Atom name. If not provided, merge across all residue atoms.
        self.a = a

    @property
    def rtype(self) -> t.Type[str]:
        return str

    def calculate(self, obj: GenericStructure, mapping: MappingT | None = None) -> str:
        mask = (
            atom_mask(self.p, self.a, obj.array, mapping)
            if self.a is not None
            else residue_mask(self.p, obj.array, mapping)
        )
        with DefaultConfig.temporary_namespace():
            DefaultConfig["ligand"]["min_atom_connections"] = 1
            DefaultConfig["ligand"]["min_res_connections"] = 0
            names = []
            for lig in obj.ligands:
                try:
                    if lig.is_locally_connected(mask):
                        names.append(lig.res_name)
                except IndexError as e:
                    raise FailedCalculation(
                        f"Failed to apply obtained position mask derived from "
                        f"{obj} to a ligand {lig.res_name} "
                        f"({lig.parent}) parent contacts."
                    ) from e
        return ",".join(names)


class ClosestLigandDist(StructureVariable):
    """
    A distance from the selected residue or a residue's atom to a connected
    ligand.

    Each ligand provides :attr:`lXtractor.core.ligand.Ligand.dist` array.
    These arrays are stacked and aggregated atom-wise using :attr:`agg_lig`.
    Then, :attr:`agg_res` aggregates the obtained vector of values into a
    single number.

    For instance, to obtain max distance for the closest ligand of a residue 1,
    use ``ClosestLigandDist(1, agg_res='max')``.

    If structure has no
    :attr:`<ligands lXtractor.core.structure.GenericStructure.ligands>`,
    this variable defaults to -1.0.

    ..note ::
        Attr :attr:`lXtractor.core.ligand.dist` provides distances from an atom
        to the closest ligand atom.
    """

    __slots__ = ("p", "a", "agg_lig", "agg_res")

    def __init__(
        self, p: int, a: str | None = None, agg_lig: str = "min", agg_res: str = "min"
    ):
        if agg_lig not in AggFns:
            raise InitError(
                f"Wrong agg_lig {agg_lig}. " f"Available aggregators: {list(AggFns)}"
            )
        if agg_res not in AggFns:
            raise InitError(
                f"Wrong agg_lig {agg_res}. " f"Available aggregators: {list(AggFns)}"
            )

        #: Residue position
        self.p = p

        #: Atom name. If not provided, aggregate across residue atoms.
        self.a = a

        #: Aggregator function for ligands.
        self.agg_lig = agg_lig

        #: Aggregator function for a residue atoms.
        self.agg_res = agg_res

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
        self, obj: GenericStructure, mapping: MappingT | None = None
    ) -> float:
        mask = (
            atom_mask(self.p, self.a, obj.array, mapping)
            if self.a is not None
            else residue_mask(self.p, obj.array, mapping)
        )

        if not obj.ligands:
            return -1.0

        dists = []
        for lig in obj.ligands:
            try:
                dists.append(lig.dist[mask])
            except IndexError as e:
                raise FailedCalculation(
                    f"Failed to apply obtained position mask "
                    f"to a ligand's {lig} parent contacts."
                ) from e
        d = np.vstack(dists)
        d = AggFns[self.agg_lig](d, axis=0)
        d = AggFns[self.agg_res](d)
        return d


class Contacts(StructureVariable):
    """
    Uses `KDTree` to find atoms within the ``r`` distance threshold of those
    defined by target position ``p``. Positions these atoms correspond to
    are returned as a ","-separated string.

    If `mapping` is provided, contact positions will be filtered to those
    covered by this `mapping`.

    .. note::
        ``r`` is defined by ``DefaultConfig["contacts"]["non-covalent"][1]``.
    """

    __slots__ = ("p",)

    def __init__(self, p: int):
        self.p = p

    @property
    def rtype(self) -> t.Type[str]:
        return str

    def calculate(self, obj: GenericStructure, mapping: MappingT | None = None) -> str:
        a = obj.array
        m = residue_mask(self.p, obj.array, mapping)
        p = a[m][0].res_id
        r = DefaultConfig["bonds"]["NC-NC"][1]

        kdtree = KDTree(a.coord)
        hit_idx = np.unique(np.hstack(kdtree.query_ball_point(a[m].coord, r)))
        hit_pos = np.setdiff1d(np.unique(a[hit_idx].res_id), [p])

        if mapping is not None:
            m_rev = dict(zip(mapping.values(), mapping.keys()))
            hit_pos = list(
                filter(
                    lambda x: x is not None and not np.isnan(x),
                    (m_rev.get(p, None) for p in hit_pos),
                )
            )

        return ",".join(map(str, sorted(hit_pos)))


if __name__ == "__main__":
    raise RuntimeError
