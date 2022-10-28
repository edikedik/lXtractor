from __future__ import annotations

import logging
import typing as t
from abc import abstractmethod
from collections import abc
from itertools import starmap

import biotite.structure as bst
import numpy as np
from more_itertools import unique_justseen
from toolz import curry, pipe

from lXtractor.core.exceptions import FailedCalculation, InitError
from lXtractor.util.structure import calculate_dihedral
from lXtractor.variables.base import StructureVariable, AggFns, MappingT

LOGGER = logging.getLogger(__name__)


@curry
def _map_pos(pos: int, mapping: t.Optional[MappingT] = None) -> int:
    if mapping is None:
        return pos
    try:
        return mapping[pos]
    except KeyError:
        raise FailedCalculation(f'Missing position {pos} in the provided mapping.')


@curry
def _get_residue_mask(
        pos: int, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
) -> np.ndarray:
    pos = _map_pos(pos, mapping)

    mask = np.equal(array.res_id, pos)

    residue_atoms = array[mask]
    if not residue_atoms.array_length():
        raise FailedCalculation(f'Missing position {pos}')

    num_starts = bst.get_residue_starts(residue_atoms)
    if len(num_starts) > 1:
        raise FailedCalculation(f'Position {pos} points to {len(num_starts)}>1 residues')

    return mask


@curry
def _get_residue(
        pos: int, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
) -> bst.AtomArray:
    mask = _get_residue_mask(pos, array, mapping)
    return array[mask]


@curry
def _get_atom(array: bst.AtomArray, name: str) -> bst.Atom:
    atom = array[array.atom_name == name]

    size = atom.array_length()

    if not size:
        raise FailedCalculation(f'Missing atom {name}')

    if size > 1:
        raise FailedCalculation(f'Non-unique atom with name {name}')

    return atom[0]


@curry
def _get_coord(
        residue: bst.AtomArray,
        atom_name: t.Optional[str],
        agg_fn: t.Optional[str] = 'mean'
) -> np.ndarray:
    if atom_name is None:
        fn = AggFns[agg_fn]
        return fn(residue.coord, axis=0)

    return _get_atom(residue, atom_name).coord


def _agg_dist(
        r1: bst.AtomArray, r2: bst.AtomArray,
        agg_fn: t.Callable[[np.ndarray], float]) -> float:
    """
    Calculate the aggregated distance between two residues
    """
    return agg_fn(np.linalg.norm(r1.coord[:, np.newaxis] - r2.coord, axis=2))


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
                f'Positions {previous} and {current} are not consecutive '
                f'in a given list {positions}')


class Dist(StructureVariable):
    __slots__ = ('p1', 'p2', 'a1', 'a2', 'com')

    def __init__(
            self, p1: int, p2: int,
            a1: t.Optional[str] = None,
            a2: t.Optional[str] = None,
            com: bool = False):
        self.p1 = p1
        self.p2 = p2
        self.a1 = a1
        self.a2 = a2
        self.com = com

        if any((not com and a1 is None, not com and a2 is None)):
            raise ValueError(
                'No atom name specified and "center of mass" flag is down. '
                'Therefore, not possible to calculate distance.')

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
            self, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
    ) -> float:
        xyz1, xyz2 = starmap(
            lambda p, a:
            pipe(
                _get_residue(p, array, mapping),
                _get_coord(atom_name=a)
            ),
            [(self.p1, self.a1), (self.p2, self.a2)]
        )
        return np.linalg.norm(xyz2 - xyz1)


class AggDist(StructureVariable):
    __slots__ = ('p1', 'p2', 'key')

    def __init__(self, p1: int, p2: int, key: str = 'min'):
        if key not in AggFns:
            raise InitError(
                f'Wrong key {key}. '
                f'Available aggregators: {list(AggFns)}')
        self.key = key
        self.p1 = p1
        self.p2 = p2

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
            self, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
    ) -> float:
        res1, res2 = map(
            lambda p: _get_residue(p, array, mapping),
            [self.p1, self.p2])
        return _agg_dist(res1, res2, AggFns[self.key])


class AllDist(StructureVariable):
    __slots__ = ('key',)

    def __init__(self, key: str = 'min'):
        if key not in AggFns:
            raise ValueError(
                f'Wrong key {key}. '
                f'Available aggregators: {list(AggFns)}')
        self.key = key

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
            self, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
    ) -> t.List[t.Tuple[int, int, float]]:
        raise NotImplementedError

        # residues = array.get_residues()
        # if mapping:
        #     residues = filter(
        #         lambda r: r.get_id()[1] in mapping.values(),
        #         residues)
        # cs = combinations(residues, 2)
        # ds = starmap(
        #     lambda r1, r2: (
        #         r1.get_id()[1],
        #         r2.get_id()[1],
        #         agg_dist(r1, r2, AggFns[self.key])),
        #     cs)
        # if mapping:
        #     m_rev = {v: k for k, v in mapping.items()}
        #     ds = starmap(
        #         lambda r1_id, r2_id, d: (
        #             m_rev[r1_id], m_rev[r2_id], d),
        #         ds)
        # return list(ds)


class Dihedral(StructureVariable):
    __slots__ = ('p1', 'p2', 'p3', 'p4', 'a1', 'a2', 'a3', 'a4', 'name')

    def __init__(
            self,
            p1: int, p2: int, p3: int, p4: int,
            a1: str, a2: str, a3: str, a4: str,
            name: str = 'Dihedral'):
        self.name = name
        self.p1, self.p2, self.p3, self.p4 = p1, p2, p3, p4
        self.a1, self.a2, self.a3, self.a4 = a1, a2, a3, a4

    @property
    def rtype(self) -> t.Type[float]:
        return float

    @property
    def positions(self) -> list[int]:
        return [self.p1, self.p2, self.p3, self.p4]

    @property
    def atoms(self) -> list[str]:
        return [self.a1, self.a2, self.a3, self.a4]

    def calculate(
            self, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
    ) -> float:
        # Map positions to the PDB numbering
        coordinates = starmap(
            lambda p, a: pipe(
                _get_residue(p, array, mapping),
                _get_coord(atom_name=a)
            ),
            zip(self.positions, self.atoms)
        )
        return calculate_dihedral(*coordinates)


class PseudoDihedral(Dihedral):
    __slots__ = ()

    def __init__(self, p1: int, p2: int, p3: int, p4: int):
        super().__init__(
            p1, p2, p3, p4,
            'CA', 'CA', 'CA', 'CA',
            name='Pseudo Dihedral')


class Phi(Dihedral):
    __slots__ = ('p',)

    def __init__(self, p: int):
        self.p = p
        super().__init__(
            p - 1, p, p, p,
            'C', 'N', 'CA', 'C',
            name='Phi')


class Psi(Dihedral):
    __slots__ = ('p',)

    def __init__(self, p: int):
        self.p = p
        super().__init__(
            p, p, p, p + 1,
            'N', 'CA', 'C', 'N',
            name='Psi')


class Omega(Dihedral):
    __slots__ = ('p',)

    def __init__(self, p: int):
        self.p = p
        super().__init__(
            p, p, p + 1, p + 1,
            'CA', 'C', 'N', 'CA',
            name='Omega')


class CompositeDihedral(StructureVariable):
    __slots__ = ('p',)

    def __init__(self, p: int):
        self.p = p

    @property
    def rtype(self) -> t.Type[float]:
        return float

    @staticmethod
    @abstractmethod
    def get_dihedrals(pos: int) -> abc.Iterable[Dihedral]:
        raise NotImplementedError('Must be implemented by the subclass')

    def calculate(
            self, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
    ) -> float:
        res = None
        dihedrals = self.get_dihedrals(self.p)
        for d in dihedrals:
            try:
                res = d.calculate(array, mapping)
                break
            except FailedCalculation:
                pass
        if res is None:
            raise FailedCalculation(
                f"Couldn't calculate any of dihedrals "
                f"{[d.id for d in dihedrals]}")
        return res


class Chi1(CompositeDihedral):
    __slots__ = ()

    @staticmethod
    def get_dihedrals(pos) -> list[Dihedral]:
        return [
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'CG', 'Chi1_CG'),
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'CG1', 'Chi1_CG1'),
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'OG', 'Chi1_OG'),
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'OG1', 'Chi1_OG1'),
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'SG', 'Chi1_SG')
        ]


class Chi2(CompositeDihedral):
    __slots__ = ()

    @staticmethod
    def get_dihedrals(pos) -> list[Dihedral]:
        return [
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'CD', 'Chi2_CG-CD'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'OD1', 'Chi2_CG-OD1'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'ND1', 'Chi2_CG-ND1'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG1', 'CD', 'Chi2_CG1-CD'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'CD1', 'Chi2_CG-CD1'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'SD', 'Chi2_CG-SD'),
        ]


class SASA(StructureVariable):
    __slots__ = ('p', 'a')

    def __init__(self, p: int, a: str | None = None):
        self.p = p
        self.a = a

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
            self, array: bst.AtomArray, mapping: t.Optional[MappingT] = None
    ) -> float | None:
        m = _get_residue_mask(self.p, array, mapping)

        if self.a is not None:
            m &= (array.atom_name == self.a)

        if m.sum() == 0:
            raise FailedCalculation('Empty selection')

        sasa = bst.sasa(array, atom_filter=m)
        return float(np.sum(sasa[~np.isnan(sasa)]))


if __name__ == '__main__':
    raise RuntimeError
