import logging
import typing as t
from itertools import starmap, combinations

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from lXtractor.base import FailedCalculation, AminoAcidDict, _StrSep, FormatError, AbstractVariable, InputSeparators
from lXtractor.utils import split_validate
from more_itertools import unique_justseen

_ParsedVariables = t.Tuple[
    t.List[AbstractVariable],
    t.List[t.Optional[str]],
    t.List[t.Optional[str]]]
_FlattenedVariable = t.Tuple[
    AbstractVariable,
    t.Optional[str],
    t.Optional[str]
]
_Aggregators = {'min': np.min, 'max': np.max, 'mean': np.mean, 'median': np.median}
Sep = InputSeparators(',', ':', '::', '_')
LOGGER = logging.getLogger(__name__)


def _try_map(
        pos: int, mapping: t.Optional[t.Mapping[int, int]] = None
) -> int:
    if mapping is None:
        return pos
    try:
        return mapping[pos]
    except KeyError:
        raise FailedCalculation(
            f'Missing position {pos} in the provided mapping of the '
            f'alignment columns to the structure residues.'
        )


def _try_find_residue(
        pos: int, structure: Structure
) -> Residue:
    residues = list(filter(
        lambda r: r.get_id()[1] == pos,
        structure.get_residues()))
    if not residues:
        raise FailedCalculation(
            f'No residue {pos} in a given structure {structure.id}'
        )
    if len(residues) > 1:
        raise FailedCalculation(
            f'More than one {pos} within structure {structure.id}. '
            f'Full IDs: {[r.get_full_id() for r in residues]}'
        )
    return residues.pop()


def _try_find_atom(
        residue: Residue, atom_name: str
) -> Atom:
    try:
        return residue[atom_name]
    except KeyError:
        raise FailedCalculation(
            f'No atom {atom_name} in residue {residue.segid}-{residue.resname}')


def _get_coord(
        residue: Residue, atom: t.Optional[str]
) -> np.ndarray:
    if atom is None:
        coords = np.vstack([a.coord for a in residue.get_atoms()])
        return coords.mean(axis=0)
    return _try_find_atom(residue, atom).coord


class SeqEl(AbstractVariable):
    """
    Sequence element. A residue at some alignment position.
    """

    def __init__(self, aln_pos: int):
        """
        :param aln_pos: Position at the MSA.
        """
        self.aln_pos = aln_pos
        self.amino_acid_dict = AminoAcidDict()

    @property
    def id(self):
        return f'Sequence Element ({self.aln_pos})'

    def calculate(
            self, structure: Structure,
            mapping: t.Optional[t.Mapping[int, int]] = None
    ) -> str:
        pos = _try_map(self.aln_pos, mapping)
        res = _try_find_residue(pos, structure)
        resname = res.get_resname()
        return f'{pos}_{resname}_{self.amino_acid_dict[resname]}'


class Dist(AbstractVariable):
    def __init__(
            self, pos1: int, pos2: int,
            atom1: t.Optional[str] = None,
            atom2: t.Optional[str] = None,
            com: bool = False):
        self.pos1 = pos1
        self.pos2 = pos2
        self.atom1 = atom1
        self.atom2 = atom2
        self.com = com

        if any((not com and atom1 is None, not com and atom2 is None)):
            raise ValueError(
                'No atom name specified and "center of mass" flag is down. '
                'Therefore, not possible to calculate distance.')

    @property
    def id(self):
        atom1 = self.atom1 or 'com'
        atom2 = self.atom2 or 'com'
        return f'Distance {self.pos1}:{atom1}-{self.pos2}:{atom2}'

    def calculate(
            self, structure: Structure,
            mapping: t.Optional[t.Mapping[int, int]] = None
    ) -> float:
        pos1, pos2 = map(
            lambda p: _try_map(p, mapping),
            [self.pos1, self.pos2])
        res1, res2 = map(
            lambda p: _try_find_residue(p, structure),
            [pos1, pos2])

        xyz1, xyz2 = starmap(
            lambda r, a: _get_coord(r, a),
            [(res1, self.atom1), (res2, self.atom2)])

        return np.linalg.norm(xyz2 - xyz1)


def agg_dist(
        r1: Residue, r2: Residue,
        agg_fn: t.Callable[[np.ndarray], float]) -> float:
    """
    Calculate the aggregated distance between two residues

    :param r1: biopython's ``Residue`` object
    :param r2: biopython's ``Residue`` object
    :param agg_fn: callable accepting numerical numpy
        three-dimensional array and returning an aggregated
        distance measurement (min, max, mean, etc.).
    :return: aggregated distance between all combinations
        of atoms of two residues
    """
    cs1, cs2 = map(
        lambda r: np.vstack([a.coord for a in r.get_atoms()]),
        [r1, r2])
    return agg_fn(np.linalg.norm(cs1[:, np.newaxis] - cs2, axis=2))


class AggDist(AbstractVariable):
    def __init__(self, pos1: int, pos2: int, key: str = 'min'):
        if key not in _Aggregators:
            raise ValueError(
                f'Wrong key {key}. '
                f'Available aggregators: {list(_Aggregators)}')
        self._key_name = key
        self.key = _Aggregators[key]
        self.pos1 = pos1
        self.pos2 = pos2

    @property
    def id(self):
        return f'{self._key_name.capitalize()} distance {self.pos1}-{self.pos2}'

    def calculate(
            self, structure: Structure,
            mapping: t.Optional[t.Mapping[int, int]] = None
    ) -> t.Union[str, float]:
        pos1, pos2 = map(
            lambda p: _try_map(p, mapping),
            [self.pos1, self.pos2])
        res1, res2 = map(
            lambda p: _try_find_residue(p, structure),
            [pos1, pos2])
        return agg_dist(res1, res2, self.key)


class AllDist(AbstractVariable):
    def __init__(self, key: str = 'min'):
        if key not in _Aggregators:
            raise ValueError(
                f'Wrong key {key}. '
                f'Available aggregators: {list(_Aggregators)}')
        self._key_name = key
        self.key = _Aggregators[key]

    @property
    def id(self):
        return f'{self._key_name.capitalize()} ALL-ALL distance'

    def calculate(
            self, structure: Structure,
            mapping: t.Optional[t.Mapping[int, int]] = None
    ) -> t.List[t.Tuple[int, int, float]]:
        # residues = filter(
        #     lambda r: r.get_full_id()[-1][0] == ' ',
        #     structure.get_residues())
        residues = structure.get_residues()
        if mapping:
            residues = filter(
                lambda r: r.get_id()[1] in mapping.values(),
                residues)
        cs = combinations(residues, 2)
        ds = starmap(
            lambda r1, r2: (
                r1.get_id()[1],
                r2.get_id()[1],
                agg_dist(r1, r2, self.key)),
            cs)
        if mapping:
            m_rev = {v: k for k, v in mapping.items()}
            ds = starmap(
                lambda r1_id, r2_id, d: (
                    m_rev[r1_id], m_rev[r2_id], d),
                ds)
        return list(ds)


class Dihedral(AbstractVariable):
    def __init__(
            self,
            pos1: int, pos2: int, pos3: int, pos4: int,
            atom1: str, atom2: str, atom3: str, atom4: str,
            name: str = 'Dihedral',
            verify_consecutive: bool = True):
        self.name = name
        self.p1, self.p2, self.p3, self.p4 = pos1, pos2, pos3, pos4
        self.a1, self.a2, self.a3, self.a4 = atom1, atom2, atom3, atom4
        self.positions = [pos1, pos2, pos3, pos4]
        self.atoms = [atom1, atom2, atom3, atom4]
        self.verify_consecutive = verify_consecutive

    @property
    def id(self):
        return f'{self.name} {self.p1}:{self.a1}-{self.p2}:{self.a2}-' \
               f'{self.p3}:{self.a3}-{self.p4}:{self.a4}'

    @staticmethod
    def _verify_consecutive(positions: t.Iterable[int]) -> None:
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

    def calculate(
            self, structure: Structure,
            mapping: t.Optional[t.Mapping[int, int]] = None
    ) -> float:
        # Map positions to the PDB numbering
        positions = list(map(lambda p: _try_map(p, mapping), self.positions))
        LOGGER.debug(f'Mapped alignment positions {self.positions} to {positions}')

        # Verify the mapped positions are consecutive
        if self.verify_consecutive:
            self._verify_consecutive(positions)
            LOGGER.debug(f'Verified positions are consecutive')

        # Attempt to find residues at mapped positions
        residues = list(map(lambda p: _try_find_residue(p, structure), positions))
        LOGGER.debug(f'Found residues {residues} at positions {positions}')

        # Attempt to find atoms of interest and their coordinates
        # within the extracted residues
        atom_coords = list(starmap(
            lambda r, a: _try_find_atom(r, a).coord,
            zip(residues, self.atoms)))
        LOGGER.debug(
            f'Found the following atomic coordinates: '
            f'{[(a, c) for a, c in zip(self.atoms, atom_coords)]}')

        # Calculate the dihedral angle and return the result
        result = calculate_dihedral(*atom_coords)
        LOGGER.debug(f'Calculated dihedral {result}')

        return result


class PseudoDihedral(Dihedral):
    def __init__(
            self, pos1: int, pos2: int, pos3: int, pos4: int):
        super().__init__(
            pos1, pos2, pos3, pos4,
            'CA', 'CA', 'CA', 'CA',
            name='Pseudo Dihedral')


class Phi(Dihedral):
    def __init__(
            self, pos: int):
        super().__init__(
            pos - 1, pos, pos, pos,
            'C', 'N', 'CA', 'C',
            name='Phi')


class Psi(Dihedral):
    def __init__(
            self, pos: int):
        super().__init__(
            pos, pos, pos, pos + 1,
            'N', 'CA', 'C', 'N',
            name='Psi')


class Omega(Dihedral):
    def __init__(
            self, pos: int):
        super().__init__(
            pos, pos, pos + 1, pos + 1,
            'CA', 'C', 'N', 'CA',
            name='Psi')


class CompositeDihedral(AbstractVariable):
    def __init__(self, dihedrals: t.Sequence[Dihedral]):
        self.dihedrals = dihedrals

    def id(self):
        raise NotImplementedError

    def calculate(
            self, structure: Structure,
            mapping: t.Optional[t.Mapping[int, int]] = None
    ) -> float:
        res = None
        for dihedral in self.dihedrals:
            try:
                res = dihedral.calculate(structure, mapping)
                break
            except FailedCalculation:
                pass
        if res is None:
            raise FailedCalculation(
                f"Couldn't calculate any of dihedrals "
                f"{[d.id for d in self.dihedrals]}")
        return res


# Chi1-Chi2 dihedral atoms: http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html
class Chi1(CompositeDihedral):
    def __init__(self, pos: int):
        self.pos = pos
        dihedrals = [
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'CG', 'Chi1_CG'),
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'CG1', 'Chi1_CG1'),
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'OG', 'Chi1_OG'),
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'OG1', 'Chi1_OG1'),
            Dihedral(pos, pos, pos, pos, 'N', 'CA', 'CB', 'SG', 'Chi1_SG')
        ]
        super().__init__(dihedrals)

    def id(self):
        return f'Chi1 Dihedral (CG, CG1, OG, OG1, SG)'


class Chi2(CompositeDihedral):
    def __init__(self, pos: int):
        self.pos = pos
        dihedrals = [
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'CD', 'Chi2_CG-CD'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'OD1', 'Chi2_CG-OD1'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'ND1', 'Chi2_CG-ND1'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG1', 'CD', 'Chi2_CG1-CD'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'CD1', 'Chi2_CG-CD1'),
            Dihedral(pos, pos, pos, pos, 'CA', 'CB', 'CG', 'SD', 'Chi2_CG-SD'),
        ]
        super().__init__(dihedrals)

    def id(self):
        return f'Chi2 Dihedral (CG-CD,CG-OD1,CG-ND1,CG1-CD,CG-CD1,CG-SD)'


def calculate_dihedral(
        atom1: np.ndarray, atom2: np.ndarray,
        atom3: np.ndarray, atom4: np.ndarray
) -> float:
    """
    Calculate angle between planes formed by
    [atom1, atom2, atom3] and [atom2, atom3, atom4].

    Each atom is an array of shape (3, ) with XYZ coordinates.

    Calculation method inspired by
    https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    """
    atoms = [atom1, atom2, atom3, atom4]
    for a in atoms:
        if not a.shape == (3,):
            raise ValueError(
                f'Expected XYZ array, got {a} '
                f'with shape {a.shape} instead')
    for a1, a2 in combinations(atoms, 2):
        if np.all(a1 == a2):
            raise ValueError(
                f'Expected for distinct atoms, but {a1} == {a2}')

    v1, v2, v3 = map(
        lambda v: v / np.linalg.norm(v),
        [atom2 - atom1, atom3 - atom2, atom4 - atom3])

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    x = n1.dot(n2)
    y = np.cross(n1, n2).dot(v2)

    return np.arctan2(y, x)


def parse_variables(inp: str) -> _ParsedVariables:
    # TODO: link a complete description of variables syntax
    """
    Parse raw input into a collection of variables, structures, and levels
    at which they should be calculated.

    :param inp: ``"[variable_specs]--[protein_specs]::[domains]"`` format, where:

        #. `variable_specs` define the variable type
        (e.g., `1:CA-2:CA` for CA-CA distance between positoins 1 and 2)
        #. `protein_specs` define proteins for which to calculate variables
        #. `domains` list the domain names for the given protein collection

    :return: a namedtuple with (1) variables, (2) list of proteins (or ``[None]``),
        and (3) a list of domains (or ``[None]``).
    """
    if _StrSep in inp and Sep.dom in inp:
        variables, tmp = split_validate(inp, _StrSep, 2)
        proteins, domains = split_validate(tmp, Sep.dom, 2)
    elif _StrSep in inp:
        variables, proteins = split_validate(inp, _StrSep, 2)
        domains = None
    elif Sep.dom in inp:
        variables, domains = split_validate(inp, Sep.dom, 2)
        proteins = None
    else:
        variables = inp
        domains, proteins = None, None

    variables = list(map(dispatch_var, variables.split(',')))
    proteins = [None] if proteins is None else proteins.split(',')
    domains = [None] if domains is None else domains.split(',')

    LOGGER.debug(f'Split input {inp} into Variables={variables},'
                 f'Proteins={proteins},Domains={domains}')

    return variables, proteins, domains


def dispatch_var(var: str) -> AbstractVariable:
    """
    Convert a textual representation of a single variable
    into a concrete and initialized variable.

    >>> assert isinstance(dispatch_var('123'), SeqEl)
    >>> assert isinstance(dispatch_var('1-2'), Dist)
    >>> assert isinstance(dispatch_var('1-2-3-4'), PseudoDihedral)

    :param var: textual representation of a variable.
    :return: subclass of a :class:`AbstractVariable`
    """

    def split_pos(_pos: str) -> t.Tuple[int, t.Optional[str]]:
        _split = _pos.split(':')
        if len(_split) == 2:
            _pos, atom = _split
        elif len(_split) == 1:
            _pos, atom = _pos, None
        else:
            raise FormatError(
                f'Expected 1 element or 2 ":"-separated elements. '
                f'Got {var}')
        try:
            if _pos != 'ALL':
                _pos = int(_pos)
        except ValueError:
            raise FormatError(
                f'Non-convertable position {_pos} in {var}')
        return _pos, atom

    split = var.split('-')

    if len(split) == 1:
        LOGGER.debug(f'Recieved one-position variable {split}')

        # If the position is simply a digit, we expect
        # the variable to be a sequence element.
        if var.isdigit():
            variable = SeqEl(int(var))
        else:
            # Otherwise, it is of the format Pos_[Phi/Psi/Omega]
            pos, angle = split_validate(var, '_', 2)
            LOGGER.debug(f'Expecting angle {angle} at position {pos}')

            if not pos.isdigit():
                raise FormatError(
                    f'Non-convertable position {pos} in {var}')

            pos = int(pos)
            if angle == 'Phi':
                variable = Phi(pos)
            elif angle == 'Psi':
                variable = Psi(pos)
            elif angle == 'Omega':
                variable = Omega(pos)
            else:
                raise FormatError(
                    f'Wrong angle name {angle} in {var}. '
                    f'Expected one of: Phi, Psi, Omega')

    elif len(split) == 2:
        LOGGER.debug(f'Recieved two positions {split} '
                     f'to init the "distance" variable')

        (pos1, atom1), (pos2, atom2) = map(split_pos, split)

        LOGGER.debug(f'Split the first position into {pos1}:{atom1}, '
                     f'and the second position into {pos2}:{atom2}')

        if pos1 == pos2 == 'ALL':
            key = atom2 or 'min'
            variable = AllDist(key)
        elif atom1 is None and atom2 in _Aggregators:
            variable = AggDist(pos1, pos2, key=atom2)
        else:
            com = atom1 is None or atom2 is None
            variable = Dist(pos1, pos2, atom1, atom2, com=com)

    elif len(split) == 4:
        LOGGER.debug(f'Received four positions {split} '
                     f'to init "dihedral"-type variable')
        # Split each of the positions into position-atom pairs, where atom
        # may be `None`.
        (pos1, atom1), (pos2, atom2), (pos3, atom3), (pos4, atom4) = map(
            split_pos, split)
        atoms = [atom1, atom2, atom3, atom4]
        positions = [pos1, pos2, pos3, pos4]
        LOGGER.debug(f'Split each of the position into the following pos-atom pairs: '
                     f'{list(zip(positions, atoms))}')

        # Variables of the form 1-2-3-4 or 1-5-10-20 are considered pseudo-dihedrals
        if all(a is None for a in atoms) and len(set(positions)) == 4:
            # We don't check whether the positions are consecutive
            # since seqs columns aren't expected to be.
            # When they are mapped back to the structure though,
            # they should be consecutive.
            variable = PseudoDihedral(*positions)
        # Otherwise, it is a custom dihedral angle.
        elif all(a is not None for a in atoms):
            variable = Dihedral(*positions, *atoms)
        else:
            raise FormatError(
                f'Incorrect dihedral specification. '
                f'Four distinct positions (columns) specify PseudoDihedral. '
                f'Four positions with atoms specify generic Dihedral. '
                f'Got none of these variants in {var}')
    else:
        raise FormatError(
            f'Wrong number of "-"-separated positions in variable {var}. '
            f'Allowed numbers are: 1, 2, 4')

    LOGGER.debug(f'Initialized {variable.id} variable')
    return variable


if __name__ == '__main__':
    raise RuntimeError
