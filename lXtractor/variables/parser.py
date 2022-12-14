from __future__ import annotations

import logging
import typing as t

from lXtractor.core.config import Sep
from lXtractor.core.exceptions import FormatError
from lXtractor.util.misc import split_validate
from lXtractor.variables.base import AggFns
from lXtractor.variables.sequential import SeqEl
from lXtractor.variables.structural import Dist, AggDist, Dihedral, PseudoDihedral, Phi, Psi, Omega

LOGGER = logging.getLogger(__name__)


def parse_var(inp: str):
    # TODO: link a complete description of variables syntax
    """
    Parse raw input into a collection of variables, structures, and levels
    at which they should be calculated.

    :param inp: ``"[variable_specs]--[protein_specs]::[domains]"`` format, where:

        - `variable_specs` define the variable type
            (e.g., `1:CA-2:CA` for CA-CA distance between positions 1 and 2)
        - `protein_specs` define proteins for which to calculate variables
        - `domains` list the domain names for the given protein collection
    :return: a namedtuple with (1) variables, (2) list of proteins (or ``[None]``),
        and (3) a list of domains (or ``[None]``).
    """
    if Sep.str in inp and Sep.dom in inp:
        variables, tmp = split_validate(inp, Sep.str, 2)
        proteins, domains = split_validate(tmp, Sep.dom, 2)
    elif Sep.str in inp:
        variables, proteins = split_validate(inp, Sep.str, 2)
        domains = None
    elif Sep.dom in inp:
        variables, domains = split_validate(inp, Sep.dom, 2)
        proteins = None
    else:
        variables = inp
        domains, proteins = None, None

    variables = list(map(init_var, variables.split(',')))
    if proteins is not None:
        proteins = proteins.split(Sep.list)
    if domains is not None:
        domains = domains.split(Sep.list)

    LOGGER.debug(f'Split input {inp} into Variables={variables},'
                 f'Proteins={proteins},Domains={domains}')

    return variables, proteins, domains


def init_var(var: str):
    """
    Convert a textual representation of a single variable
    into a concrete and initialized variable.

    >>> assert isinstance(init_var('123'), SeqEl)
    >>> assert isinstance(init_var('1-2'), Dist)
    >>> assert isinstance(init_var('1-2-3-4'), PseudoDihedral)

    :param var: textual representation of a variable.
    :return: initialized variable, a concrete subclass of an :class:`AbstractVariable`
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
        LOGGER.debug(f'Received one-position variable {split}')

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
        LOGGER.debug(f'Received two positions {split} '
                     f'to init the "distance" variable')

        (pos1, atom1), (pos2, atom2) = map(split_pos, split)

        LOGGER.debug(f'Split the first position into {pos1}:{atom1}, '
                     f'and the second position into {pos2}:{atom2}')

        if pos1 == pos2 == 'ALL':
            key = atom2 or 'min'
            variable = AllDist(key)
        elif atom1 is None and atom2 in AggFns:
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
            # since _seqs columns aren't expected to be.
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
