from __future__ import annotations

import logging
import typing as t

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from lXtractor.core.exceptions import FailedCalculation

LOGGER = logging.getLogger(__name__)


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


# Chi1-Chi2 dihedral atoms: http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html


if __name__ == '__main__':
    raise RuntimeError
