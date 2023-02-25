from __future__ import annotations

import typing as t

import biotite.structure as bst
import numpy as np

from lXtractor.core.base import BondThresholds, DefaultBondThresholds
from lXtractor.core.config import MetaNames
from lXtractor.core.exceptions import FormatError
from lXtractor.util.structure import filter_ligand, iter_residue_masks

if t.TYPE_CHECKING:
    from lXtractor.core.structure import GenericStructure


class Ligand:
    """
    Ligand object is a part of the structure falling under certain criteria.
    Namely, ligand molecule is a non-polymer and non-solvent molecule or a
    single monomer (thus, standalone amino acids can be ligands, while peptides
    of length >= 2 are not).

    ..seealso ::
        `find_ligands`
    """

    def __init__(
        self,
        name: str,
        parent: GenericStructure,
        parent_ligand_mask: np.ndarray,
        parent_contact_mask: np.ndarray,
        parent_contacts: np.ndarray,
        ligand_idx: np.ndarray,
        dist: np.ndaray,
        meta: dict[str, str] | None = None,
    ):
        if not (len(parent) == len(parent_ligand_mask) == len(parent_contact_mask)):
            raise FormatError(
                'The number of atoms in parent, the mask size and parent_contacts size '
                f'must all have the same len. Got {len(parent)}, '
                f'{len(parent_ligand_mask)}, and {len(parent_contact_mask)}, resp.'
            )
        if not (len(parent_contacts) == len(ligand_idx) == len(dist)):
            raise FormatError(
                'The number of contact atoms, ligand indices and distances must match. '
                f'Got {len(parent_contacts)}, {len(parent_contacts)}, and {len(dist)}, '
                'resp.'
            )
        if len(parent_contacts[parent_contacts != 0]) == 0:
            raise FormatError(
                'Ligand must have at least one contact atom in parent structure. Got 0.'
            )

        #: Name of the ligand. Defaults to the ligand residue PDB code.
        self.name: str = name

        #: Parent structure.
        self.parent: GenericStructure = parent

        #: A boolean mask such that when applied to the parent, subsets the
        #: latter to the ligand residues.
        self.parent_ligand_mask = parent_ligand_mask

        #: A boolean mask such that when applied to the parent, subsets the
        #: latter to its ligand-contacting atoms.
        self.parent_contact_mask: np.ndarray = parent_contact_mask

        self.parent_contacts = parent_contacts
        self.ligand_idx = ligand_idx
        self.dist = dist
        self.meta = meta

    @property
    def parent_contact_atoms(self) -> bst.AtomArray:
        return self.parent.array[self.parent_contact_mask]

    @property
    def ligand_atoms(self) -> bst.AtomArray:
        return self.parent.array[self.parent_ligand_mask]

    @property
    def parent_contact_chains(self) -> set[str]:
        return set(self.parent_contact_atoms.chain_id)


def find_ligands(
    structure: GenericStructure, ts: BondThresholds = DefaultBondThresholds
):
    a = structure.array
    is_ligand = filter_ligand(a)

    if is_ligand.sum() == 0:
        return

    for m_res in iter_residue_masks(a):
        m_ligand = is_ligand & m_res

        if not np.any(m_ligand):
            continue

        # An MxL matrix where L is the number of atoms in the structure and M is the
        # number of atoms in the ligand residue
        d = np.linalg.norm(a[m_ligand].coord[:, np.newaxis] - a.coord, axis=-1)
        d_min = np.min(d, axis=0)  # min distance from ligand atoms to structure
        d_argmin = np.argmin(d, axis=0)  # ligand atom indices contacting structure

        contacts = np.zeros_like(d_min)
        contacts[
            (d_min >= ts.non_covalent.lower) & (d_min <= ts.non_covalent.upper)
        ] = 1
        contacts[(d_min >= ts.covalent.lower) & (d_min <= ts.covalent.upper)] = 2
        contacts[m_ligand] = 0
        m_contacts = contacts != 0

        name = a[m_res].res_name[0]
        meta = {
            MetaNames.res_name: name,
            MetaNames.res_id: str(a[m_res].res_id[0]),
            MetaNames.pdb_chain: a[m_res].chain_id[0],
        }

        yield Ligand(
            name,
            structure,
            m_ligand,
            m_contacts,
            contacts[m_contacts],
            d_argmin[m_contacts],
            d_min[m_contacts],
            meta,
        )


if __name__ == '__main__':
    raise RuntimeError
