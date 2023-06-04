from __future__ import annotations

import typing as t
from collections import abc

import biotite.structure as bst
import numpy as np

from lXtractor.core.config import MetaNames, LigandConfig
from lXtractor.core.exceptions import FormatError
from lXtractor.util.structure import (
    filter_ligand,
    iter_residue_masks,
    find_contacts,
    filter_solvent_extended,
)

if t.TYPE_CHECKING:
    from lXtractor.core.structure import GenericStructure


class Ligand:
    """
    Ligand object is a part of the structure falling under certain criteria.

    Namely, a ligand is a non-polymer and non-solvent molecule or a single
    monomer (thus, standalone amino acids are considered ligands, while peptides
    of length >= 2 are not).

    Attributes :attr:`mask` and :attr:`contact_mask` are
    boolean masks allowing to obtain ligand and ligand-contacting atoms from
    :attr:`parent`.

    All array-type attributes, have the number of elements equal to the number
    of atoms in :attr:`parent`.

    .. seealso::
        `find_ligands`

    Methods ``__repr__`` and ``__str__`` output a string in the format:
    ``{res_name}_{res_id}:{chain_id}<-({parent})``.
    """

    def __init__(
        self,
        parent: GenericStructure,
        mask: np.ndarray,
        contact_mask: np.ndarray,
        parent_contacts: np.ndarray,
        ligand_idx: np.ndarray,
        dist: np.ndaray,
        meta: dict[str, str] | None = None,
    ):
        if not len(parent) == len(mask) == len(contact_mask):
            raise FormatError(
                "The number of atoms in parent, the mask size and parent_contacts size "
                f"must all have the same len. Got {len(parent)}, "
                f"{len(mask)}, and {len(contact_mask)}, resp."
            )
        if not len(parent_contacts) == len(ligand_idx) == len(dist):
            raise FormatError(
                "The number of contact atoms, ligand indices and distances must match. "
                f"Got {len(parent_contacts)}, {len(parent_contacts)}, and {len(dist)}, "
                "resp."
            )
        if len(parent_contacts[parent_contacts != 0]) == 0:
            raise FormatError(
                "Ligand must have at least one contact atom in parent structure. Got 0."
            )
        ligand_atoms = parent.array[mask]
        ligand_chains = set(ligand_atoms.chain_id)
        ligand_resnames = set(ligand_atoms.res_name)
        ligand_res_ids = set(ligand_atoms.res_id)
        if len(ligand_chains) > 1:
            raise FormatError(
                f"Ligand atoms point to more than one chain: {ligand_chains}"
            )
        if len(ligand_resnames) > 1:
            raise FormatError(
                f"Ligand atoms point to more than one ligand res name {ligand_resnames}"
            )
        if len(ligand_res_ids) > 1:
            raise FormatError(
                f"Ligand atoms point to more than one ligand res id {ligand_res_ids}"
            )

        #: Parent structure.
        self.parent: GenericStructure = parent

        #: A boolean mask such that when applied to the parent, subsets the
        #: latter to the ligand residues.
        self.mask = mask

        #: A boolean mask such that when applied to the parent, subsets the
        #: latter to its ligand-contacting atoms.
        self.contact_mask: np.ndarray = contact_mask

        #: An integer array where numbers indicate contact types.
        #: ``1`` signifies a non-covalent contact, ``2`` -- a covalent one.
        self.parent_contacts = parent_contacts

        #: An integer array with indices pointing to ligand atoms contacting
        #: the parent structure.
        self.ligand_idx = ligand_idx

        #: An array of distances for each ligand-contacting parent's atom.
        self.dist = dist

        #: A dictionary of meta info.
        self.meta = meta

    def __str__(self):
        return f"{self.res_name}_{self.res_id}:{self.chain_id}<-({self.parent})"

    def __repr__(self):
        return str(self)

    @property
    def array(self) -> bst.AtomArray:
        """
        :return: An array of ligand atoms within :attr:`parent`.
        """
        return self.parent.array[self.mask]

    @property
    def parent_contact_atoms(self) -> bst.AtomArray:
        """
        :return: An array of ligand-contacting atoms within :attr:`parent`.
        """
        return self.parent.array[self.contact_mask]

    @property
    def parent_contact_chains(self) -> set[str]:
        """
        :return: A set of chain IDs involved in forming contacts with ligand.
        """
        return set(self.parent_contact_atoms.chain_id)

    @property
    def chain_id(self) -> str:
        """
        :return: Ligand chain ID.
        """
        return self.array.chain_id[0]

    @property
    def res_name(self) -> str:
        """
        :return: Ligand residue name.
        """
        return self.array.res_name[0]

    @property
    def res_id(self) -> int:
        """
        :return: Ligand residue number.
        """
        return self.array.res_id[0]

    def is_locally_connected(self, mask: np.ndarray, min_connections: int = 1) -> bool:
        """
        Check whether this ligand is connected to a subset of parent atoms.

        :param mask: A boolean mask to filter parent atoms.
        :param min_connections: A minimum number of connections.
        :return: ``True`` if the ligand has at least `min_connections` to
            :attr:`parent` substructure imposed by the provided `mask`.
        """
        counts = np.bincount(self.parent_contacts[mask])
        if len(counts) == 1:
            return False
        return counts[1:].sum() >= min_connections


def find_ligands(
    structure: GenericStructure,
    cfg: LigandConfig = LigandConfig(),
) -> abc.Generator[Ligand, None, None]:
    """
    Find ligands within the structure. It divides all `structure` into a ligand
    part and non-ligand part. Namely, the ligand part comprises all non-solvent
    residues, while residues of any macromolecular polymeric entity make up for
    a non-ligand part. Then, for each residue within the "ligand part", it
    calculates the distance to the atoms of the "non-ligand part." Finally,
    finding contacts depends on the threshold values specified by `ts`.

    .. seealso::
        :func:`lXtractor.util.structure.filter_ligand`
        :func:`lXtractor.util.structure.filter_solvent_extended`
        :func:`lXtractor.util.structure.filter_any_polymer`
        :func:`lXtractor.util.structure.find_contacts`

    :param structure: Arbitrary (generic) structure.
    :param cfg: Ligand detection config.
    :return: A generator of :class:`Ligand` objects.
    """
    a = structure.array
    is_ligand = filter_ligand(a)
    is_solvent = filter_solvent_extended(a)

    if is_ligand.sum() == 0:
        return

    for m_res in iter_residue_masks(a):
        m_ligand = is_ligand & m_res

        if np.sum(m_ligand) < cfg.min_atoms:
            continue

        contacts, dist, ligand_idx = find_contacts(a, m_ligand, cfg.bond_thresholds)
        contacts[is_ligand | is_solvent] = 0
        m_contacts = contacts != 0

        if np.sum(contacts > 0) < cfg.min_connections:
            continue

        name = a[m_res].res_name[0]
        meta = {
            MetaNames.res_name: name,
            MetaNames.res_id: str(a[m_res].res_id[0]),
            MetaNames.pdb_chain: a[m_res].chain_id[0],
        }

        yield Ligand(structure, m_ligand, m_contacts, contacts, ligand_idx, dist, meta)


if __name__ == "__main__":
    raise RuntimeError
