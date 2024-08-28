from __future__ import annotations

import logging
import typing as t
from collections import abc

import numpy as np
import pandas as pd
import rustworkx as rx
from biotite import structure as bst
from numpy import typing as npt

import lXtractor.util as util
from lXtractor.core.config import DefaultConfig, AtomMark, POL_MARKS
from lXtractor.core.exceptions import FormatError, LengthMismatch


if t.TYPE_CHECKING:
    from lXtractor.core.structure import GenericStructure


LOGGER = logging.getLogger(__name__)


class Ligand:
    """
    Ligand object is a part of the structure falling under certain criteria.

    Namely, a ligand is a non-polymer and non-solvent molecule or a single
    monomer. Such ligands will be designated using the format::

        {res_name}_{res_id}:{chain_id}<-({parent})

    If a ligand contains multiple monomers, by convention, this is a polymer ligand.
    Such ligands should be named using the first letter of the polymer type;
    one of the ``("p", "n", "c")``. In this case, it's ID will be of the following
    format::

        {polymer_type}_{min_res_id}-{max_res_id}:{chain_id}<-({parent})

    This information is provided by :attr:`meta` and shouldn't be changed. However,
    any additional fields can be stored in :attr:`meta` which will be retrieved
    when constructing :meth:`summary`.

    Attributes :attr:`mask` and :attr:`contact_mask` are
    boolean masks allowing to obtain ligand and ligand-contacting atoms from
    :attr:`parent`.

    ..seealso ::
        :func:`make_ligand` to initialize a new ligand in an easy way.
    """

    __slots__ = (
        "is_polymer",
        "parent",
        "mask",
        "contact_mask",
        "ligand_idx",
        "dist",
        "meta",
    )

    def __init__(
        self,
        parent: GenericStructure,
        mask: np.ndarray,
        contact_mask: np.ndarray,
        ligand_idx: np.ndarray,
        dist: np.ndarray,
        meta: dict[str, str] | None = None,
    ):
        sizes = [
            ("parent", len(parent)),
            ("mask", len(mask)),
            ("contact_mask", len(contact_mask)),
            ("ligand_idx", len(ligand_idx)),
            ("dist", len(dist)),
        ]
        if len(set(x[1] for x in sizes)) != 1:
            raise LengthMismatch(
                f"The sizes of all input arrays must match. Got {sizes}"
            )

        names = DefaultConfig["metadata"]
        for k in [names["res_name"], names["res_id"], names["structure_chain_id"]]:
            if k not in meta:
                raise KeyError(f"Missing required key {k} in meta.")
        ligand_atoms = parent.array[mask]
        ligand_chains = set(ligand_atoms.chain_id)
        ligand_resnames = set(ligand_atoms.res_name)
        ligand_res_ids = set(ligand_atoms.res_id)
        if len(ligand_chains) > 1:
            raise FormatError(
                f"Ligand atoms point to more than one chain: {ligand_chains}"
            )
        self.is_polymer = len(ligand_resnames) > 1 or len(ligand_res_ids) > 1

        #: Parent structure.
        self.parent: GenericStructure = parent

        #: A boolean mask such that when applied to the parent, subsets the
        #: latter to the ligand residues.
        self.mask = mask

        #: A boolean mask such that when applied to the parent, subsets the
        #: latter to its ligand-contacting atoms.
        self.contact_mask: np.ndarray = contact_mask

        #: An integer array with indices pointing to ligand atoms contacting
        #: the parent structure.
        self.ligand_idx = ligand_idx

        #: An array of distances for each ligand-contacting parent's atom.
        self.dist = dist

        #: A dictionary of meta info.
        self.meta = meta

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, Ligand):
            return False
        return (
            self.id == other.id
            and np.all(self.mask == other.mask)
            and np.all(self.contact_mask == other.contact_mask)
            and np.all(self.ligand_idx == other.ligand_idx)
            and util.compare_arrays(self.dist, other.dist)
        )

    @property
    def id(self) -> str:
        return f"{self.res_name}_{self.res_id}:{self.chain_id}<-({self.parent})"

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
        return self.meta[DefaultConfig["metadata"]["structure_chain_id"]]

    @property
    def res_name(self) -> str:
        """
        :return: Ligand residue name.
        """
        return self.meta[DefaultConfig["metadata"]["res_name"]]

    @property
    def res_id(self) -> str:
        """
        :return: Ligand residue number.
        """
        return self.meta[DefaultConfig["metadata"]["res_id"]]

    def is_locally_connected(self, mask: np.ndarray) -> bool:
        """
        Check whether this ligand is connected to a subset of parent atoms.

        :param mask: A boolean mask to filter parent atoms.
        :return: ``True`` if the ligand has at least `min_atom_connections` to
            :attr:`parent` substructure imposed by the provided `mask`.
        """
        contact_atoms = self.parent.array[mask & self.contact_mask]
        cfg = DefaultConfig["ligand"]
        return (
            len(contact_atoms) >= cfg["min_atom_connections"]
            and bst.get_residue_count(contact_atoms) >= cfg["min_res_connections"]
        )

    def summary(self, meta: bool = True) -> pd.Series:
        d = self.meta if meta else {}
        d["ObjectID"] = self.id
        d["ParentID"] = self.parent.id
        return pd.Series(d.values(), index=d.keys())


# def find_ligands(
#     structure: GenericStructure,
# ) -> abc.Generator[Ligand, None, None]:
#     """
#     Find ligands within the structure. It divides all `structure` into a ligand
#     part and non-ligand part. Namely, the ligand part comprises all non-solvent
#     residues, while residues of any macromolecular polymeric entity make up for
#     a non-ligand part. Then, for each residue within the "ligand part", it
#     calculates the distance to the atoms of the "non-ligand part."
#
#     Finally, a discovered ligand is retained if it has:
#
#     #. Enough atoms.
#     #. Enough connected structure atoms.
#     #. Enough connected structure residues.
#
#     What "enough" means is defined by the supplied `cfg`.
#
#     .. seealso::
#         :func:`lXtractor.util.structure.filter_ligand`
#         :func:`lXtractor.util.structure.filter_solvent_extended`
#         :func:`lXtractor.util.structure.filter_any_polymer`
#         :func:`lXtractor.util.structure.find_contacts`
#         :class:`lXtractor.core.config.LigandConfig`
#
#     :param structure: Arbitrary (generic) structure.
#     :param cfg: Ligand detection config.
#     :return: A generator of :class:`Ligand` objects.
#     """
#     a, cfg = structure.array, DefaultConfig["ligand"]
#     is_ligand = filter_ligand(a)
#     is_solvent = filter_solvent_extended(a)
#
#     if is_ligand.sum() == 0:
#         return
#
#     # Iter over all residues in a structure
#     for m_res in iter_residue_masks(a):
#         # A mask that is a single ligand residue
#         m_ligand = is_ligand & m_res
#
#         # Check whether ligand has enough atoms
#         if np.sum(m_ligand) < cfg.min_atoms:
#             continue
#
#         contacts, dist, ligand_idx = find_contacts(a, m_ligand, cfg.bond_thresholds)
#         parent_contacts = contacts.copy()
#         parent_contacts[is_ligand | is_solvent] = 0
#         m_contacts = parent_contacts != 0
#
#         # The number of residues connected to a ligand
#         num_residues = bst.get_residue_count(a[m_contacts])
#
#         if (
#             np.sum(m_contacts) < cfg.min_atom_connections
#             or num_residues < cfg.min_res_connections
#         ):
#             continue
#
#         name = a[m_res].res_name[0]
#         meta = {
#             MetaNames.res_name: name,
#             MetaNames.res_id: str(a[m_res].res_id[0]),
#             MetaNames.structure_chain_id: a[m_res].chain_id[0],
#         }
#
#         yield Ligand(structure, m_ligand, m_contacts, contacts, ligand_idx, dist, meta)


def make_ligand(
    m_lig: npt.NDArray[np.bool_],
    m_pol: npt.NDArray[np.bool_],
    structure: GenericStructure,
) -> Ligand | None:
    """
    Create a new :class:`Ligand` object. The criteria to qualify for a ligand
    are defined by the global config (``DefaultConfig["ligand"]``).

    Whether a ligand molecule is created is subject to several checks::

        #. It has a certain number of atoms.
        #. It has a certain number of contacts with the polymer.
        #. It contacts a certain number of residues in the polymer.
        #. Its atoms span a single chain.

    If a ligand doesn't pass any of these checks, the function returns ``None``.

    :param m_lig: A boolean mask pointing to putative ligand atoms.
    :param m_pol: A boolean mask pointing to polymer atoms that supposedly
        contact ligand atoms.
    :param structure: A parent structure to which the masks can be applied.
    :return: An instantiated ligand or ``None`` if the checks were not passed.
    """
    a, cfg = structure.array, DefaultConfig["ligand"]

    if m_lig.sum() < cfg["min_atoms"]:
        return None

    lig_chains = bst.get_chains(a[m_lig])
    if len(lig_chains) != 1:
        LOGGER.warning(
            f"Ligand must correspond to a single chain. "
            f"Found {len(lig_chains)}: {lig_chains}."
        )
        return None

    # Find contacts
    m_cont, dist, ligand_idx = util.find_contacts(a, m_lig)
    m_cont[~m_pol] = 0
    dist[~m_pol] = -1

    # The number of residues connected to a ligand
    num_residues = bst.get_residue_count(a[m_cont & m_pol])

    if (
        np.sum(m_cont & m_pol) < cfg["min_atom_connections"]
        or num_residues < cfg["min_res_connections"]
    ):
        return None

    lig_num_residues = bst.get_residue_count(a[m_lig])
    if lig_num_residues == 1:
        name, res_id = a[m_lig].res_name[0], a[m_lig].res_id[0]
    elif lig_num_residues > 1:
        lig_poly_type = util.find_first_polymer_type(a[m_lig])
        # _, lig_poly_type = find_primary_polymer_type(a[m_lig])
        if lig_poly_type == "x":
            LOGGER.warning(
                f"Ligand of a structure {structure.name} contains "
                f"{lig_num_residues} residues but is not polymeric."
            )
        name = f"{lig_poly_type}{lig_num_residues}"
        res_id = f"{a[m_lig].res_id[0]}-{a[m_lig].res_id[-1]}"
    else:
        raise RuntimeError("Zero residues in a putative ligand")

    names = DefaultConfig["metadata"]
    meta = {
        names["res_name"]: name,
        names["res_id"]: res_id,
        names["structure_chain_id"]: lig_chains[0],
    }

    return Ligand(structure, m_lig, m_cont, ligand_idx, dist, meta)


def ligands_from_atom_marks(
    structure: GenericStructure,
) -> abc.Generator[Ligand, None, None]:
    a, m, g = structure.array, structure.atom_marks, structure.graph
    assert len(a) == len(m)

    pol_type = util.find_first_polymer_type(m)
    if pol_type == "x":
        LOGGER.warning(
            f"Failed to determine polymer type for structure {structure.name}. "
            f"Returning empty ligands."
        )
        return
    pol_marks = dict(POL_MARKS)
    pol_mask = m == pol_marks[pol_type]

    # For atoms marked as ligands, it's safe to iterate over residues
    # to recreate ligands.
    ligand_mask = (m == AtomMark.LIGAND) | (m == (AtomMark.LIGAND | AtomMark.COVALENT))
    ligand_idx = np.where(ligand_mask)[0]
    starts = np.unique(bst.get_residue_starts_for(a, ligand_idx))
    for r_mask in bst.get_residue_masks(a, starts):
        lig = make_ligand(r_mask, pol_mask & ~r_mask, structure)
        if lig is not None:
            yield lig

    # For polymer ligands, one should either iterate over connected components
    # or series of consecutive residues.
    lig_poly_m = (
        (m == AtomMark.LIGAND | AtomMark.PEP)
        | (m == AtomMark.LIGAND | AtomMark.NUC)
        | (m == AtomMark.LIGAND | AtomMark.CARB)
    )
    if not np.any(lig_poly_m):
        return

    # The process follows the same logic as as in `mark_atoms_g`.
    sg = g.subgraph(np.where(lig_poly_m)[0])
    cc_idx_viewed = set()

    for cc_idx in map(list, rx.connected_components(sg)):
        cc_idx = sg.subgraph(cc_idx).nodes()
        r_mask = util.extend_residue_mask(a, cc_idx)
        r_mask[list(cc_idx_viewed)] = False

        if not np.any(r_mask):
            continue

        cc_idx_viewed |= set(np.where(r_mask)[0])

        # Attempt making a ligand
        lig = make_ligand(r_mask, pol_mask & ~r_mask, structure)
        if lig is not None:
            yield lig


if __name__ == "__main__":
    raise RuntimeError
