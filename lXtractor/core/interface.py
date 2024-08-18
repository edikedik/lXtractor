"""
The module defines :class:`Interface`, representing an interface between
two partners in a molecular structure.
"""
from __future__ import annotations

import json
import operator
import typing as t
from collections import abc
from dataclasses import dataclass
from itertools import chain, product, starmap, filterfalse
from os import PathLike
from pathlib import Path

import biotite.structure as bst
import numpy as np
import numpy.typing as npt
import rustworkx as rx
from more_itertools import unique_everseen, ilen
from scipy.spatial import KDTree
from toolz import curry, compose_left, valmap

from lXtractor.core import GenericStructure, DefaultConfig
from lXtractor.core.exceptions import MissingData, AmbiguousData, LengthMismatch
from lXtractor.util import get_files
from lXtractor.util.misc import molgraph_to_json

_PartnerUnit = abc.Sequence[str] | str
_Partners: t.TypeAlias = tuple[_PartnerUnit, _PartnerUnit] | str
_ChainIDs: t.TypeAlias = abc.Sequence[str] | str | None
EMPTY = np.empty(0, dtype=int)


def parse_partner(
    partner: _PartnerUnit, parent_structure: GenericStructure
) -> abc.Iterator[str]:
    if isinstance(partner, str):
        partner = list(partner)

    chain_it = parent_structure.split_chains(polymer=True)
    chain2gs = {s.chain_ids_polymer.pop(): s for s in chain_it}

    for c in partner:
        if c not in chain2gs:
            raise MissingData(
                f"Missing partner chain {c} in structure {parent_structure}."
            )
        yield c


def filter_edges_by_nodes(g: rx.PyGraph, node_indices: abc.Sequence[int]) -> list[int]:
    edge_idx = chain.from_iterable(list(g.incident_edges(idx)) for idx in node_indices)
    return list(unique_everseen(edge_idx))


def split_atoms_by_chains(
    a: bst.AtomArray, chains_a: abc.Sequence[str], chains_b: abc.Sequence[str]
) -> tuple[bst.AtomArray, bst.AtomArray]:
    idx_a = np.isin(a.chain_id, chains_a)
    idx_b = np.isin(a.chain_id, chains_b)
    return a[idx_a], a[idx_b]


def residue_count_disjoint(a: bst.AtomArray) -> int:
    def wrap_atom(atom: bst.Atom) -> tuple[int, ...]:
        return atom.chain_id, atom.ins_code, atom.res_id, atom.res_name

    return ilen(unique_everseen(map(wrap_atom, a)))


def _group_chains(
    interface_ccs,
) -> dict[str, tuple[list[bst.AtomArray], list[bst.AtomArray]]]:
    groups = dict()
    for a, b in interface_ccs:
        chains_a, chains_b = map(bst.get_chains, (a, b))
        for c_a, c_b in product(chains_a, chains_b):
            key = c_a + c_b
            if key not in groups:
                groups[key] = ([], [])
            atoms_a = a[a.chain_id == c_a]
            atoms_b = b[b.chain_id == c_b]
            groups[key][0].append(atoms_a)
            groups[key][1].append(atoms_b)
    return groups


def _read_bond_graph(
    inp: Path | dict, parent_structure: GenericStructure
) -> rx.PyGraph:
    if not isinstance(inp, dict):
        with open(inp) as f:
            inp: dict = json.load(f)
    n = inp["num_nodes"]
    if n != len(parent_structure):
        raise LengthMismatch(
            f"The expected number of nodes {n} does not correspond to the number "
            f"of atoms {len(parent_structure)} in structure {parent_structure}."
        )
    g = rx.PyGraph(multigraph=False)
    atom_nodes = list(starmap(AtomNode, enumerate(parent_structure.array)))
    g.add_nodes_from(atom_nodes)
    edges = [
        (a_i, b_i, ContactEdge.from_node_indices(a_i, b_i, g))
        for a_i, b_i in inp["edges"]
    ]
    g.add_edges_from(edges)
    return g


@dataclass
class AtomNode:
    """
    Represents an atom node in the interface graph.
    """

    #: Index of the atom in the parent structure
    idx: int
    #: The :class:`Atom` object of `biotite`.
    atom: bst.Atom
    #: Indicates if the atom is involved in any contacts
    has_edge: bool = False

    def __eq__(self, other: t.Any) -> bool:
        def atoms_equal(a1: bst.Atom, a2: bst.Atom) -> bool:
            return (
                a1.chain_id == a2.chain_id
                and a1.res_id == a2.res_id
                and a1.res_name == a2.res_name
                and a1.ins_code == a2.ins_code
                and a1.atom_name == a2.atom_name
            )

        if not isinstance(other, self.__class__):
            return False
        return (
            self.idx == other.idx
            and self.has_edge == other.has_edge
            and atoms_equal(self.atom, other.atom)
        )


@dataclass
class ContactEdge:
    """
    Represents a contact edge in the interface graph.

    By convention, in an edge ``(i, j)``, ``i`` belongs to partner chains "a",
    whereas ``j`` belongs to partner chains "b".
    """

    #: AtomNode object for the first atom in the contact
    atom_a: AtomNode
    #: AtomNode object for the second atom in the contact
    atom_b: AtomNode
    #: Distance between the two atoms
    dist: float

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            abs(self.dist - other.dist) < 1e-3
            and self.atom_a == other.atom_a
            and self.atom_b == other.atom_b
        )

    @classmethod
    def from_node_indices(cls, i: int, j: int, g: rx.PyGraph) -> t.Self:
        """
        Create a ContactEdge from node indices in the graph.

        :param i: Index of the first atom node.
        :param j: Index of the second atom node.
        :param g: The interface graph.
        :return: A new ContactEdge instance.
        """
        a_i, a_j = g[i], g[j]
        d = np.linalg.norm(a_i.atom.coord - a_j.atom.coord)
        return cls(atom_a=a_i, atom_b=a_j, dist=d)


@dataclass(frozen=True)
class RetainCondition:
    """
    Defines conditions for retaining an interface component.

    This condition is applied to interface parts of  either `a` or `b` partners
    in :meth:`Interface.split_connected`. and then combined into a decision
    whether this interface part is to be retained upon splitting.
    """

    #: Minimum number of atoms required
    min_atoms: int = 1
    #: Minimum number of residues required
    min_res: int = 1
    #: Optional function to select specific atoms
    selector: abc.Callable[[bst.AtomArray], bst.AtomArray] | None = None

    def apply(
        self,
        a: bst.AtomArray,
        return_counts: bool = False,
    ) -> bool | tuple[int, int]:
        """
        Apply the retention condition to an atom array.

        :param a: The atom array to check.
        :param return_counts: If True, return atom and residue counts instead
            of a boolean.
        :return: Boolean indicating if the condition is met, or atom and
            residue counts if return_counts is True.
        """
        if self.selector is not None:
            a = self.selector(a)
        num_atoms = len(a)
        num_res = residue_count_disjoint(a)
        if return_counts:
            return num_atoms, num_res
        return num_atoms >= self.min_atoms and num_res >= self.min_res

    def __call__(self, a: bst.AtomArray) -> bool:
        return self.apply(a, return_counts=False)


@dataclass
class InterfaceSASA:
    """
    Stores Solvent Accessible Surface Area (SASA) values for an interface.
    """

    #: SASA of partner "a" alone.
    a_free: float
    #: SASA of partner "b" alone.
    b_free: float
    #: SASA of partner "a" in the complex.
    a_complex: float
    #: SASA of partner "b" in the complex.
    b_complex: float
    #: SASA of the entire complex.
    complex: float

    @property
    def bsa_a(self) -> float:
        """
        :return: Buried Surface Area of partner "a". Computed as a difference
            between :attr:`a_free` and :attr:`a_complex`.
        """
        return self.a_free - self.a_complex

    @property
    def bsa_b(self) -> float:
        """
        :return: Buried Surface Area of partner "b". Computed as a difference
            between :attr:`b_free` and :attr:`b_complex`.
        """
        return self.b_free - self.b_complex

    @property
    def bsa_complex(self) -> float:
        """
        :return: Total buried surface area. Computed as a difference between
            the sum of the free `a` and `b`  and the complex SASAs.
        """
        return self.a_free + self.b_free - self.complex

    def as_record(self) -> dict[str, float]:
        """
        :return: return SASA and BSA values as a dictionary.
        """
        rec = dict(
            SASA_a_free=self.a_free,
            SASA_b_free=self.b_free,
            SASA_a_complex=self.a_complex,
            SASA_b_complex=self.b_complex,
            SASA_complex=self.complex,
            BSA_a=self.bsa_a,
            BSA_b=self.bsa_b,
            BSA_ab=self.bsa_a + self.bsa_b,
            BSA_complex=self.bsa_complex,
        )
        rec = valmap(float, rec)
        return rec


class Interface:
    # TODO: implement cutoff setter as edge filtering
    """
    An asymmetric interface between two partners in a molecular structure.

    The interface is defined by two distinct sets of partner chains
    (typically protein), designated as "a" and "b", and their interactions.
    The interface is constructed using a graph representation where nodes are
    atoms and edges represent contacts between atoms from different partners.
    For a given edge ``(i, j)``, ``i``  belong to "a" and "b" chain groups, resp.
    A spatial tree (KD-tree) is used to efficiently compute these contacts within
    a specified cutoff distance.

    The class provides methods to analyze the interface, including:

    #. Retrieving contact atoms and indices for each partner
    #. Counting contacts and interacting residues
    #. Calculating SASA and BSA for each partner and the complex
    #. Splitting the interface into connected components

    Two Interfaces are considered equal if they have the same partners
    and their graphs are identical (same nodes and edges).

    :param cutoff: The maximum distance (in Angstroms) for considering
        two atoms to be in contact.
    :param partners: Two tuples of chain identifiers representing the
        two sides of the interface ("a" and "b" partners).
    :param is_subset: Indicates if the interface is a subset of the
        parent structure.

    .. note::
        The asymmetric nature of the interface means that methods and properties
        often have separate versions for "a" and "b" partners
        (e.g., `mask_a` and `mask_b`).

    .. seealso::
        :class:`AtomNode` A node of the interface graph.
        :class:`ContactEdge` An edge of the interface graph.

    .. warning::
        The :meth:`parent_structure` is considered immutable, while the :meth:`G`
        can only change edges and their properties; the atom nodes must stay
        the same.
    """

    def __init__(
        self,
        parent_structure: GenericStructure,
        partners: _Partners,
        subset_parent_to_partners: bool = True,
        cutoff: float = 6.0,
        graph: rx.PyGraph | None = None,
    ):
        """
        Initialize the Interface object.

        :param parent_structure: The parent structure containing the interface.
        :param partners: Chain identifiers for the two sides of the interface.
            Can be a string "A_B" or a tuple of two sequences of chain ids.
        :param subset_parent_to_partners: If ``True``, subset the parent structure
            to only include atoms from the specified partners.
        :param cutoff: The maximum distance for considering two atoms to be
            in contact.
        :param graph: A pre-computed graph representing the interface contacts.
            If None, a new graph will be created.
        :raises AssertionError: If the cutoff is not greater than zero.
        :raises MissingData: If either set of partners is empty.
        :raises AmbiguousData: If the two sets of partners overlap.
        :raises LengthMismatch: If the provided graph doesn't match the structure.
        """
        assert cutoff > 0, "Cutoff must be greater than zero"
        self.cutoff = cutoff

        if isinstance(partners, str):
            partners = partners.split("_", maxsplit=2)
        parser = curry(parse_partner)(parent_structure=parent_structure)
        partners_a, partners_b = map(compose_left(parser, sorted, tuple), partners)
        self.partners: tuple[tuple[str, ...], tuple[str, ...]] = (
            partners_a,
            partners_b,
        )
        self._validate_partners()

        self.is_subset = False
        if subset_parent_to_partners:
            mask = np.isin(parent_structure.array.chain_id, self.partners_joined)
            if mask.sum() != len(parent_structure):
                self.is_subset = True
            # Ligands have to be omitted here as they might trigger LengthMismatch
            # below (eg, in case they have a different chain but the graph was
            # subset without it).
            parent_structure = parent_structure.subset(mask, ligands=False)

        self._parent_structure: GenericStructure = parent_structure
        if graph is not None:
            if graph.num_nodes() != len(parent_structure):
                raise LengthMismatch(
                    f"Graph must have have the same number of atoms as the parent "
                    f"structure. Got {graph.num_nodes()} and {len(parent_structure)}"
                )
            if list(graph.node_indices()) != [x.idx for x in graph.nodes()]:
                raise AmbiguousData(
                    "Node indices and indices stored in `AtomNode` instances must match"
                )
            if any(
                s_atom != node.atom
                for s_atom, node in zip(parent_structure.array, graph.nodes())
            ):
                raise AmbiguousData(
                    "Atoms within the graph must correspond to the structure's atoms."
                )
            self._graph = graph
        else:
            self._graph: rx.PyGraph = self._make_graph()

    def __repr__(self) -> str:
        return self.id

    def __eq__(self, other: t.Any) -> False:
        def graphs_equal(a: rx.PyGraph, b: rx.PyGraph):
            return (
                len(a) == len(b)
                and list(a.nodes()) == list(b.nodes())
                and list(a.edges()) == list(b.edges())
            )

        if not isinstance(other, self.__class__):
            return False

        return self.partners == other.partners and graphs_equal(self.G, other.G)

    @property
    def id(self) -> str:
        """
        Get the unique identifier for this Interface.

        :return: A string representing the interface in the format
            "Interface(partners)<-(parent_structure)".
        """
        return f"Interface({self.partners_fmt})<-({self.parent_structure})"

    @property
    def parent_structure(self) -> GenericStructure:
        """
        Get the parent structure of the interface.

        :return: The parent structure containing this interface.
        """
        return self._parent_structure

    @property
    def G(self) -> rx.PyGraph:
        """
        Get the graph representation of the interface contacts.

        :return: A graph where nodes represent atoms and edges represent
            contacts between atoms from different partners.
        """
        return self._graph

    @property
    def partners_a(self) -> tuple[str, ...]:
        """
        :return: A tuple of chain identifiers for the first partner group.
        """
        return self.partners[0]

    @property
    def partners_b(self) -> tuple[str, ...]:
        """
        :return: A tuple of chain identifiers for the second partner group.
        """

        return self.partners[1]

    @property
    def partners_joined(self) -> list[str]:
        """
        :return: A list of all chain identifiers from both partners.
        """
        return [*self.partners_a, *self.partners_b]

    @property
    def partners_fmt(self) -> str:
        """
        Get a formatted string representation of the partners.

        :return: A string in the format "A_B" where A and B are the
            concatenated chain identifiers of each partner.
        """
        a = "".join(self.partners_a)
        b = "".join(self.partners_b)
        return f"{a}_{b}"

    @property
    def mask_a(self) -> npt.NDArray[bool]:
        """
        :return: A numpy array of booleans, ``True`` for atoms in the first
            partner group.
        """
        return self._chain_atom_mask(self.partners_a)

    @property
    def mask_b(self) -> npt.NDArray[bool]:
        """
        :return: A numpy array of booleans, ``True`` for atoms in the second
            partner group.
        """
        return self._chain_atom_mask(self.partners_b)

    def _parse_chain_ids(self, chain_ids: _ChainIDs) -> tuple[str, ...] | None:
        match chain_ids:
            case None:
                return None
            case "a":
                return self.partners_a
            case "b":
                return self.partners_b
            case abc.Sequence():
                return tuple(chain_ids)
            case str():
                if "," in chain_ids:
                    return tuple(chain_ids.split(","))
                return (chain_ids,)
            case _:
                raise TypeError(f"Invalid `chain_ids` type {type(chain_ids)}")

    def _make_graph(self) -> rx.PyGraph:
        def setup_tree(chain_ids):
            idx = self._chain_atom_idx(chain_ids)
            atoms = array[idx]
            tree = KDTree(atoms.coord)
            return idx, atoms, tree

        def iter_edges(graph: rx.PyGraph) -> abc.Iterator[tuple[int, int, ContactEdge]]:
            idx_a, atoms_a, tree_a = setup_tree(self.partners_a)
            idx_b, atoms_b, tree_b = setup_tree(self.partners_b)

            solvent_mask = self.parent_structure.mask.solvent
            solvent_mask_a, solvent_mask_b = solvent_mask[idx_a], solvent_mask[idx_b]
            idx_contact = tree_a.query_ball_tree(tree_b, r=self.cutoff)

            for a_i, b_indices in enumerate(idx_contact):
                if solvent_mask_a[a_i]:
                    continue
                b_indices = list(filterfalse(lambda x: solvent_mask_b[x], b_indices))
                if len(b_indices) == 0:
                    continue

                a_i_real = idx_a[a_i]
                for b_i in b_indices:
                    b_i_real = idx_b[b_i]
                    yield a_i_real, b_i_real, ContactEdge.from_node_indices(
                        a_i_real, b_i_real, graph
                    )

        array = self.parent_structure.array
        g = rx.PyGraph(multigraph=False)
        atom_nodes = list(starmap(AtomNode, enumerate(array)))
        g.add_nodes_from(atom_nodes)
        g.add_edges_from(list(iter_edges(g)))

        return g

    def _subset_graph(
        self, node_indices: abc.Sequence[int], remove: bool
    ) -> rx.PyGraph:
        if remove:
            g = self.G.copy()
            for a, b in g.edge_list():
                if not (a in node_indices or b in node_indices):
                    g.remove_edge(a, b)
            return g
        g = self.G.subgraph(node_indices)
        for i in g.node_indices():
            g[i].idx = i
        return g

    def _validate_partners(self):
        if len(self.partners_a) == 0:
            raise MissingData("Empty partners `a`.")
        if len(self.partners_b) == 0:
            raise MissingData("Empty partners `b`.")
        common = set(self.partners_a) & set(self.partners_b)
        if len(common) > 0:
            raise AmbiguousData(f"Provided partners overlap over chains: {common}.")

    def _chain_atom_mask(self, chain_ids: _ChainIDs) -> npt.NDArray[bool]:
        chain_ids = self._parse_chain_ids(chain_ids)
        if chain_ids is not None and len(chain_ids) > 0:
            return np.isin(
                self.parent_structure.array.chain_id, self._parse_chain_ids(chain_ids)
            )
        return np.ones(len(self.parent_structure), dtype=bool)

    def _chain_atom_idx(self, chain_ids: _ChainIDs) -> npt.NDArray[int]:
        return np.where(self._chain_atom_mask(chain_ids))[0]

    def get_contact_idx(self, chain_ids: _ChainIDs = None) -> npt.NDArray[int]:
        """
        Get the indices of contacting atom pairs.

        :param chain_ids: Optional list of chain IDs to filter the contacts.
        :return: A numpy array of shape (N, 2) where each row contains
            the indices of a contacting atom pair.
        """
        idx = np.array(list(self.G.edge_list()))
        if chain_ids:
            idx = idx[np.isin(idx, self._chain_atom_idx(chain_ids)).any(axis=1)]
        return idx

    def get_contact_idx_a(self, chain_ids: _ChainIDs = None) -> npt.NDArray[int]:
        """
        :param chain_ids: Optional; contacts must involve the provided chains.
        :return: A numpy array of indices of contacting atoms from partner "a".
        """
        idx = self.get_contact_idx(chain_ids)
        if len(idx) == 0:
            return EMPTY
        return idx[:, 0]

    def get_contact_idx_b(self, chain_ids: _ChainIDs = None) -> npt.NDArray[int]:
        """
        :param chain_ids: Optional; contacts must involve the provided chains.
        :return: A numpy array of indices of contacting atoms from partner "b".
        """
        idx = self.get_contact_idx(chain_ids)
        if len(idx) == 0:
            return EMPTY
        return idx[:, 1]

    def get_contact_atoms(
        self, chain_ids: _ChainIDs = None
    ) -> tuple[bst.AtomArray, bst.AtomArray]:
        """
        Get the contacting atoms from both partners.

        :param chain_ids: Optional; include results only for the provided chains.
        :return: A tuple of two AtomArrays of equal sizes containing the
            contacting atoms from partners "a" and "b" respectively.
        """
        a = self.parent_structure.array
        idx = self.get_contact_idx(chain_ids)
        if len(idx) == 0:
            return bst.AtomArray(0), bst.AtomArray(0)
        idx_a, idx_b = idx[:, 0], idx[:, 1]
        return a[idx_a], a[idx_b]

    def get_contact_atoms_mask(self, chain_ids: _ChainIDs = None) -> npt.NDArray[bool]:
        """
        Get a mask pointing to contact atoms.

        :param chain_ids: Contacts must involve the provided chains.
        :return: An array where ``True`` points to atoms involved in interface contacts.
        """
        contact_idx = np.unique(self.get_contact_idx(chain_ids))
        mask = np.zeros(len(self.parent_structure), dtype=bool)
        mask[contact_idx] = True
        return mask

    def count_contacts(self, chain_ids: _ChainIDs = None) -> int:
        """
        Count the number of contacts in the interface. Equivalent to a number
        of edges in :meth:`G`.

        :param chain_ids: Optional; count counts involving the provided chains.
        :return: The number of atom-atom contacts in the interface
        """
        return len(self.get_contact_idx(chain_ids))

    def count_contact_atoms(
        self, chain_ids: _ChainIDs = None, strict: bool = False
    ) -> int:
        """
        Count the number of atoms involved in contacts. Equivalent to the total
        number of nodes connected with edges in :meth:`G`.

        :param chain_ids: Optional; count contact atoms involving the provided
            chains.
        :param strict: Used only if `chain_ids` is provided. If ``True``, will
            filter to atoms from provided `chain_ids`. Otherwise, will count
            all atoms making contacts with specified `chain_ids`.
        :return: The number of unique atoms involved in contacts.
        """
        idx = np.unique(self.get_contact_idx(chain_ids))

        chain_ids = self._parse_chain_ids(chain_ids)
        if strict and chain_ids is not None:
            idx_chain = self._chain_atom_idx(chain_ids)
            idx = idx[np.isin(idx, idx_chain)]

        return len(idx)

    def count_contact_residues(
        self, chain_ids: _ChainIDs = None, strict: bool = False
    ) -> int:
        """
        Count the number of residues involved in contacts.

        :param chain_ids: Optional; include results only for the provided chains.
        :param strict: Used only if `chain_ids` is provided. If ``True``, will
            filter to residues from provided `chain_ids`. Otherwise, will count
            all residues making contacts with specified `chain_ids`.
        :return: The number of unique residues involved in contacts.
        """
        atoms_a, atoms_b = self.get_contact_atoms(chain_ids)
        chain_ids = self._parse_chain_ids(chain_ids)
        if chain_ids is not None and strict:
            atoms_a = atoms_a[np.isin(atoms_a.chain_id, chain_ids)]
            atoms_b = atoms_b[np.isin(atoms_b.chain_id, chain_ids)]
        count_a = 0 if len(atoms_a) == 0 else residue_count_disjoint(atoms_a)
        count_b = 0 if len(atoms_b) == 0 else residue_count_disjoint(atoms_b)
        return count_a + count_b

    def iter_ccs(
        self, as_: str = "nodes", min_nodes: int = 2
    ) -> (
        abc.Iterator[list[int]] | abc.Iterator[rx.PyGraph] | abc.Iterator[bst.AtomArray]
    ):
        """
        Iterate over the connected components of the interface graph.

        :param as_: The format to return the connected components. Options are:

            #. "nodes" (default): List of node indices
            #. "subgraph": Subgraph of the interface graph
            #. "edges": List of edge indices
            #. "atoms": AtomArray of the atoms in the connected component

        :param min_nodes: Minimum number of nodes for a connected component to
            be included. Should be ``>=2``.
        :return: An iterator over the connected components in the specified format.
        """

        def parse_cc(cc):
            cc = sorted(cc)
            if as_ == "nodes":
                return cc
            elif as_ == "subgraph":
                return self.G.subgraph(cc)
            elif as_ == "edges":
                return filter_edges_by_nodes(self.G, cc)
            elif as_ == "atoms":
                idx = np.array(cc)
                return self.parent_structure.array[idx]
            else:
                raise ValueError(f"Invalid `as_` parameter {as_}.")

        ccs = filter(lambda x: len(x) >= min_nodes, rx.connected_components(self.G))
        yield from map(parse_cc, ccs)

    def sasa(
        self, mask: npt.NDArray[bool] | None = None, canonical: bool = True
    ) -> InterfaceSASA:
        """
        Calculate the Solvent Accessible Surface Area (SASA) for the interface.
        See :class:`InterfaceSASA` for more details.

        :param mask: A custom atom mask of :meth:`parent_structure` pointing to
            atoms to include in calculation.
        :param canonical: Use only atoms of canonical residues for calculating
            SASA. In some cases, this may save from unexpected exceptions that
            happen due to biotite missing some atoms in non-canonical residues
            that it expects to be there (atomic radii are required for each
            atom for SASA calculation).
        :return: An InterfaceSASA object containing SASA values for partners
            "a" and "b" individually and in complex.
        """

        array = self.parent_structure.array

        m = np.ones(len(array), dtype=bool)
        if mask is not None:
            m &= mask
        if canonical:
            m &= bst.filter_canonical_amino_acids(array)

        array = array[m]
        mask_a, mask_b = self.mask_a[m], self.mask_b[m]
        array_a, array_b = array[mask_a], array[mask_b]
        sasa_a, sasa_b, sasa_ab = map(
            lambda x: np.nansum(bst.sasa(x)), [array_a, array_b, array]
        )
        sasa_a_comp, sasa_b_comp = starmap(
            lambda x, m: np.nansum(bst.sasa(x, atom_filter=m)),
            [(array, mask_a), (array, mask_b)],
        )
        return InterfaceSASA(sasa_a, sasa_b, sasa_a_comp, sasa_b_comp, sasa_ab)

    def split_connected(
        self,
        condition_a: abc.Callable[[bst.AtomArray], bool] = RetainCondition(),
        condition_b: abc.Callable[[bst.AtomArray], bool] = RetainCondition(),
        conditions_op: abc.Callable[[bool, bool], bool] = operator.and_,
        conditions_apply_to: str = "chains",
        into_pairs: bool = False,
        subset_parent_to_partners: bool = True,
        cutoff: float = 6.0,
    ) -> abc.Iterator[t.Self]:
        # TODO: use cutoff to filter CCs; otherwise, it's misleading
        """
        Split the interface into connected components based on specified
        conditions.

        This method allows for sophisticated filtering and splitting of the
        interface based on user-defined conditions. It can be used to identify
        specific sub-interfaces or to analyze the interface at different levels
        of granularity.

        :param condition_a: Conditions for filtering partner "a" components.
            Can be any callable accepting an arbitrary atom array corresponding
            to an interface "side" and returning a boolean.
        :param condition_b: Conditions for filtering partner "b" components.
        :param conditions_op: Operator to combine conditions_a and conditions_b.
            Default is `operator.and_` (both conditions must be met).
        :param conditions_apply_to: Whether to apply conditions to "chains"
            (default) or individual connected components.
        :param into_pairs: If True, split into pairwise interfaces between
            individual chains.
        :param subset_parent_to_partners: If True, subset the parent structure
            in the resulting interfaces.
        :param cutoff: Distance cutoff for contacts in the resulting interfaces.
        :return: An iterator of :class:`Interface` objects representing the split
            components.

        .. note::
            The `RetainCondition` objects allow for flexible filtering based on
            number of atoms, residues, or custom selectors. This enables complex
            splitting strategies, such as retaining only interfaces with a minimum
            number of interacting residues or specific types of interactions.
            The default retain condition is to have at least one contact residue.

        .. seealso::
            :class:`RetainCondition` for details on how to specify filtering conditions.
        """
        interface_ccs = []
        ccs: abc.Iterator[bst.AtomArray] = self.iter_ccs("atoms")
        for cc in ccs:
            atoms_a, atoms_b = split_atoms_by_chains(
                cc, self.partners_a, self.partners_b
            )
            interface_ccs.append((atoms_a, atoms_b))

        # Group connected components by chain pairs
        cc_groups = _group_chains(interface_ccs)
        chain_pairs = set()
        # Apply conditions to determine which chain pairs to retain
        for k, v in cc_groups.items():
            atoms_a, atoms_b = v[0], v[1]
            if conditions_apply_to == "chains":
                atoms_a = [bst.array(list(chain(*v[0])))]
                atoms_b = [bst.array(list(chain(*v[1])))]
            for a, b in zip(atoms_a, atoms_b):
                a_passes = condition_a(a)
                b_passes = condition_b(b)
                ab_passes = conditions_op(a_passes, b_passes)
                if ab_passes:
                    chain_pairs.add(k)
                    break

        if into_pairs:
            for pair in chain_pairs:
                atom_idx = self._chain_atom_idx(list(pair))
                graph = self._subset_graph(
                    atom_idx, remove=not subset_parent_to_partners
                )
                yield self.__class__(
                    self.parent_structure,
                    f"{pair[0]}_{pair[1]}",
                    subset_parent_to_partners=subset_parent_to_partners,
                    cutoff=cutoff,
                    graph=graph,
                )
        else:
            g = rx.PyGraph(multigraph=False)

            def get_chain_map(idx: int) -> dict[str, int]:
                cs = list(unique_everseen(x[idx] for x in chain_pairs))
                node_idx = g.add_nodes_from(cs)
                return dict(zip(cs, node_idx))

            chain2idx_a, chain2idx_b = map(get_chain_map, (0, 1))
            for pair in chain_pairs:
                g.add_edge(chain2idx_a[pair[0]], chain2idx_b[pair[1]], None)

            for cc in rx.connected_components(g):
                chains = [g[idx] for idx in cc]
                chains_a, chains_b = map(
                    lambda d: "".join(x for x in chains if x in d),
                    [chain2idx_a, chain2idx_b],
                )
                atom_idx = self._chain_atom_idx(chains)
                graph = self._subset_graph(
                    atom_idx, remove=not subset_parent_to_partners
                )
                yield self.__class__(
                    self.parent_structure,
                    f"{chains_a}_{chains_b}",
                    subset_parent_to_partners=subset_parent_to_partners,
                    cutoff=cutoff,
                    graph=graph,
                )

    @classmethod
    def read(cls, path: PathLike | str) -> t.Self:
        """
        Read an Interface object from a file.

        :param path: Path to the directory containing the interface files.
        :return: An Interface object.
        :raises FileNotFoundError: If required files are missing.
        :raises MissingData: If required metadata is missing.
        """
        path = Path(path)
        files = get_files(path)

        if "meta.json" not in files:
            raise FileNotFoundError(f"No metadata file in {path}.")

        meta = json.loads(files["meta.json"].read_text(encoding="utf-8"))

        try:
            parent_filename = meta["parent_filename"]
        except KeyError as e:
            raise MissingData("Missing `parent_filename` in metadata.") from e
        try:
            partners = meta["partners"]
        except KeyError as e:
            raise MissingData("Missing `partners` in metadata.") from e
        try:
            graph_filename = meta["graph_filename"]
        except KeyError as e:
            raise MissingData("Missing `graph_filename` in metadata.") from e

        parent_path = path / parent_filename
        if not parent_path.exists():
            raise FileNotFoundError(f"No parent structure file found at {parent_path}.")
        graph_path = path / graph_filename
        if not graph_path.exists():
            raise FileNotFoundError(f"No graph file found at {graph_path}.")

        parent = GenericStructure.read(parent_path)
        graph = _read_bond_graph(path / graph_filename, parent)
        return cls(
            parent,
            partners,
            meta.get("is_subset", True),
            meta.get("cutoff", 6.0),
            graph,
        )

    def write(
        self,
        base_dir: PathLike | str,
        overwrite: bool = False,
        name: str | None = None,
        str_fmt: str = DefaultConfig["structure"]["fmt"],
        additional_meta: dict[str, t.Any] | True | None = None,
    ) -> Path:
        """
        Write the Interface object to files.

        :param base_dir: Base directory to write the files.
        :param overwrite: If True, overwrite existing files.
        :param name: Name for the interface directory (default is the interface ID).
        :param str_fmt: Format for writing the structure file.
        :param additional_meta: Additional metadata to include in the JSON file.
            If ``dict``, ads it to the default metadata records. If ``True``,
            includes :class:`InterfaceSASA`.
        :return: Path to the destination directory.
        """
        name = name or self.id
        base = Path(base_dir) / name
        base = Path(base)
        base.mkdir(parents=True, exist_ok=overwrite)

        parent_path = base / f"{self.parent_structure.name}.{str_fmt}"
        parent_path = self.parent_structure.write(parent_path)
        parent_filename = parent_path.name
        graph_path = molgraph_to_json(self.G, base / "graph.json")

        meta = dict(
            parent=self.parent_structure.name,
            parent_id=self.parent_structure.id,
            parent_filename=parent_filename,
            graph_filename=graph_path.name,
            partners=self.partners_fmt,
            cutoff=self.cutoff,
            subset=self.is_subset,
        )
        if isinstance(additional_meta, abc.Mapping):
            meta.update(additional_meta)
        if additional_meta is True:
            meta.update(self.sasa().as_record())

        with (base / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return base


class InterfaceComparator:
    """
    A class for comparing interfaces corresponding to different states of the
    same binding partners. It assumes that parent structures of these states
    have the same atoms in the same order but perhaps with different coordinates.
    To check if the interfaces are comparable, one may use :meth:`are_comparable`
    before initializing.

    It superposes parent structure of :attr:`state_mob` over :attr:`state_ref`
    during initialization. Then, common metrics such as :meth:`irmsd`,
    :meth:`lrmsd` and :meth:`dockq` can be computed fast and reliably.
    """

    def __init__(
        self,
        state_ref: Interface,
        state_mob: Interface,
        superpose_by: str | np.ndarray = "a",
        min_spp_atoms: int = 5,
    ):
        """
        :param state_ref: A reference state of the interface.
        :param state_mob: A mobile state of the interface. Its structure copy
            after superposition will be stored after init and can be accessed
            via :meth:`superposed_mob`.
        :param superpose_by: Defines which set of atoms is used to superpose
            mobile state over the fixed one. Can be a ``"a"`` or ``"b"`` to
            indicate corresponding binding partners or a ``str`` with
            ","-separated chains. Can also be a` numpy` array with atom indices
            or boolean mask pointing to atoms to use for superposition.
        :param min_spp_atoms: Minimum number of atoms necessary to superpose
            structures after `superpose_by` is applied.
        :raises AmbiguousData: if interfaces are not comparable.

        .. note::
            A ``str`` type of :attr:`superpose_by` is used to filter the
            interface contacts to specified chains. For instance, setting it
            with ``"A"`` will result in the selection of atom contact atoms
            involving chain A as opposed to using only chain A contacts for
            superposition. Hence, values ``"a"`` and ``"b"`` are essentially
            equivalent since they'll result in the same selection of contact
            atoms: those involved in the interface formation.
        """
        if not self.are_comparable(state_ref, state_mob):
            raise AmbiguousData("States are not comparable.")

        #: Reference interface state.
        self.state_ref = state_ref
        #: Mobile interface state.
        self.state_mob = state_mob
        #: Superpose selection specifications.
        self.superpose_by = superpose_by

        self._superpose_atom_mask = self._infer_spp_atom_mask(min_spp_atoms)

        (
            self._superposed_mob,
            self._spp_rmsd,
            self._transformation,
        ) = self.state_ref.parent_structure.superpose(
            self.state_mob.parent_structure,
            mask_self=self._superpose_atom_mask,
            mask_other=self._superpose_atom_mask,
        )

    def _validate_args(self):
        if isinstance(self.superpose_by, int):
            pass

    def _validate_states(self):
        if self.state_ref.count_contacts() == 0:
            raise MissingData(f"No contacts in reference state {self.state_ref}.")
        # if self.state_mob.count_contacts() == 0:
        #     raise MissingData(f"No contacts in mobile state {self.state_mob}.")

    def _infer_spp_atom_mask(self, min_atoms: int) -> npt.NDArray[np.bool_]:
        if isinstance(self.superpose_by, str):
            try:
                idx = np.unique(self.state_ref.get_contact_idx(self.superpose_by))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to infer indices from specs {self.superpose_by}"
                ) from e
            mask = np.zeros(len(self.state_ref.parent_structure), dtype=bool)
            mask[idx] = True
        elif isinstance(self.superpose_by, np.ndarray):
            if len(self.superpose_by.shape) > 1:
                raise AmbiguousData(
                    "Multidimensional arrays are not supported in `superpose_by` specs."
                )
            if len(self.superpose_by) == 0:
                raise AmbiguousData("Empty `superpose_by` array.")
            if np.issubdtype(self.superpose_by.dtype, np.bool_):
                if len(self.superpose_by) != len(self.state_ref.parent_structure):
                    raise ValueError(
                        f"Boolean mask size {len(self.superpose_by)} does not match "
                        f"the number of atoms in the structure "
                        f"{len(self.state_ref.parent_structure)}."
                    )
                mask = self.superpose_by
            elif np.issubdtype(self.superpose_by.dtype, np.integer):
                mask = np.zeros(len(self.state_ref.parent_structure), dtype=bool)
                mask[self.superpose_by] = True
            else:
                raise TypeError("Invalid array input dtype for `superpose_by`.")
        else:
            raise TypeError(
                f"Invalid type {type(self.superpose_by)} for `superpose_by`."
            )

        num_atoms = mask.sum()
        if num_atoms < min_atoms:
            raise ValueError(
                f"The number of atoms {num_atoms} for superposing is below "
                f"the allowed minimum {min_atoms}."
            )
        return mask

    @property
    def superposed_mob(self) -> GenericStructure:
        """
        :return: A copy of mobile structure with coordinates transformed
            following superpositions.
        """
        return self._superposed_mob

    @property
    def superposed_rmsd(self) -> float:
        """
        :return: RMSD after superposing :attr:`state_mob`.
        """
        return self._spp_rmsd

    @property
    def transformation(
        self,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        :return: Transformation matrices of the inferred for superposition
            of the :attr:`state_mob`.
        """
        return self._transformation

    @classmethod
    def are_comparable(cls, state1: Interface, state2: Interface) -> bool:
        """
        A method to check whether two interface states are comparable to be
        used in this class.

        :param state1: First interface.
        :param state2: Second interface.
        :return: ``True`` if interfaces are directly comparable and can be used
            in this comparator and ``False`` otherwise.
        """
        a1, a2 = state1.parent_structure.array, state2.parent_structure.array
        return (
            len(a1) == len(a2)
            and np.all(a1.atom_name == a2.atom_name)
            and np.all(a1.chain_id == a2.chain_id)
        )

    def rmsd_over(self, atom_mask: npt.NDArray[np.bool_]) -> float:
        """
        A general-purpose method to compute RMSD between reference and mobile
        states over arbitrary set of atoms.

        :param atom_mask: A boolean mask where ``True`` indicates target atoms.
        :return: RMSD over target atoms.
        """
        a_ref = self.state_ref.parent_structure.array[atom_mask]
        a_mob = self.superposed_mob[atom_mask]
        return bst.rmsd(a_ref, a_mob)

    def irmsd(self) -> float:
        """
        Compute interface RMSD.

        :return: RMSD computed over atoms comprising interface of the
            :attr:`state_ref`.
        """
        return self.rmsd_over(self.state_ref.get_contact_atoms_mask())

    def lrmsd(self, ligand_chains: str | abc.Sequence[str] = "b") -> float:
        """
        Compute "ligand" RMSD.

        :param ligand_chains: Specification of which chains to consider "ligand".
            By default, this points to :meth:`Interface.partner_b`.
        :return: RMSD computed over chains posing as "ligand".
        """
        mask = np.isin(
            self.state_ref.parent_structure.array.chain_id,
            self.state_ref._parse_chain_ids(ligand_chains),
        )
        return self.rmsd_over(mask)

    def fnat(self) -> float:
        """
        :return: A fraction of contacts preserved in :attr:`state_mob`.
        """
        idx_ref = self.state_ref.get_contact_idx()
        idx_mob = self.state_mob.get_contact_idx()
        if len(idx_mob) == 0:
            return 0.0
        isec_size = np.sum(np.isin(idx_mob, idx_ref).all(axis=1))
        return isec_size / len(idx_ref)

    def dockq(self, d1: float = 8.5, d2: float = 1.5) -> float:
        """
        A DockQ score from :cite:`sankar16`.

        :param d1: A constant to scale :meth:`lrmsd`.
        :param d2: A constant to scale :meth:`irmsd`.
        :return: DockQ score ranging from 0 (no match) to 1 (perfect match).

        .. bibliography ::

        """
        irms, lrms, fnat = self.irmsd(), self.lrmsd(), self.fnat()
        lrmss = 1 / (1 + (irms / d1) ** 2)
        irmss = 1 / (1 + (lrms / d2) ** 2)
        return (fnat + lrmss + irmss) / 3


if __name__ == "__main__":
    raise RuntimeError
