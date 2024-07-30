from __future__ import annotations

import operator
import typing as t
from collections import abc
from dataclasses import dataclass
from itertools import chain, product, starmap

import biotite.structure as bst
import numpy as np
import numpy.typing as npt
import rustworkx as rx
from more_itertools import unique_everseen, ilen
from scipy.spatial import KDTree
from toolz import curry, compose_left

from lXtractor.core import GenericStructure
from lXtractor.core.exceptions import MissingData, AmbiguousData, LengthMismatch

_PartnerUnit = abc.Sequence[str] | str
_Partners: t.TypeAlias = tuple[_PartnerUnit, _PartnerUnit] | str
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


@dataclass
class AtomNode:
    idx: int
    atom: bst.Atom
    has_edge: bool = False


@dataclass
class ContactEdge:
    atom_a: AtomNode
    atom_b: AtomNode
    dist: float

    @classmethod
    def from_node_indices(cls, i: int, j: int, g: rx.PyGraph) -> t.Self:
        a_i, a_j = g[i], g[j]
        d = np.linalg.norm(a_i.atom.coord - a_j.atom.coord)
        return cls(atom_a=a_i, atom_b=a_j, dist=d)


@dataclass(frozen=True)
class RetainCondition:
    min_atoms: int = 1
    min_res: int = 1
    selector: abc.Callable[[bst.AtomArray], bst.AtomArray] | None = None

    def apply(
        self,
        a: bst.AtomArray,
        return_counts: bool = False,
    ) -> bool | tuple[int, int]:
        if self.selector is not None:
            a = self.selector(a)
        num_atoms = len(a)
        num_res = residue_count_disjoint(a)
        if return_counts:
            return num_atoms, num_res
        return num_atoms >= self.min_atoms and num_res >= self.min_res


class Interface:
    def __init__(
        self,
        parent_structure: GenericStructure,
        partners: _Partners,
        subset_parent_to_partners: bool = True,
        cutoff: float = 6.0,
        graph: rx.PyGraph | None = None,
    ):
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

        if subset_parent_to_partners:
            mask = np.isin(parent_structure.array.chain_id, self.partners_joined)
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
        return f"Interface({self.partners_fmt})"

    @property
    def parent_structure(self) -> GenericStructure:
        return self._parent_structure

    @property
    def G(self) -> rx.PyGraph:
        return self._graph

    @property
    def partners_a(self) -> tuple[str, ...]:
        return self.partners[0]

    @property
    def partners_b(self) -> tuple[str, ...]:
        return self.partners[1]

    @property
    def partners_joined(self) -> list[str]:
        return [*self.partners_a, *self.partners_b]

    @property
    def partners_fmt(self) -> str:
        a = "".join(self.partners_a)
        b = "".join(self.partners_b)
        return f"{a}_{b}"

    @property
    def mask_a(self) -> npt.NDArray[bool]:
        return self._chain_atom_mask(self.partners_a)

    @property
    def mask_b(self) -> npt.NDArray[bool]:
        return self._chain_atom_mask(self.partners_b)

    def _make_graph(self) -> rx.PyGraph:
        def setup_tree(chain_ids):
            idx = self._chain_atom_idx(chain_ids)
            atoms = array[idx]
            tree = KDTree(atoms.coord)
            return idx, atoms, tree

        array = self.parent_structure.array
        g = rx.PyGraph(multigraph=False)
        atom_nodes = list(starmap(AtomNode, enumerate(array)))
        g.add_nodes_from(atom_nodes)

        idx_a, atoms_a, tree_a = setup_tree(self.partners_a)
        idx_b, atoms_b, tree_b = setup_tree(self.partners_b)

        idx_contact = tree_a.query_ball_tree(tree_b, r=self.cutoff)

        for a_i, b_indices in enumerate(idx_contact):
            if len(b_indices) == 0:
                continue

            indices = ((idx_a[a_i], idx_b[b_i]) for b_i in b_indices)

            edges = [
                (a_i, b_i, ContactEdge.from_node_indices(a_i, b_i, g))
                for a_i, b_i in indices
            ]
            g.add_edges_from(edges)

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

    def _chain_atom_mask(self, chain_ids: abc.Sequence[str]) -> npt.NDArray[bool]:
        return np.isin(self.parent_structure.array.chain_id, chain_ids)

    def _chain_atom_idx(self, chain_ids: abc.Sequence[str]) -> npt.NDArray[int]:
        return np.where(self._chain_atom_mask(chain_ids))[0]

    def get_contact_idx_ab(
        self, chain_ids: abc.Sequence[str] | None = None
    ) -> npt.NDArray[int]:
        idx = np.array(list(self.G.edge_list()))
        if chain_ids:
            idx = idx[np.isin(idx, self._chain_atom_idx(chain_ids).any(axis=1))]
        return idx

    def get_contact_idx_a(self) -> npt.NDArray[int]:
        idx = self.get_contact_idx_ab()
        if len(idx) == 0:
            return EMPTY
        return self.get_contact_idx_ab()[:, 0]

    def get_contact_idx_b(self) -> npt.NDArray[int]:
        idx = self.get_contact_idx_ab()
        if len(idx) == 0:
            return EMPTY
        return self.get_contact_idx_ab()[:, 1]

    def get_contact_atoms_ab(
        self, chain_ids: abc.Sequence[str] | None = None
    ) -> tuple[bst.AtomArray, bst.AtomArray]:
        a = self.parent_structure.array
        idx = self.get_contact_idx_ab(chain_ids)
        if len(idx) == 0:
            return bst.AtomArray(0), bst.AtomArray(0)
        idx_a, idx_b = idx[:, 0], idx[:, 1]
        return a[idx_a], a[idx_b]

    def num_contacts(self, chain_ids: abc.Sequence[str] | None = None) -> int:
        return len(self.get_contact_idx_ab(chain_ids))

    def num_contact_atoms(self, chain_ids: abc.Sequence[str] | None = None) -> int:
        idx = self.get_contact_idx_ab(chain_ids)
        return len(np.unique(idx))

    def num_contact_residues(self, chain_ids: abc.Sequence[str] | None = None) -> int:
        atoms_a, atoms_b = self.get_contact_atoms_ab(chain_ids)
        count_a = 0 if len(atoms_a) == 0 else residue_count_disjoint(atoms_a)
        count_b = 0 if len(atoms_b) == 0 else residue_count_disjoint(atoms_b)
        return count_a + count_b

    def iter_ccs(
        self, as_: str = "nodes", min_nodes: int = 2
    ) -> (
        abc.Iterator[list[int]] | abc.Iterator[rx.PyGraph] | abc.Iterator[bst.AtomArray]
    ):
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
                raise ValueError("Invalid `as_` parameter.")

        ccs = filter(lambda x: len(x) >= min_nodes, rx.connected_components(self.G))
        yield from map(parse_cc, ccs)

    def split_connected(
        self,
        condition_a: RetainCondition = RetainCondition(),
        condition_b: RetainCondition = RetainCondition(),
        conditions_op: abc.Callable[[bool, bool], bool] = operator.and_,
        conditions_apply_to: str = "chains",
        into_pairs: bool = False,
        subset_parent_to_partners: bool = True,
        cutoff: float = 6.0,
    ) -> abc.Iterator[t.Self]:
        interface_ccs = []
        ccs: abc.Iterator[bst.AtomArray] = self.iter_ccs("atoms")
        for cc in ccs:
            atoms_a, atoms_b = split_atoms_by_chains(
                cc, self.partners_a, self.partners_b
            )
            interface_ccs.append((atoms_a, atoms_b))

        cc_groups = _group_chains(interface_ccs)
        chain_pairs = set()
        for k, v in cc_groups.items():
            atoms_a, atoms_b = v[0], v[1]
            if conditions_apply_to == "chains":
                atoms_a = [bst.array(list(chain(*v[0])))]
                atoms_b = [bst.array(list(chain(*v[1])))]
            for a, b in zip(atoms_a, atoms_b):
                a_passes = condition_a.apply(a)
                b_passes = condition_b.apply(b)
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


if __name__ == "__main__":
    raise RuntimeError
