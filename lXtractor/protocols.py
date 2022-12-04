"""
A sandbox module to encapsulate high-level operations based on core lXtractor's functionality.
"""
from collections import abc
from itertools import islice

import biotite.structure as bst
import numpy as np
from more_itertools import unzip

from lXtractor.core.chain import ChainStructure
from lXtractor.core.config import SeqNames
from lXtractor.util.structure import filter_selection, filter_to_common_atoms


# Select longest consecutive list of residues?
# Use sequence alignment to select common residues?


def filter_selection_extended(
        c: ChainStructure, pos: abc.Sequence[int] | None = None,
        atom_names: abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None = None,
        map_name: str | None = None, exclude_hydrogen: bool = False,
) -> np.ndarray:
    """
    Get mask for certain positions and atoms of a chain structure.

    .. seealso:
        :func:`lXtractor.util.seq.filter_selection`

    :param c: Arbitrary chain structure.
    :param pos: A sequence of positions.
    :param atom_names: A sequence of atom names. Or Iterable over such sequences.
        In this case, the number of elements in the latter must match the number
        of positions.
    :param map_name: A map name to map from `pos` to
        :meth:`numbering <lXtractor.core.Chain.ChainSequence.numbering>`
    :param exclude_hydrogen: For convenience, exclude hydrogen atoms.
        Especially useful during pre-processing for superposition.
    :return: A binary mask, ``True`` for selected atoms.
    """
    if pos is not None and map_name:
        _map = c.seq.get_map(map_name)
        pos = [_map[p].numbering for p in pos]

    m = filter_selection(c.array, pos, atom_names)

    if exclude_hydrogen:
        m &= c.array.element != 'H'

    return m


def subset_to_matching(
        reference: ChainStructure, c: ChainStructure,
        map_name: str | None = None, **kwargs
) -> tuple[ChainStructure, ChainStructure]:
    """
    Subset both chains structures to aligned residues using **sequence alignment**.

    .. note::
        It's not necessary, but it makes sense for `c1` and `c2` to be somehow related.

    :param reference: A chain structure to align to.
    :param c: A chain structure to align.
    :param map_name: If provided, `c` is considered "pre-aligned" to the `reference`, and
        `reference` possessed the numbering under `map_name`.
    :return: A pair of new structures having the same number of residues that were
        successfully matched during the alignment.
    """
    if not map_name:
        pos2 = reference.seq.map_numbering(c.seq, **kwargs)
        pos1 = reference.seq[SeqNames.enum]
        pos_pairs = zip(pos1, pos2, strict=True)
    else:
        pos_pairs = zip(reference.seq[SeqNames.enum], reference.seq[map_name], strict=True)

    pos_pairs = filter(
        lambda x: x[0] is not None and x[1] is not None,
        pos_pairs
    )
    pos1, pos2 = map(list, unzip(pos_pairs))

    c1_new, c2_new = map(
        lambda x: ChainStructure.from_structure(
            x[0].array[filter_selection(x[0].array, x[1])]),
        [(reference, pos1), (c, pos2)]
    )

    return c1_new, c2_new


def superpose_pair(
        fixed: ChainStructure,
        mobile: ChainStructure,
        pos: abc.Sequence[int] | None = None,
        atom_names: abc.Sequence[str] | None = None,
        map_name: str | None = None,
        exclude_hydrogen: bool = False,
        allow_residue_mismatch: bool = False,
):
    m_fixed, m_mobile = map(
        lambda x: filter_selection_extended(x, pos=pos, atom_names=atom_names, map_name=map_name,
                                            exclude_hydrogen=exclude_hydrogen),
        [fixed, mobile]
    )
    m_fixed, m_mobile = filter_to_common_atoms(
        fixed.array[m_fixed], mobile.array[m_mobile],
        allow_residue_mismatch=allow_residue_mismatch
    )


def superpose_pairwise(
        fixed: abc.Iterable[ChainStructure],
        mobile: abc.Iterable[ChainStructure] | None = None,
        pos: abc.Sequence[int] | None = None,
        map_name: str | None = None,
        exclude_hydrogen: bool = False,
        allow_residue_mismatch: bool = False,
        num_proc: int | None = None,
):
    def yield_pairs():
        _mobile = islice(fixed, 1, None) if mobile is None else mobile
        for fs in fixed:
            for ms in _mobile:
                yield fixed, mobile

    def get_atoms(fs: ChainStructure, ms: ChainStructure):
        if pos:
            if map_name:
                map_fs, map_ms = map(lambda x: x.seq.get_map(map_name), [fs, ms])


if __name__ == '__main__':
    raise RuntimeError
