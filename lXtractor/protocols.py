"""
A sandbox module to encapsulate high-level operations based on core lXtractor's functionality.
"""
import logging
import typing as t
from collections import abc
from concurrent.futures import ProcessPoolExecutor
from itertools import starmap, combinations

import biotite.structure as bst
import numpy as np
from more_itertools import unzip
from tqdm.auto import tqdm

from lXtractor.core.chain import ChainStructure
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData, LengthMismatch
from lXtractor.util.structure import filter_selection

LOGGER = logging.getLogger(__name__)

_StagedSupInp: t.TypeAlias = tuple[str, bst.AtomArray, bst.AtomArray]
_SupOutput: t.TypeAlias = tuple[str, str, float, tuple[np.ndarray, np.ndarray, np.ndarray]]


def filter_selection_extended(
        c: ChainStructure, pos: abc.Sequence[int] | None = None,
        atom_names: abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None = None,
        map_name: str | None = None, exclude_hydrogen: bool = False,
        tolerate_missing: bool = False
) -> np.ndarray:
    """
    Get mask for certain positions and atoms of a chain structure.

    .. seealso:
        :func:`lXtractor.util.seq.filter_selection`

    :param c: Arbitrary chain structure.
    :param pos: A sequence of positions.
    :param atom_names: A sequence of atom names (broadcasted to each position in `res_id`)
        or an iterable over such sequences for each position in `res_id`.
    :param map_name: A map name to map from `pos` to
        :meth:`numbering <lXtractor.core.Chain.ChainSequence.numbering>`
    :param exclude_hydrogen: For convenience, exclude hydrogen atoms.
        Especially useful during pre-processing for superposition.
    :param tolerate_missing: If certain positions failed to map, does not raise an error.
    :return: A binary mask, ``True`` for selected atoms.
    """
    if pos is not None and map_name:
        _map = c.seq.get_map(map_name)

        pos = [(p, _map.get(p, None)) for p in pos]

        if not tolerate_missing:
            for p, p_mapped in pos:
                if p_mapped is None:
                    raise MissingData(f'Position {p} failed to map')

        pos = [x[1].numbering for x in pos if x[1] is not None]

        if len(pos) == 0:
            LOGGER.warning('No positions were selected.')
            return np.zeros_like(c.array, dtype=bool)

    m = filter_selection(c.array, pos, atom_names)

    if exclude_hydrogen:
        m &= c.array.element != 'H'

    return m


def subset_to_matching(
        reference: ChainStructure, c: ChainStructure,
        map_name: str | None = None, **kwargs
) -> tuple[ChainStructure, ChainStructure]:
    """
    Subset both chain structures to aligned residues using **sequence alignment**.

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


def superpose(fs: _StagedSupInp, ms: _StagedSupInp) -> _SupOutput:
    """
    A lower-level function performing superposition and rmsd calculation of already
    prepared :class:`AtomArray`'s.

    .. seealso::
        :func:`superpose_pairwise_strict`, :func:`superpose_pairwise_flex`

    :param fs: Staged input for a fixed structure.
    :param ms: Staged input for a mobile structure.
    :return: (id_fixed, id_mobile, rmsd, transformation)
    """
    f_id, fs_sup, fs_rmsd = fs
    m_id, ms_sup, ms_rmsd = ms

    if len(fs_sup) != len(ms_sup):
        raise LengthMismatch(
            'For superposition, expected fixed and mobile array to have '
            f'the same number of atoms, but {len(fs_sup)} != {len(ms_sup)}'
        )
    if len(fs_rmsd) != len(ms_rmsd):
        raise LengthMismatch(
            'For RMSD calculation, expected fixed and mobile array to have '
            f'the same number of atoms, but {len(fs_sup)} != {len(ms_sup)}'
        )

    superposed, transformation = bst.superimpose(fs_sup, ms_sup)

    ms_rmsd = bst.superimpose_apply(ms_rmsd, transformation)

    rmsd = bst.rmsd(fs_rmsd, ms_rmsd)

    return f_id, m_id, rmsd, transformation


def superpose_pairwise_strict(
        fixed: abc.Iterable[ChainStructure],
        mobile: abc.Iterable[ChainStructure] | None = None,
        selection_superpose: tuple[
            abc.Sequence[int] | None,
            abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None] = (None, None),
        selection_rmsd: tuple[
            abc.Sequence[int] | None,
            abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None] = (None, None),
        map_name: str | None = None,
        exclude_hydrogen: bool = False,
        verbose: bool = False,
        num_proc: int | None = None,
) -> abc.Generator[_SupOutput]:
    """
    Superpose pairs of structures using "strict" protocol, where every pair must have
    the same number of atoms for superposition and rmsd calculation.

    It's faster and more memory efficient than :func:superpose_pairwise_flex`.

    .. seealso::
        :func:`superpose_pairwise_flex` -- for a more flexible (but slower) approach.

        :func:`filter_selection_extended` -- for selection arguments breakdown.

        :func:`biotite.structure.superimpose` -- a function that actually superposes.

    :param fixed: An iterable over chain structures that won't be moved.
    :param mobile: An iterable over chain structures to superpose onto fixed ones.
        If ``None``, will use the combinations of `fixed`.
    :param selection_superpose: A tuple with (residue positions, atom names) to select
        atoms for superposition. Will be applied to each `fixed` and `mobile` structure.
    :param selection_rmsd: A tuple with (residue positions, atom names) to select
        atoms for RMSD calculation.
    :param map_name: Mapping for positions in both selection arguments. If used, must
        exist within :attr:`Seq <lXtractor.core.chain.ChainStructure.seq>` of each
        fixed and mobile structure.
        A good candidate is a mapping to a reference sequence or
        :class:`Alignment <lXtractor.core.alignment.Alignment>`.
    :param exclude_hydrogen: Exclude all hydrogen atoms during selection.
    :param verbose: Display progress bar.
    :param num_proc: The number of parallel processes. For large selections, may consume a
        lot of RAM, so caution advised.
    :return: A generator over tuples (id_fixed, id_mobile, rmsd, transformation) where
        transformation is a set of vectors and matrices used to superpose the mobile structure.
        It can be used directly with :func:`biotite.structure.superimpose_apply`.
    """

    def stage_inp(c: ChainStructure) -> _StagedSupInp:
        pos_sup, atoms_sup = selection_superpose
        pos_rmsd, atoms_rmsd = selection_rmsd

        if pos_rmsd is None or atoms_rmsd is None:
            pos_rmsd, atoms_rmsd = pos_sup, atoms_sup

        mask_sup = filter_selection_extended(
            c, pos=pos_sup, atom_names=atoms_sup, map_name=map_name,
            exclude_hydrogen=exclude_hydrogen)
        mask_rmsd = filter_selection_extended(
            c, pos=pos_rmsd, atom_names=atoms_rmsd, map_name=map_name,
            exclude_hydrogen=exclude_hydrogen)

        return c.id, c.array[mask_sup], c.array[mask_rmsd]

    def yield_staged_pairs():
        _fixed = map(stage_inp, fixed)
        if mobile is None:
            yield from combinations(_fixed, 2)
        else:
            _mobile = list(map(stage_inp, mobile))
            for fs in _fixed:
                for ms in _mobile:
                    yield fs, ms

    pairs = yield_staged_pairs()
    pairs = list(pairs)

    if verbose:
        pairs = tqdm(pairs, desc='Superposing pairs')

    if num_proc is not None and num_proc > 1:
        with ProcessPoolExecutor(num_proc) as executor:
            yield from executor.map(superpose, *unzip(pairs), chunksize=1)
    else:
        yield from starmap(superpose, pairs)


def superpose_pairwise_flex():
    pass


if __name__ == '__main__':
    raise RuntimeError
