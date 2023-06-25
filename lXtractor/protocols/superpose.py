"""
A sandbox module to encapsulate high-level operations based on core
`lXtractor`'s functionality.
"""
import logging
import typing as t
from collections import abc, namedtuple
from itertools import combinations

import biotite.structure as bst
import numpy as np
from toolz import curry

from lXtractor.core.chain import ChainStructure
from lXtractor.core.chain.structure import filter_selection_extended, subset_to_matching
from lXtractor.core.exceptions import MissingData, LengthMismatch, InitError
from lXtractor.util import apply
from lXtractor.util.seq import biotite_align
from lXtractor.util.structure import filter_to_common_atoms

LOGGER = logging.getLogger(__name__)

_Diff = namedtuple(
    "_Diff", ["SuperposeFixed", "RmsdFixed", "SuperposeMobile", "RmsdMobile"]
)
_InpSuperpose: t.TypeAlias = tuple[str, bst.AtomArray, bst.AtomArray | None]
_InpAlignSuperpose: t.TypeAlias = tuple[str, ChainStructure, ChainStructure]
_OutSuperpose = t.TypeAlias = tuple[
    str, str, float, t.Any, tuple[np.ndarray, np.ndarray, np.ndarray]
]
_StagedSupInp = t.TypeVar("_StagedSupInp", _InpSuperpose, _InpAlignSuperpose)
_DistFn = t.TypeAlias = abc.Callable[[bst.AtomArray, bst.AtomArray], t.Any]

SuperposeOutput = t.NamedTuple(
    "SuperposeOutput",
    [
        ("ID_fix", str),
        ("ID_mob", str),
        ("RmsdSuperpose", float),
        ("RmsdTarget", t.Any),
        ("Transformation", tuple[np.ndarray, np.ndarray, np.ndarray]),
    ],
)


@curry
def _stage_inp(
    c: ChainStructure,
    selection_superpose: tuple[
        abc.Sequence[int] | None,
        abc.Sequence[abc.Sequence[str]] | abc.Sequence[str] | None,
    ],
    selection_rmsd: tuple[
        abc.Sequence[int] | None,
        abc.Sequence[abc.Sequence[str]] | abc.Sequence[str] | None,
    ],
    map_name: str | None,
    exclude_hydrogen: bool,
    to_array: bool,
    tolerate_missing: bool,
) -> _InpSuperpose | _InpAlignSuperpose:
    def init_sub_chain(a):
        try:
            return ChainStructure.from_structure(a, c.pdb.id, c.pdb.chain)
        except Exception as e:
            raise InitError(
                f"Failed to create ChainStructure from array {a} for {c}"
            ) from e

    pos_sup, atoms_sup = selection_superpose
    pos_rmsd, atoms_rmsd = selection_rmsd

    if pos_rmsd is None or atoms_rmsd is None:
        pos_rmsd, atoms_rmsd = pos_sup, atoms_sup

    mask_sup = filter_selection_extended(
        c,
        pos=pos_sup,
        atom_names=atoms_sup,
        map_name=map_name,
        exclude_hydrogen=exclude_hydrogen,
        tolerate_missing=tolerate_missing,
    )
    mask_rmsd = filter_selection_extended(
        c,
        pos=pos_rmsd,
        atom_names=atoms_rmsd,
        map_name=map_name,
        exclude_hydrogen=exclude_hydrogen,
        tolerate_missing=tolerate_missing,
    )

    a_sup, a_rmsd = c.array[mask_sup], c.array[mask_rmsd]

    if len(a_sup) == 0:
        raise MissingData(f"Empty selection for superposition atoms in structure {c}")

    if len(a_rmsd) == 0:
        raise MissingData(f"Empty selection for RMSD atoms in structure {c}")

    if to_array:
        return c.id, a_sup, a_rmsd

    c_sup = init_sub_chain(a_sup)
    c_rmsd = init_sub_chain(a_rmsd)

    return c.id, c_sup, c_rmsd


def _yield_staged_pairs(
    fixed: abc.Iterable[ChainStructure],
    mobile: abc.Iterable[ChainStructure] | None,
    stage: abc.Callable[[ChainStructure], _StagedSupInp],
) -> abc.Generator[tuple[_StagedSupInp, _StagedSupInp], None, None]:
    _fixed = map(stage, fixed)
    if mobile is None:
        yield from combinations(_fixed, 2)
    else:
        _mobile = list(map(stage, mobile))
        for fs in _fixed:
            for ms in _mobile:
                yield fs, ms


def superpose_pair(
    pair: tuple[_InpSuperpose, _InpSuperpose], dist_fn: _DistFn | None
) -> _OutSuperpose:
    """
    A function performing superposition and rmsd calculation of already prepared
    :class:`AtomArray` objects. Each must have the same number of atoms.

    :param pair: A pair of staged inputs. A staged input is a tuple with an
        identifier, an atom array to superpose, and an optional atom array for
        the `dist_fn`.
    :param dist_fn: An optional distance function accepting two positional args:
        "fixed" atom array and superposed atom array.
    :return: a tuple with id_fixed, id_mobile, rmsd of the superposed atoms,
        calculated distance, and the transformation matrices.
    """
    # f_ for fixed, m_ for mobile
    (f_id, f_array_sup, f_array_dist), (m_id, m_array_sup, m_array_dist) = pair

    if len(f_array_sup) != len(m_array_sup):
        raise LengthMismatch(
            "For superposition, expected fixed and mobile array to have "
            f"the same number of atoms, but {len(f_array_sup)} != {len(m_array_sup)}"
        )

    _, transformation = bst.superimpose(f_array_sup, m_array_sup)

    m_array_sup = bst.superimpose_apply(m_array_sup, transformation)
    rmsd_sup = bst.rmsd(f_array_sup, m_array_sup)

    if all(x is not None for x in [dist_fn, f_array_dist, m_array_dist]):
        m_array_dist = bst.superimpose_apply(m_array_dist, transformation)
        dist = dist_fn(f_array_dist, m_array_dist)
    else:
        dist = None

    return f_id, m_id, rmsd_sup, dist, transformation


def align_and_superpose_pair(
    pair: tuple[_InpAlignSuperpose, _InpAlignSuperpose],
    dist_fn: _DistFn | None,
    skip_aln_if_match: str,
) -> _OutSuperpose:
    """
    A lower-level function performing superposition and rmsd calculation
    of already prepared :class:`AtomArray`'s.

    :param pair: A pair of staged inputs.
    :param dist_fn: An optional distance function accepting two positional args:
        "fixed" atom array and superposed atom array.
    :param skip_aln_if_match: Passed to
        :func:`lXtractor.core.chain.subset_to_matching`.
    :return: a tuple with id_fixed, id_mobile, rmsd of the superposed atoms,
        calculated distance, and the transformation matrices.
    """
    (f_id, f_str_sup, f_str_dist), (m_id, m_str_sup, m_str_dist) = pair

    f_str_aln, m_str_aln = subset_to_matching(
        f_str_sup,
        m_str_sup,
        skip_if_match=skip_aln_if_match,
        align_method=biotite_align,
        name="Mobile",
    )
    f_mask, m_mask = filter_to_common_atoms(
        f_str_aln.array, m_str_aln.array, allow_residue_mismatch=True
    )

    return superpose_pair(
        (
            (f_id, f_str_aln.array[f_mask], f_str_dist.array),
            (m_id, m_str_aln.array[m_mask], m_str_dist.array),
        ),
        dist_fn,
    )


def superpose_pairwise(
    fixed: abc.Iterable[ChainStructure],
    mobile: abc.Iterable[ChainStructure] | None = None,
    selection_superpose: tuple[
        abc.Sequence[int] | None,
        abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None,
    ] = (None, None),
    selection_dist: tuple[
        abc.Sequence[int] | None,
        abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None,
    ] = (None, None),
    dist_fn: _DistFn | None = None,
    *,
    strict: bool = True,
    map_name: str | None = None,
    exclude_hydrogen: bool = False,
    skip_aln_if_match: str = "len",
    verbose: bool = False,
    num_proc: int = 1,
    **kwargs,
) -> abc.Generator[SuperposeOutput, None, None]:
    """

    Superpose pairs of structures.
    Two modes are available:

        1. ``strict=True`` -- potentially faster and memory efficient,
        more parallelization friendly. In this case, after selection using
        the provided positions and atoms, the number of atoms between each
        fixed and mobile structure must match exactly.

        2. ``strict=False`` -- a "flexible" protocol. In this case, after the
        selection of atoms, there are two additional steps:

            1. Sequence alignment between the selected subsets. It's guaranteed
            to produce the same number of residues between fixed and mobile,
            which may be less than the initially selected number
            (see :func:`subset_to_matching`).

            2. Following this, subset each pair of residues between fixed and
            mobile to a common list of atoms (see :func:`filter_to_common_atoms
            <lXtractor.util.structure.filter_to_common_atoms>`).

    As a result, the "flexible" mode may be suitable for distantly related
    structures, while the "strict" mode may be used whenever it's guaranteed
    that the selection will produce the same sets of atoms between fixed and
    mobile.

    .. seealso::
        :func:`lXtractor.util.structure.filter_selection_extended` --
        used to apply the selections.

    :param fixed: An iterable over chain structures that won't be moved.
    :param mobile: An iterable over chain structures to superpose onto
        fixed ones. If ``None``, will use the combinations of `fixed`.
    :param selection_superpose: A tuple with (residue positions, atom names)
        to select atoms for superposition. Will be applied to each `fixed`
        and `mobile` structure.
    :param selection_dist: A tuple with (residue positions, atom names)
        to select atoms for `dist_fn` calculation.
    :param dist_fn: An optional distance function applied to a pair of
        superposed atom arrays, possibly different from the arrays selected
        for superposition, which is controlled via `selection_dist`.
    :param map_name: Mapping for positions in both selection arguments.
        If used, must exist within :attr:`Seq <lXtractor.core.chain.
        ChainStructure.seq>` of each fixed and mobile structure.
        A good candidate is a mapping to a reference sequence or
        :class:`Alignment <lXtractor.core.alignment.Alignment>`.
    :param exclude_hydrogen: Exclude all hydrogen atoms during selection.
    :param strict: Enable/disable the "strict" protocol. See the explanation
        above.
    :param skip_aln_if_match: Skip the sequence alignment if this field matches.
    :param verbose: Display progress bar.
    :param num_proc: The number of parallel processes. For large selections,
        may consume a lot of RAM, so caution advised.
    :param kwargs: Passed to :meth:`ProcessPoolExecutor.map`. Useful for
        controlling `chunksize` and `timeout` parameters.
    :return: A generator of ``namedtuple`` outputs each containing the IDs of
        the superposed objects, the RMSD between superposed structures,
        the distance function output, and the transformation matrices.
    """

    stage = _stage_inp(  # pylint: disable=no-value-for-parameter
        selection_superpose=selection_superpose,
        selection_rmsd=selection_dist,
        map_name=map_name,
        exclude_hydrogen=exclude_hydrogen,
        to_array=strict,
        tolerate_missing=not strict,
    )
    pairs = _yield_staged_pairs(fixed, mobile, stage)

    n = None
    if verbose:
        if isinstance(fixed, abc.Sized):
            if isinstance(mobile, abc.Sized):
                n = len(fixed) * len(mobile)
            else:
                n = int(len(fixed) * (len(fixed) - 1) / 2)

    fn = (
        curry(superpose_pair)(dist_fn=dist_fn)
        if strict
        else curry(align_and_superpose_pair)(  # pylint: disable=no-value-for-parameter
            skip_aln_if_match=skip_aln_if_match,
            dist_fn=dist_fn
        )
    )
    results = apply(fn, pairs, verbose, "Superposing pairs", num_proc, n, **kwargs)
    yield from map(lambda x: SuperposeOutput(*x), results)


if __name__ == "__main__":
    raise RuntimeError
