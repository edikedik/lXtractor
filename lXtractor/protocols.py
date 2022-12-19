"""
A sandbox module to encapsulate high-level operations based on core
`lXtractor`'s functionality.
"""
import logging
import typing as t
from collections import abc, namedtuple
from concurrent.futures import ProcessPoolExecutor
from itertools import starmap, combinations

import biotite.structure as bst
import numpy as np
from more_itertools import unzip
from toolz import curry
from tqdm.auto import tqdm

from lXtractor.core.chain import ChainStructure
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import MissingData, LengthMismatch
from lXtractor.util.seq import biotite_align
from lXtractor.util.structure import filter_selection, filter_to_common_atoms

LOGGER = logging.getLogger(__name__)

_Diff = namedtuple(
    'SubsetDiff', ['SuperposeFixed', 'RmsdFixed', 'SuperposeMobile', 'RmsdMobile']
)
_StagedSupInpStrict: t.TypeAlias = tuple[str, bst.AtomArray, bst.AtomArray]
_StagedSupInpFlex: t.TypeAlias = tuple[str, ChainStructure, ChainStructure]
_StagedSupInp = t.TypeVar('_StagedSupInp', _StagedSupInpStrict, _StagedSupInpFlex)

_SupOutputStrict = namedtuple(
    'OutputStrict', ['ID1', 'ID2', 'RmsdSuperpose', 'RmsdTarget', 'Transformation']
)
_SupOutputStrictT: t.TypeAlias = tuple[
    str, str, float, float, tuple[np.ndarray, np.ndarray, np.ndarray]
]
_SupOutputFlex = namedtuple(
    'OutputFlex',
    [
        'ID1',
        'ID2',
        'RmsdSuperpose',
        'RmsdTarget',
        'Transformation',
        'DiffSeq',
        'DiffAtoms',
    ],
)
_SupOutputFlexT: t.TypeAlias = tuple[
    str,
    str,
    float,
    float,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
_SupOutputT = t.TypeVar('_SupOutputT', _SupOutputStrict, _SupOutputFlex)


def filter_selection_extended(
    c: ChainStructure,
    pos: abc.Sequence[int] | None = None,
    atom_names: abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None = None,
    map_name: str | None = None,
    exclude_hydrogen: bool = False,
    tolerate_missing: bool = False,
) -> np.ndarray:
    """
    Get mask for certain positions and atoms of a chain structure.

    .. seealso:
        :func:`lXtractor.util.seq.filter_selection`

    :param c: Arbitrary chain structure.
    :param pos: A sequence of positions.
    :param atom_names: A sequence of atom names (broadcasted to each position
        in `res_id`) or an iterable over such sequences for each position
        in `res_id`.
    :param map_name: A map name to map from `pos` to
        :meth:`numbering <lXtractor.core.Chain.ChainSequence.numbering>`
    :param exclude_hydrogen: For convenience, exclude hydrogen atoms.
        Especially useful during pre-processing for superposition.
    :param tolerate_missing: If certain positions failed to map, does not
        raise an error.
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
    reference: ChainStructure,
    c: ChainStructure,
    map_name: str | None = None,
    skip_if_match: str = SeqNames.seq1,
    **kwargs,
) -> tuple[ChainStructure, ChainStructure]:
    """
    Subset both chain structures to aligned residues using
    **sequence alignment**.

    .. note::
        It's not necessary, but it makes sense for `c1` and `c2` to be
        somehow related.

    :param reference: A chain structure to align to.
    :param c: A chain structure to align.
    :param map_name: If provided, `c` is considered "pre-aligned" to the
        `reference`, and `reference` possessed the numbering under `map_name`.
    :param skip_if_match: Two options:

        1. Sequence/Map name, e.g., "seq1" -- if sequences under this name
        match exactly, skip alignment and return original chain structures.

        2. "len" -- if sequences have equal length, skip alignment and return
        original chain structures.
    :return: A pair of new structures having the same number of residues
        that were successfully matched during the alignment.
    """
    if skip_if_match == 'len':
        if len(reference.seq) == len(c.seq):
            return reference, c
    else:
        if reference.seq[skip_if_match] == c.seq[skip_if_match]:
            return reference, c

    if not map_name:
        pos2 = reference.seq.map_numbering(c.seq, **kwargs)
        pos1 = reference.seq[SeqNames.enum]
        pos_pairs = zip(pos1, pos2, strict=True)
    else:
        pos_pairs = zip(
            reference.seq[SeqNames.enum], reference.seq[map_name], strict=True
        )

    pos_pairs = filter(lambda x: x[0] is not None and x[1] is not None, pos_pairs)
    pos1, pos2 = map(list, unzip(pos_pairs))

    c1_new, c2_new = map(
        lambda x: ChainStructure.from_structure(
            x[0].array[filter_selection(x[0].array, x[1])]
        ),
        [(reference, pos1), (c, pos2)],
    )

    return c1_new, c2_new


def superpose(fs: _StagedSupInpStrict, ms: _StagedSupInpStrict) -> _SupOutputStrictT:
    """
    A lower-level function performing superposition and rmsd calculation
    of already prepared :class:`AtomArray`'s.

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
            f'the same number of atoms, but {len(fs_rmsd)} != {len(ms_rmsd)}'
        )

    _, transformation = bst.superimpose(fs_sup, ms_sup)

    ms_rmsd = bst.superimpose_apply(ms_rmsd, transformation)
    ms_sup = bst.superimpose_apply(ms_sup, transformation)

    rmsd_sup = bst.rmsd(fs_sup, ms_sup)
    rmsd_target = bst.rmsd(fs_rmsd, ms_rmsd)

    return f_id, m_id, rmsd_sup, rmsd_target, transformation


@curry
def _stage_inp(
    c: ChainStructure,
    selection_superpose: tuple[
        abc.Sequence[int] | None,
        abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None,
    ],
    selection_rmsd: tuple[
        abc.Sequence[int] | None,
        abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None,
    ],
    map_name: str | None,
    exclude_hydrogen: bool,
    to_array: bool,
    tolerate_missing: bool,
) -> _StagedSupInpStrict | _StagedSupInpFlex:
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
        raise MissingData(f'Empty selection for superposition atoms in structure {c}')

    if len(a_rmsd) == 0:
        raise MissingData(f'Empty selection for RMSD atoms in structure {c}')

    if to_array:
        return c.id, a_sup, a_rmsd

    c_sup, c_rmsd = map(
        lambda x: ChainStructure.from_structure(x, c.pdb.id), [a_sup, a_rmsd]
    )

    return c.id, c_sup, c_rmsd


def _yield_staged_pairs(
    fixed: abc.Iterable[ChainStructure],
    mobile: abc.Iterable[ChainStructure] | None,
    stage: abc.Callable[[ChainStructure], _StagedSupInp],
) -> abc.Generator[_StagedSupInp, _StagedSupInp]:
    _fixed = map(stage, fixed)
    if mobile is None:
        yield from combinations(_fixed, 2)
    else:
        _mobile = list(map(stage, mobile))
        for fs in _fixed:
            for ms in _mobile:
                yield fs, ms


@curry
def _align_and_superpose(
    fs: _StagedSupInpFlex, ms: _StagedSupInpFlex, skip_aln_if_match: str
) -> _SupOutputFlexT:
    def subset_to_common(c1, c2):
        m1, m2 = filter_to_common_atoms(c1.array, c2.array, allow_residue_mismatch=True)
        c1_sub, c2_sub = c1.array[m1], c2.array[m2]
        return c1_sub, c2_sub, len(c1_sub) / len(c1.array), len(c2_sub) / len(c2.array)

    (id1, c_sup1, c_rmsd1), (id2, c_sup2, c_rmsd2) = fs, ms
    c_sup1_aligned, c_sup2_aligned = subset_to_matching(
        c_sup1,
        c_sup2,
        skip_if_match=skip_aln_if_match,
        align_method=biotite_align,
        name='Mobile',
    )
    c_rmsd1_aligned, c_rmsd2_aligned = subset_to_matching(
        c_rmsd1,
        c_rmsd2,
        skip_if_match=skip_aln_if_match,
        align_method=biotite_align,
        name='Mobile',
    )

    a_sup1, a_sup2, sup1_diff, sup2_diff = subset_to_common(
        c_sup1_aligned, c_sup2_aligned
    )
    a_rmsd1, a_rmsd2, rmsd1_diff, rmsd2_diff = subset_to_common(
        c_rmsd1_aligned, c_rmsd2_aligned
    )

    # wrap into namedtuples later
    diff_seqs = (
        len(c_sup1.seq) - len(c_sup1_aligned.seq),
        len(c_rmsd1.seq) - len(c_rmsd1_aligned.seq),
        len(c_sup2.seq) - len(c_sup2_aligned.seq),
        len(c_rmsd2.seq) - len(c_rmsd2_aligned.seq),
    )
    diff_atoms = (sup1_diff, rmsd1_diff, sup2_diff, rmsd2_diff)

    id1, id2, rmsd_sup, rsmd_target, transformation = superpose(
        (id1, a_sup1, a_rmsd1), (id2, a_sup2, a_rmsd2)
    )

    return id1, id2, rmsd_sup, rsmd_target, transformation, diff_seqs, diff_atoms


def superpose_pairwise(
    fixed: abc.Iterable[ChainStructure],
    mobile: abc.Iterable[ChainStructure] | None = None,
    selection_superpose: tuple[
        abc.Sequence[int] | None,
        abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None,
    ] = (None, None),
    selection_rmsd: tuple[
        abc.Sequence[int] | None,
        abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None,
    ] = (None, None),
    map_name: str | None = None,
    exclude_hydrogen: bool = False,
    strict: bool = True,
    skip_aln_if_match: str = 'len',
    verbose: bool = False,
    num_proc: int | None = None,
    **kwargs,
) -> abc.Generator[_SupOutputT]:
    """

    Superpose pairs of structures.
    Two modes are available:

        1. ``strict=True`` -- potentially faster and memory efficient,
        more parallelization friendly. In this case, after selection using
        the provided positions and atoms, the number of atoms between each
        fixed and mobile structure must match exactly.

        2. ``strict=False`` -- a "flexible" protocol. In this case, aftter the
        selection of atoms, there are two additional steps:

            1. Sequence alignment between the selected subsets. It's guaranteed
            to produce the same number of residues between fixed and mobile,
            which may be less than the initially selected number
            (see :func:`subset_to_matching`).

            2. Following this, subset each pair of residues betwen fixed and
            mobile to a common list of atoms (see :func:`filter_to_common_atoms
            <lXtractor.util.structure.filter_to_common_atoms>`).

    As a result, the "flexible" mode may be suitable for distantly related
    structures, while the "strict" mode may be used whenever it's guaranteed
    that the selection will produce the same sets of atoms between fixed and
    mobile.

    .. seealso::
        :func:`filter_selection_extended` -- for selection arguments breakdown.

        :func:`biotite.structure.superimpose`

    :param fixed: An iterable over chain structures that won't be moved.
    :param mobile: An iterable over chain structures to superpose onto
        fixed ones. If ``None``, will use the combinations of `fixed`.
    :param selection_superpose: A tuple with (residue positions, atom names)
        to select atoms for superposition. Will be applied to each `fixed`
        and `mobile` structure.
    :param selection_rmsd: A tuple with (residue positions, atom names)
        to select atoms for RMSD calculation.
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
    :return: A generator over tuples (id_fixed, id_mobile, rmsd,
        transformation) where transformation is a set of vectors and matrices
        used to superpose the mobile structure. It can be used directly with
        :func:`biotite.structure.superimpose_apply`.
    """

    def wrap_output(res):
        if strict:
            return _SupOutputStrict(*res)
        id1, id2, rmsd_sup, rmsd_tar, tr, diff_seq, diff_atoms = res
        return _SupOutputFlex(
            id1, id2, rmsd_sup, rmsd_tar, tr, _Diff(*diff_seq), _Diff(*diff_atoms)
        )

    stage = _stage_inp(  # pylint: disable=no-value-for-parameter
        selection_superpose=selection_superpose,
        selection_rmsd=selection_rmsd,
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
        superpose
        if strict
        else _align_and_superpose(  # pylint: disable=no-value-for-parameter
            skip_aln_if_match=skip_aln_if_match
        )
    )

    if num_proc is not None and num_proc > 1:
        with ProcessPoolExecutor(num_proc) as executor:
            results = map(wrap_output, executor.map(fn, *unzip(pairs), **kwargs))
            if verbose:
                results = tqdm(results, desc='Superposing pairs', total=n)
            yield from results
    else:
        results = map(wrap_output, starmap(fn, pairs))
        if verbose:
            results = tqdm(results, desc='Superposing pairs', total=n)
        yield from results


if __name__ == '__main__':
    raise RuntimeError
