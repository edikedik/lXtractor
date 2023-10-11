"""
A module encompassing the :class:`ChainInitializer` used to init ``Chain*``-type
objects from various input types. It enables parallelization of reading structures
and seq2seq mappings and is flexible thanks to callbacks.
"""
from __future__ import annotations

import logging
import typing as t
from collections import abc
from concurrent.futures import Future
from itertools import chain
from pathlib import Path

from biotite import structure as bst
from more_itertools import split_into, collapse, ilen
from toolz import curry, compose_left
from tqdm.auto import tqdm

from lXtractor.core.chain import ChainList, map_numbering_many2many
from lXtractor.core.config import STRUCTURE_EXT, DefaultConfig
from lXtractor.core.exceptions import InitError
from lXtractor.core.structure import (
    GenericStructure,
    ProteinStructure,
    NucleotideStructure,
    CarbohydrateStructure,
)
from lXtractor.util.io import parse_suffix
from lXtractor.util.misc import apply
from lXtractor.util.seq import biotite_align

LOGGER = logging.getLogger(__name__)
_STRUCTURE_TYPES = {
    "a": GenericStructure,
    "n": NucleotideStructure,
    "p": ProteinStructure,
    "c": CarbohydrateStructure,
}

if t.TYPE_CHECKING:
    from lXtractor.core.chain import Chain, ChainStructure, ChainSequence

    CT = t.TypeVar("CT", ChainStructure, ChainSequence, Chain)
    _O: t.TypeAlias = ChainSequence | ChainStructure | list[ChainStructure] | None

__all__ = (
    "SingletonCallback",
    "ItemCallback",
    "ChainInitializer",
)


class SingletonCallback(t.Protocol):
    """
    A protocol defining signature for a callback used with
    :class:`ChainInitializer` on single objects right after parsing.
    """

    @t.overload
    def __call__(self, inp: CT) -> CT | None:
        ...

    @t.overload
    def __call__(self, inp: list[ChainStructure]) -> list[ChainStructure] | None:
        ...

    @t.overload
    def __call__(self, inp: None) -> None:
        ...

    def __call__(
        self, inp: CT | list[ChainStructure] | None
    ) -> CT | list[ChainStructure] | None:
        ...


class ItemCallback(t.Protocol):
    """
    A callback applied to processed items in
    :meth:`ChainInitializer.from_mapping`.
    """

    def __call__(
        self, inp: tuple[Chain, list[ChainStructure]]
    ) -> tuple[Chain | None, list[ChainStructure]]:
        ...


def _read_path(
    path: Path,
    tolerate_failures: bool,
    supported_seq_ext: abc.Container[str],
    supported_str_ext: abc.Container[str],
) -> ChainSequence | list[ChainStructure] | None:
    from lXtractor.core.chain import ChainStructure, ChainSequence

    suffix = parse_suffix(path)

    if suffix in supported_seq_ext:
        return ChainSequence.from_file(path)
    if suffix in supported_str_ext:
        # TODO: This takes only the first altloc but should take all of them
        # Read initial structures, split by chains ant altloc
        # and wrap into a ChainStructure
        pol_type = DefaultConfig["structure"]["primary_pol_type"][0]
        structure_cls = _STRUCTURE_TYPES[pol_type]
        structure = next(structure_cls.read(path, altloc=True).split_altloc())
        return list(map(ChainStructure, structure.split_chains(polymer=True)))
    if tolerate_failures:
        return None
    raise InitError(f"Suffix {suffix} of the path {path} is not supported")


def _init(
    inp: t.Any,
    tolerate_failures: bool,
    supported_seq_ext: list[str],
    supported_str_ext: list[str],
    callbacks: list[SingletonCallback] | None,
) -> _O:
    from lXtractor.core.chain import Chain, ChainStructure, ChainSequence

    res: _O
    match inp:
        case Chain() | ChainSequence() | ChainStructure():
            res = inp
        case [str(), str()]:
            res = ChainSequence.from_string(inp[1], name=inp[0])
        case [Path(), xs]:
            structures = _read_path(
                inp[0], tolerate_failures, supported_seq_ext, supported_str_ext
            )
            structures = [s for s in structures if s.chain_id in xs]
            res = structures or None
        case (
            GenericStructure()
            | ProteinStructure()
            | NucleotideStructure()
            | CarbohydrateStructure()
        ):
            res = ChainStructure(inp)
        case Path():
            res = _read_path(
                inp, tolerate_failures, supported_seq_ext, supported_str_ext
            )
        case _:
            res = None
            if not tolerate_failures:
                raise InitError(f"Unsupported input type {type(inp)}")
    if callbacks:
        for c in callbacks:
            res = c(res)
    return res


def _try_fn(inp, fn, tolerate_failures):
    try:
        return fn(inp)
    except Exception as e:
        LOGGER.warning(f"Input {inp} failed with an error {e}")
        if not tolerate_failures:
            raise e
        return None


class ChainInitializer:
    """
    In contrast to :class:`ChainIO`, this object initializes new
    :class:`Chain`, :class:`ChainStructure`, or :class:`Chain` objects from
    various input types.

    To initialize :class:`Chain` objects, use :meth:`from_mapping`.

    To initialize :class:`ChainSequence` or :class:`ChainStructure` objects,
    use :meth:`from_iterable`.

    """

    def __init__(self, tolerate_failures: bool = False, verbose: bool = False):
        """
        :param tolerate_failures: Don't stop the execution if some object fails
            to initialize.
        :param verbose: Output progress bars.
        """
        self.tolerate_failures = tolerate_failures
        self.verbose = verbose

    @property
    def supported_seq_ext(self) -> list[str]:
        """
        :return: Supported sequence file extensions.
        """
        return [".fasta"]

    @property
    def supported_str_ext(self) -> list[str]:
        """
        :return: Supported structure file extensions.
        """
        return [*STRUCTURE_EXT, *(e + ".gz" for e in STRUCTURE_EXT)]

    def from_iterable(
        self,
        it: abc.Iterable[
            ChainSequence
            | ChainStructure
            | Path
            | tuple[Path, abc.Sequence[str]]
            | tuple[str, str]
            | GenericStructure
        ],
        num_proc: int = 1,
        callbacks: abc.Sequence[SingletonCallback] | None = None,
        desc: str = "Initializing objects",
    ) -> abc.Generator[_O | Future, None, None]:
        """
        Initialize :class:`ChainSequence`s or/and :class:`ChainStructure`'s
        from (possibly heterogeneous) iterable.

        :param it:
            Supported elements are:
                1) Initialized objects (passed without any actions).
                2) Path to a sequence or a structure file.
                3) (Path to a structure file, list of target chains).
                4) A pair (header, _seq) to initialize a :class:`ChainSequence`.
                5) A :class:`GenericStructure` with a single chain.

        :param num_proc: The number of processes to use.
        :param callbacks: A sequence of callables accepting and returning an
            initialized object.
        :param desc: Progress bar description used if :attr:`verbose` is
            ``True``.
        :return: A generator yielding initialized chain sequences and
            structures parsed from the inputs.
        """

        __init = curry(_init)(
            tolerate_failures=self.tolerate_failures,
            supported_seq_ext=self.supported_seq_ext,
            supported_str_ext=self.supported_str_ext,
            callbacks=callbacks,
        )
        __try_fn = curry(_try_fn, fn=__init, tolerate_failures=self.tolerate_failures)

        yield from apply(__try_fn, it, self.verbose, desc, num_proc)

    def from_mapping(
        self,
        m: abc.Mapping[
            ChainSequence | Chain | tuple[str, str] | Path,
            abc.Sequence[
                ChainStructure
                | GenericStructure
                | bst.AtomArray
                | Path
                | tuple[Path, abc.Sequence[str]]
            ],
        ],
        key_callbacks: abc.Sequence[SingletonCallback] | None = None,
        val_callbacks: abc.Sequence[SingletonCallback] | None = None,
        item_callbacks: abc.Sequence[ItemCallback] | None = None,
        *,
        map_numberings: bool = True,
        num_proc_read_seq: int = 1,
        num_proc_read_str: int = 1,
        num_proc_item_callbacks: int = 1,
        num_proc_map_numbering: int = 1,
        num_proc_add_structure: int = 1,
        **kwargs,
    ) -> ChainList[Chain]:
        """
        Initialize :class:`Chain`'s from mapping between sequences and
        structures.

        It will first initialize objects to which the elements of `m`
        refer (see below) and then create maps between each sequence and
        associated structures, saving these into structure
        :attr:`ChainStructure._seq`'s.

        .. note::
            ``key/value_callback`` are distributed to parser and applied right
            after parsing the object. As a result, their application will
            be parallelized depending on the``num_proc_read_seq`` and
            ``num_proc_read_str`` parameters.

        :param m:
            A mapping of the form ``{_seq => [structures]}``, where `_seq`
            is one of:

                1) Initialized :class:`ChainSequence`.
                2) A pair (header, _seq).
                3) A path to a **fasta** file containing a single sequence.

            While each structure is one of:

                1) Initialized :class:`ChainStructure`.
                2) :class:`GenericStructure` with a single chain.
                3) :class:`biotite.AtomArray` corresponding to a single chain.
                4) A path to a structure file.
                5) (A path to a structure file, list of target chains).

            In the latter two cases, the chains will be expanded
            and associated with the same sequence.

        :param key_callbacks: A sequence of callables accepting and returning
            a :class:`ChainSequence`.
        :param val_callbacks: A sequence of callables accepting and returning
            a :class:`ChainStructure`.
        :param item_callbacks: A sequence of callables accepting and returning
            a parsed item -- a tuple of :class:`Chain` and a sequence of
            associated :class:`ChainStructure`s. Callbacks are applied
            sequentially to each item as a function composition in the supplied
            order (left to right). It the last callback returns ``None`` as a
            first element or an empty list as a second element, such item will
            be filtered out. Item callbacks are applied after parsing sequences
            and structures and converting chain sequences to chains.
        :param map_numberings: Map PDB numberings to canonical sequence's
            numbering via pairwise sequence alignments.
        :param num_proc_read_seq: A number of processes to devote to sequence
            parsing. Typically, sequence reading doesn't benefit from parallel
            processing, so it's better to leave this default.
        :param num_proc_read_str: A number of processes dedicated to structures
            parsing.
        :param num_proc_item_callbacks: A number of CPUs to parallelize item
            callbacks' application.
        :param num_proc_map_numbering: A number of processes to use for mapping
            between numbering of sequences and structures. Generally, this
            should be as high as possible for faster processing. In contrast
            to the other operations here, this one seems more CPU-bound and
            less resource hungry (although, keep in mind the size of the
            canonical sequence: if it's too high, the RAM usage will likely
            explode). If ``None``, will default to :attr:`num_proc`.
        :param num_proc_add_structure: In case of parallel numberings mapping,
            i.e, when ``num_proc_map_numbering > 1``, this option allows to
            transfer these numberings and add structures to chains in parallel.
            It may be useful to when ``add_to_children=True`` is passed in
            ``kwargs`` as it allows creating sub-structures in parallel.
        :param kwargs: Passed to :meth:`Chain.add_structure`.
        :return: A list of initialized chains.
        """

        from lXtractor.core.chain import Chain

        # Process keys and values
        keys = self.from_iterable(
            m,
            num_proc=num_proc_read_seq,
            callbacks=key_callbacks,
            desc="Initializing sequences",
        )  # ChainSequences
        values_flattened = self.from_iterable(  # ChainStructures
            chain.from_iterable(m.values()),
            num_proc=num_proc_read_str,
            callbacks=val_callbacks,
            desc="Initializing structures",
        )
        values = map(
            compose_left(  # Collapse all separated chains into a list
                collapse, lambda x: list(filter(bool, x))
            ),
            split_into(  # Split into original sizes
                values_flattened, map(len, m.values())
            ),
        )
        items = (
            (Chain(key) if not isinstance(key, Chain) else key, vs_group)
            for key, vs_group in zip(keys, values, strict=True)
            if key is not None and len(vs_group) > 0
        )

        if item_callbacks:
            fn = compose_left(*item_callbacks)
            # 1. Apply a callback to each item
            # 2. Filter out cases yielding empty seqs or structures
            items = apply(
                fn,
                items,
                self.verbose,
                "Applying item callbacks",
                num_proc_item_callbacks,
            )
            items = filter(lambda x: x[0] is not None and len(x[1]) > 0, items)

        items = list(items)
        if ilen(items) == 0:
            LOGGER.warning(
                "No items left after parsing and applying callbacks; "
                "returning an empty ChainList."
            )
            return ChainList([])

        if num_proc_map_numbering <= 1 or not map_numberings:
            items = (
                tqdm(items, desc="Adding structures sequentially")
                if self.verbose
                else items
            )
            if not map_numberings and "map_to_seq" not in kwargs:
                kwargs["map_to_seq"] = False
            for c, ss in items:
                for s in ss:
                    c.add_structure(s, align_method=biotite_align, **kwargs)
            chains = ChainList(x[0] for x in items)
        else:
            default_name = DefaultConfig["mapnames"]["map_canonical"]
            map_name = kwargs.get("map_name") or default_name

            # create numbering groups -- lists of lists with numberings
            # for each structure in values
            numbering_groups = list(
                map_numbering_many2many(
                    [x.seq for x, _ in items],
                    [[x.seq for x in strs] for _, strs in items],
                    num_proc=num_proc_map_numbering,
                    verbose=self.verbose,
                )
            )
            _fn = curry(
                _add_structures,
                map_name=map_name,
                tolerate_failures=self.tolerate_failures,
                **kwargs,
            )
            __try_fn = curry(_try_fn, fn=_fn, tolerate_failures=self.tolerate_failures)
            inputs = list(zip(items, numbering_groups, strict=True))
            chains = ChainList(
                apply(
                    __try_fn,
                    inputs,
                    self.verbose,
                    "Adding structures to chains",
                    num_proc_add_structure,
                )
            )

        return chains


def _add_structures(
    inp: tuple[tuple[Chain, abc.Iterable[ChainStructure]], list[list[int | None]]],
    map_name: str,
    tolerate_failures: bool,
    **kwargs,
):
    (c, ss), num_group = inp
    for s, n in zip(ss, num_group, strict=True):
        try:
            s.seq.add_seq(map_name, n)
            c.add_structure(s, map_to_seq=False, **kwargs)
        except Exception as e:
            LOGGER.warning(f"Failed to add structure {s} to chain {c} due to {e}")
            LOGGER.exception(e)
            if not tolerate_failures:
                raise e
    return c


if __name__ == "__main__":
    raise RuntimeError
