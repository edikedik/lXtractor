from __future__ import annotations

import logging
import typing as t
from collections import abc
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat, chain
from pathlib import Path

from biotite import structure as bst
from more_itertools import unzip, split_into, collapse
from toolz import valmap, keymap, keyfilter
from tqdm.auto import tqdm

from lXtractor.core.alignment import Alignment
from lXtractor.core.chain.chain import Chain
from lXtractor.core.chain.sequence import ChainSequence
from lXtractor.core.chain.structure import ChainStructure
from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import InitError, LengthMismatch
from lXtractor.core.structure import GenericStructure

CT = t.TypeVar('CT', ChainStructure, ChainSequence, Chain)
LOGGER = logging.getLogger(__name__)


class InitializerCallback(t.Protocol):
    """
    A protocol defining signature for a callback used with
    :class:`ChainInitializer`.
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


def _read_path(x, tolerate_failures, supported_seq_ext, supported_str_ext):
    if x.suffix in supported_seq_ext:
        return ChainSequence.from_file(x)
    if x.suffix in supported_str_ext:
        return [
            ChainStructure.from_structure(c)
            for c in GenericStructure.read(x).split_chains()
        ]
    if tolerate_failures:
        return None
    raise InitError(f"Suffix {x.suffix} of the path {x} is not supported")


def _init(x, tolerate_failures, supported_seq_ext, supported_str_ext, callbacks):
    match x:
        case ChainSequence() | ChainStructure():
            res = x
        case [str(), str()]:
            res = ChainSequence.from_string(x[1], name=x[0])
        case [Path(), xs]:
            structures = _read_path(
                x[0], tolerate_failures, supported_seq_ext, supported_str_ext
            )
            structures = [s for s in structures if s.pdb.chain in xs]
            res = structures or None
        case GenericStructure():
            res = ChainStructure.from_structure(x)
        case Path():
            res = _read_path(x, tolerate_failures, supported_seq_ext, supported_str_ext)
        case _:
            res = None
            if not tolerate_failures:
                raise InitError(f"Unsupported input type {type(x)}")
    if callbacks:
        for c in callbacks:
            res = c(res)
    return res


def _map_numbering(seq1, seq2):
    return seq1.map_numbering(seq2, save=False)


def map_numbering_12many(
    obj_to_map: str | tuple[str, str] | ChainSequence | Alignment,
    seqs: abc.Iterable[ChainSequence],
    num_proc: t.Optional[int] = None,
) -> abc.Iterator[list[int | None]]:
    """
    Map numbering of a single sequence to many other sequences.

    **This function does not save mapped numberings.**

    .. seealso::
        :meth:`ChainSequence.map_numbering`.

    :param obj_to_map: Object whose numbering should be mapped to `seqs`.
    :param seqs: Chain sequences to map the numbering to.
    :param num_proc: A number of parallel processes to use.
        If ``None``, run sequentially.
    :return: An iterator over the mapped numberings.
    """
    if num_proc:
        with ProcessPoolExecutor(num_proc) as executor:
            yield from executor.map(_map_numbering, seqs, repeat(obj_to_map))
    else:
        yield from (x.map_numbering(obj_to_map, save=False) for x in seqs)


def map_numbering_many2many(
    objs_to_map: abc.Sequence[str | tuple[str, str] | ChainSequence | Alignment],
    seq_groups: abc.Sequence[abc.Sequence[ChainSequence]],
    num_proc: t.Optional[int] = None,
    verbose: bool = False,
) -> abc.Iterator[list[list[int | None]]]:
    """
    Map numbering of each object `o` in `objs_to_map` to each sequence
    in each group of the `seq_groups` ::

        o1 -> s1_1 s1_1 s1_3 ...
        o2 -> s2_1 s2_1 s2_3 ...
                  ...

    **This function does not save mapped numberings.**

    For a single object-group pair, it's the same as
    :func:`map_numbering_12many`. The benefit comes from parallelization
    of this functionality.

    .. seealso::
        :meth:`ChainSequence.map_numbering`.
        :func:`map_numbering_12many`

    :param objs_to_map: An iterable over objects whose numbering to map.
    :param seq_groups: Group of objects to map numbering to.
    :param num_proc: A number of processes to use. If ``None``,
        run sequentially.
    :param verbose: Output a progress bar.
    :return: An iterator over lists of lists with numeric mappings

    ::

         [[s1_1 map, s1_2 map, ...]
          [s2_1 map, s2_2 map, ...]
                    ...
          ]

    """
    if len(objs_to_map) != len(seq_groups):
        raise LengthMismatch(
            f"The number of objects to map {len(objs_to_map)} != "
            f"the number of sequence groups {len(seq_groups)}"
        )
    staged = chain.from_iterable(
        ((obj, s) for s in g) for obj, g in zip(objs_to_map, seq_groups)
    )
    group_sizes = map(len, seq_groups)
    if num_proc:
        objs, seqs = unzip(staged)
        with ProcessPoolExecutor(num_proc) as executor:
            results = executor.map(_map_numbering, seqs, objs, chunksize=1)
            if verbose:
                results = tqdm(results, desc="Mapping numberings")
            yield from split_into(results, group_sizes)
    else:
        results = (s.map_numbering(o, save=False) for o, s in staged)
        if verbose:
            results = tqdm(results, desc="Mapping numberings")
        yield from split_into(results, group_sizes)


class ChainInitializer:
    """
    In contrast to :class:`ChainIO`, this object initializes new
    :class:`Chain`, :class:`ChainStructure`, or :class:`Chain` objects from
    various input types.

    To initialize :class:`Chain` objects, use :meth:`from_mapping`.

    To initialize :class:`ChainSequence` or :class:`ChainStructure` objects,
    use :meth:`from_iterable`.

    """

    def __init__(
        self,
        num_proc: int | None = None,
        tolerate_failures: bool = False,
        verbose: bool = False,
    ):
        """

        :param num_proc: The number of processes to use.
        :param tolerate_failures: Don't stop the execution if some object fails
            to initialize.
        :param verbose: Output progress bars.
        """
        self.num_proc = num_proc
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
        return [".cif", ".pdb", ".pdbx", ".mmtf", ".npz"]

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
        callbacks: list[InitializerCallback] | None = None,
    ) -> abc.Generator[ChainSequence | ChainStructure]:
        """
        Initialize :class:`ChainSequence`s or/and :class:`ChainStructure`'s
        from (possibly heterogeneous) iterable.

        :param it:
            Supported elements are:
                1) Initialized objects (passed without any actions).
                2) Path to a sequence or a structure file.
                3) (Path to a structure file, list of target chains).
                4) A pair (header, seq) to initialize a :class:`ChainSequence`.
                5) A :class:`GenericStructure` with a single chain.

        :param callbacks: A sequence of callables accepting and returning an
            initialized object.
        :return: A generator yielding initialized chain sequences and
            structures parsed from the inputs.
        """
        if self.num_proc is not None:
            with ProcessPoolExecutor(self.num_proc) as executor:
                futures = [
                    (
                        x,
                        executor.submit(
                            _init,
                            x,
                            self.tolerate_failures,
                            self.supported_seq_ext,
                            self.supported_str_ext,
                            callbacks,
                        ),
                    )
                    for x in it
                ]
                if self.verbose:
                    futures = tqdm(futures, desc="Initializing objects in parallel")
                for x, future in futures:
                    try:
                        yield future.result()
                    except Exception as e:
                        LOGGER.warning(f"Input {x} failed with an error {e}")
                        # LOGGER.exception(e)
                        if not self.tolerate_failures:
                            raise e
                        yield None
        else:
            if self.verbose:
                it = tqdm(it, desc="Initializing objects sequentially")
            yield from (
                _init(
                    x,
                    self.tolerate_failures,
                    self.supported_seq_ext,
                    self.supported_str_ext,
                    callbacks,
                )
                for x in it
            )

    def from_mapping(
        self,
        m: abc.Mapping[
            ChainSequence | tuple[str, str] | Path,
            abc.Sequence[
                ChainStructure
                | GenericStructure
                | bst.AtomArray
                | Path
                | tuple[Path, abc.Sequence[str]]
            ],
        ],
        key_callbacks: t.Optional[list[InitializerCallback]] = None,
        val_callbacks: t.Optional[list[InitializerCallback]] = None,
        num_proc_map_numbering: int | None = None,
        **kwargs,
    ) -> list[Chain]:
        """
        Initialize :class:`Chain`'s from mapping between sequences and
        structures.

        It will first initialize objects to which the elements of `m`
        refer (see below) and then create maps between each sequence and
        associated structures, saving these into structure
        :attr:`ChainStructure.seq`'s.

        :param m:
            A mapping of the form ``{seq => [structures]}``, where `seq`
            is one of:

                1) Initialized :class:`ChainSequence`.
                2) A pair (header, seq).
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
        :param num_proc_map_numbering: A number of processes to use for mapping
            between numbering of sequences and structures. Generally, this
            should be as high as possible for faster processing. In contrast
            to the other operations here, this one seems more CPU-bound and
            less resource hungry (although, keep in mind the size of the
            canonical sequence: if it's too high, the RAM usage will likely
            explode). If ``None``, will default to :attr:`num_proc`.
        :param kwargs: Passed to :meth:`Chain.add_structure`.
        :return: A list of initialized chains.
        """
        # Process keys and values
        keys = self.from_iterable(m, callbacks=key_callbacks)  # ChainSequences
        values_flattened = self.from_iterable(  # ChainStructures
            chain.from_iterable(m.values()), callbacks=val_callbacks
        )
        values = split_into(values_flattened, map(len, m.values()))

        m_new = valmap(
            lambda vs: collapse(filter(bool, vs)),
            keymap(
                Chain,  # create `Chain` objects from `ChainSequence`s
                keyfilter(bool, dict(zip(keys, values))),  # Filter possible failures
            ),
        )

        num_proc = num_proc_map_numbering or self.num_proc

        if num_proc is None or num_proc == 1:
            for c, ss in m_new.items():
                for s in ss:
                    c.add_structure(s, **kwargs)
        else:
            map_name = kwargs.get("map_name") or SeqNames.map_canonical

            # explicitly unpack value iterable into lists
            m_new = valmap(list, m_new)

            # create numbering groups -- lists of lists with numberings
            # for each structure in values
            numbering_groups = map_numbering_many2many(
                [x.seq for x in m_new],
                [[x.seq for x in val] for val in m_new.values()],
                num_proc=num_proc,
                verbose=self.verbose,
            )
            for (c, ss), num_group in zip(m_new.items(), numbering_groups, strict=True):
                if len(num_group) != len(ss):
                    raise LengthMismatch(
                        f"The number of mapped numberings {len(num_group)} must match "
                        f"the number of structures {len(ss)}."
                    )
                for s, n in zip(ss, num_group):
                    try:
                        s.seq.add_seq(map_name, n)
                        c.add_structure(s, map_to_seq=False, **kwargs)
                    except Exception as e:
                        LOGGER.warning(
                            f"Failed to add structure {s} to chain {c} due to {e}"
                        )
                        LOGGER.exception(e)
                        if not self.tolerate_failures:
                            raise e

        return list(m_new)
