"""
Module comprises utils for fetching, splitting, extracting sub-structures
and sub-sequences of/from PDB files.
"""
import logging
import typing as t
from io import StringIO
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from more_itertools import flatten, unzip, partition
from toolz import groupby, curry
from toolz.curried import map, filter

from lXtractor.core.base import MissingData, AmbiguousData, SeqRec, Seq
from lXtractor.core.protein import Protein
from lXtractor.util.io import try_fetching_until, download_text, fetch_iterable
from lXtractor.util.structure import extract_sub_structure, get_sequence

WrappedResult = t.Tuple[str, Structure, t.Optional[t.List[t.Tuple[str, t.Any]]]]
META_FIELDS = (
    'idcode',
    'name',
    'resolution',
    'structure_method',
    'deposition_date',
)
LOGGER = logging.getLogger(__name__)


def _parse_structure(
        path: Path, **kwargs
) -> t.Union[t.Tuple[WrappedResult, str], t.Tuple[Exception, str]]:
    try:
        res = _wrap_raw_pdb(path.read_text(), **kwargs)
        return res, path.stem
    except Exception as e:
        LOGGER.exception(f'Failed to read {path} due to {e}')
        return e, path.stem


class PDB:
    """
    An interface to fetch PDB structures (in PDB format) and process the results.
    """

    def __init__(
            self, max_retries: int = 3,
            num_threads: t.Optional[int] = None,
            meta_fields: t.Optional[t.Tuple[str, ...]] = META_FIELDS,
            expected_method: t.Optional[str] = 'x-ray diffraction',
            min_resolution: t.Optional[int] = None,
            pdb_dir: t.Optional[Path] = None,
            verbose: bool = False,
    ):
        """
        :param max_retries: a maximum number of fetching attempts.
        :param num_threads: a number of threads for the :class:`ThreadPoolExecutor`.
        :param meta_fields: a tuple of metadata names (potentially) returned by
            :func:`Bio.PDB.parse_pdb_header`. :meth:`PDB.fetch` will include
            these fields into :attr:`lXtractor.protein.Protein.metadata`.
        :param expected_method: filter to structures with "structure_method"
            annotated by a given value.
        :param min_resolution: filter to structures having "resolution" lower or
            equal than a given value.
        :param verbose: progress bar on/off.
        """
        self.max_retries = max_retries
        self.num_threads = num_threads
        self.meta_fields = tuple(meta_fields)
        self.expected_method = expected_method
        self.min_resolution = min_resolution
        if pdb_dir is not None:
            pdb_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_dir = pdb_dir
        self.verbose = verbose

    @staticmethod
    def _check_input(proteins: t.Iterable[Protein]):
        """
        Check whether each ``Protein`` has a valid PDB code.
        """
        for p in proteins:
            if p is None:
                raise MissingData(f'Missing PDB ID for {p}')
            if len(p.pdb) != 4:
                raise AmbiguousData(f'Not a valid PDB ID {p.pdb} for {p}')

    def fetch(
            self, proteins: t.Collection[Protein],
    ) -> t.List[Protein]:
        """
        Obtains a list of unique PDB IDs from a given collection of
        :class:`lXtractor.protein.Protein` objects, and downloads
        the corresponding PDB structures.
        Splits fetched (full) structures by chains and distributes them
        between :class:`lXtractor.protein.Protein` objects.

        Additionally, parses :attr:`PDB.meta_fields` from a PDB header
        and saves the results.

        :param proteins: a collection of :class:`lXtractor.protein.Protein` objects.
        :return: a list of :class:`lXtractor.protein.Protein` objects with populated
            :attr:`lXtractor.protein.Protein.structure` (and, optionally,
            :attr:`lXtractor.protein.Protein.metadata`) attributes
        """

        def fetcher(ids: t.Iterable[str]):
            ids = list(ids)
            res = []
            if self.pdb_dir:
                res_read, ids_read = read_pdb(ids, self.pdb_dir)
                ids = [x for x in ids if x not in ids_read]
                res += res_read
                LOGGER.debug(
                    f'Read {len(res_read)} structures from {self.pdb_dir}. '
                    f'Remaining IDs: {len(ids)}')
            if ids:
                res_fetched = fetch_pdb(
                    ids, num_threads=self.num_threads,
                    verbose=self.verbose)
                res += res_fetched
                LOGGER.debug(f'Fetched {len(res_fetched)} structures')

            return res

        def get_remaining(
                fetched: t.Sequence[WrappedResult],
                _remaining: t.Sequence[str]
        ) -> t.List[str]:
            _current = set(_structure.id for _, _structure, _ in fetched)
            _remaining = list(set(_remaining) - _current)
            LOGGER.debug(f'PDB fetcher has {len(_remaining)} IDs to fetch')
            return _remaining

        @curry
        def accept_result(
                result: WrappedResult,
                resolution_field: str, method_field: str
        ) -> bool:
            _, _structure, _meta = result
            if not _structure.id:
                LOGGER.warning(
                    f'Filtering out a result with no parsed structure ID. '
                    f'Meta fields: {_meta}')
                return False
            if _meta is None or (
                    self.min_resolution is None and
                    self.expected_method is None):
                LOGGER.debug(
                    f'No metadata or filtering criteria -> '
                    f'accepting result {result}')
                return True
            _meta = dict(_meta)
            if self.min_resolution is not None:
                resolution = _meta.get(resolution_field)
                if resolution is None:
                    LOGGER.warning(
                        f'Could not find "{resolution_field}" '
                        f'in the metadata of result {result}. '
                        f'Will reject this result.')
                    accept_resolution = False
                else:
                    if not isinstance(resolution, float):
                        LOGGER.warning(
                            f'Expected to find a number in '
                            f'{resolution_field} of {result}')
                        accept_resolution = False
                    else:
                        accept_resolution = resolution <= self.min_resolution
            else:
                accept_resolution = True
            if self.expected_method is not None:
                method = _meta.get(method_field)
                if method is None:
                    LOGGER.warning(
                        f'Could not find "{method_field}" '
                        f'in the metadata of result {result}. '
                        f'Will reject this result.')
                    accept_method = True
                else:
                    accept_method = method == self.expected_method
            else:
                accept_method = True

            accept = accept_resolution and accept_method
            if accept:
                LOGGER.debug(
                    f'Accepting {_structure.id}: '
                    f'{accept_resolution} & {accept_method} = {accept}')
            else:
                LOGGER.info(
                    f'Filtering out {_structure.id} '
                    f'(accept_resolution={accept_resolution},'
                    f'accept_method={accept_method})')
            return accept

        # Download unique IDs
        _ids = set(p.pdb for p in proteins)
        LOGGER.debug(f'Found {len(_ids)} unique PDB IDs to fetch: {_ids}')

        results, remaining = try_fetching_until(
            _ids,
            fetcher=fetcher,
            get_remaining=get_remaining,
            max_trials=self.max_retries,
            desc='Fetching PDBs')
        if remaining:
            LOGGER.warning(f'Failed to fetch {remaining}')
        results = list(flatten(results))

        # Save raw structures if ``pdb_dir`` was provided
        if self.pdb_dir is not None:
            for raw, structure, meta in results:
                path = self.pdb_dir / f"{structure.id}.pdb"
                if not path.exists():
                    with path.open('w') as f:
                        print(raw, file=f, end='')
                    LOGGER.debug(f'Saved new PDB to {path}')

        # flatten and filter the results
        results = filter(
            accept_result(
                resolution_field='resolution',
                method_field='structure_method'),
            results)
        # LOGGER.debug(f'Obtained {len(results)} filtered results')

        results = {
            structure.id: (structure, meta)
            for _, structure, meta in results}

        # prepare proteins mapping (pdb_id, chain_id) -> protein
        # it's unlikely but possible that a PDB chain maps to >1
        # UniProt IDs
        _proteins = groupby(lambda p: (p.pdb, p.chain), proteins)

        for (pdb_id, chain_id), group in _proteins.items():
            if pdb_id not in results:
                LOGGER.info(
                    f'No fetched results for {pdb_id}')
                continue
            structure, meta = results[pdb_id]
            sub_structure = extract_sub_structure(
                structure, chain_id, None, None)
            seq_3_letter = get_sequence(sub_structure, convert=False)
            seq_1_letter = get_sequence(sub_structure, convert=True)
            for p in group:
                p.metadata += meta
                p.structure = sub_structure
                p.pdb_seq = SeqRec(
                    Seq(seq_1_letter),
                    id=sub_structure.id,
                    description=sub_structure.id,
                    name=sub_structure.id)
                p.pdb_seq_raw = seq_3_letter

        return list(flatten(_proteins.values()))


@curry
def _wrap_raw_pdb(
        res: str,
        meta_fields: t.Optional[t.Tuple[str]] = META_FIELDS
) -> WrappedResult:
    """
    :param res: string with raw PDB text file.
    :param meta_fields: expected metadata fields to parse from header.
    :return: PDB.Structure object and basic metadata.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', StringIO(res))
    header = structure.header
    if header and 'idcode' in header:
        structure.id = header['idcode'].upper()

    # Extract metadata fields from a `header` dictionary
    meta = [(x, header.get(x)) for x in meta_fields] if meta_fields else None

    return res, structure, meta


def fetch_pdb(
        ids: t.Iterable[str],
        meta_fields: t.Optional[t.Tuple[str]] = META_FIELDS,
        **kwargs
) -> t.List[WrappedResult]:
    """
    A helper function to download PDB files.
    Additional kwargs are passed to :func:`fetch_iterable`.

    :param ids: an iterable over PDB IDs.
    :param meta_fields: a collection of metadata fields to parse from the fetched file.
    :return: a list with fetched structures and corresponding metadata.
    """
    url = 'https://files.rcsb.org/download/'

    def fetch_chunk(chunk: t.Iterable[str]) -> str:
        # `chunk` is an iterable over PDB IDs for compatibility reasons;
        chunk = list(chunk)
        if len(chunk) != 1:
            raise ValueError(f'Chunk {chunk} contains more than 1 element')
        return download_text(f'{url}{chunk.pop()}.pdb')

    results = fetch_iterable(ids, fetch_chunk, **kwargs)

    return list(map(_wrap_raw_pdb(meta_fields=meta_fields), results))


def read_pdb(
        ids: t.Collection[str], pdb_dir: Path, **kwargs,
) -> t.Tuple[t.List[WrappedResult], t.List[str]]:
    """
    "Soft-read" structures with given ids from a directory.

    :param ids: A collection of PDB codes.
    :param pdb_dir: A directory where proteins specified by ``ids``
        (supposedly) reside.
    :return: Two lists: (1) successfully parsed structures -- tuples of the form
        (raw text, structure, metadata fields), and (2) extracted PDB IDs,
        for each structure in (1).
    """
    ids = list(ids)
    paths = [p for p in pdb_dir.iterdir() if p.stem in ids]
    if not paths:
        LOGGER.debug(f'No structures among {ids} are in {pdb_dir}')
        return [], []
    LOGGER.debug(f'Will read {len(paths)} from {pdb_dir}')
    results = [_parse_structure(p, **kwargs) for p in paths]
    outputs, _ = partition(lambda x: isinstance(x[0], Exception), results)
    results, ids = map(list, unzip(outputs))
    return results, ids


if __name__ == '__main__':
    raise RuntimeError
