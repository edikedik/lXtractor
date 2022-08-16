"""
Module comprises utils for fetching PDB entries and manipulating the fetched structures.
"""
import logging
import typing as t
from io import StringIO
from itertools import chain
from pathlib import Path

from Bio.PDB import PDBParser, Select, PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from more_itertools import flatten, unzip, partition, mark_ends, pairwise, rstrip
from toolz import groupby, curry, pipe, identity
from toolz.curried import map, filter

from lXtractor.base import AminoAcidDict, MissingData, AmbiguousData, SeqRec, Seq
from lXtractor.protein import Protein
from lXtractor.utils import download_text, fetch_iterable, try_fetching_until

WrappedResult = t.Tuple[
    str, Structure, t.Optional[t.List[t.Tuple[str, t.Any]]]]
LOGGER = logging.getLogger(__name__)
META_FIELDS = (
    'idcode',
    'name',
    'resolution',
    'structure_method',
    'deposition_date',
)


class PDB:
    """
    An interface to fetch PDB structures (in PDB format) and process the results.
    """

    def __init__(
            self, max_retries: int = 10,
            num_threads: t.Optional[int] = None,
            meta_fields: t.Optional[t.Tuple[str, ...]] = META_FIELDS,
            expected_method: t.Optional[str] = 'x-ray diffraction',
            min_resolution: t.Optional[int] = None,
            pdb_dir: t.Optional[Path] = None,
            verbose: bool = False,
    ):
        """
        :param max_retries: a maximum number of fetching attempts.
        :param num_threads: a number of threads for the ``ThreadPoolExecutor``.
        :param meta_fields: a tuple of metadata names (potentially) returned by
            :func:`Bio.PDB.parse_pdb_header`. :meth:`PDB.fetch` will include
            these fields into :attr:`lXtractor.protein.Protein.metadata`.
        :param expected_method: filter to structures with "structure_method"
            annotated by a given value.
        :param min_resolution: filter to structures having "resolution" lower or
            equal than a given value.
        :param verbose: ...
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
            :attr:`lXtractor.protein.Protein._structure` (and, optionally,
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
def wrap_raw_pdb(
        res: str,
        meta_fields: t.Optional[t.Tuple[str]] = META_FIELDS
) -> WrappedResult:
    """
    :param res: string with raw PDB text file
    :param meta_fields: expected metadata fields to parse from header
    :return: PDB.Structure object and basic metadata
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
        num_threads: t.Optional[int] = None,
        meta_fields: t.Optional[t.Tuple[str]] = META_FIELDS,
        verbose: bool = False
) -> t.List[WrappedResult]:
    """
    A helper function to download PDB files.

    :param ids: an iterable over PDB IDs.
    :param num_threads: a number of threads for the ``ThreadPoolExecutor``.
    :param meta_fields:
    :param verbose: expose progress bar.
    :return:
    """
    url = 'https://files.rcsb.org/download/'

    def fetch_chunk(chunk: t.Iterable[str]) -> str:
        # `chunk` is an iterable over PDB IDs for compatibility reasons;
        chunk = list(chunk)
        if len(chunk) != 1:
            raise ValueError(f'Chunk {chunk} contains more than 1 element')
        return download_text(f'{url}{chunk.pop()}.pdb')

    results = fetch_iterable(ids, fetch_chunk, chunk_size=1, num_threads=num_threads, verbose=verbose)

    return list(map(wrap_raw_pdb(meta_fields=meta_fields), results))


def read_pdb(
        ids: t.Collection[str], pdb_dir: Path
) -> t.Tuple[t.List[WrappedResult], t.List[str]]:
    ids = list(ids)
    paths = [p for p in pdb_dir.iterdir() if p.stem in ids]
    if not paths:
        LOGGER.debug(f'No structures among {ids} are in {pdb_dir}')
        return [], []
    LOGGER.debug(f'Will read {len(paths)} from {pdb_dir}')
    results = [parse_structure(p) for p in paths]
    outputs, failures = partition(
        lambda x: isinstance(x[0], Exception), results)
    for stem, e in failures:
        LOGGER.warning(f'Failed to read {stem} due to {e}')
    results, ids = map(list, unzip(outputs))
    return results, ids


def parse_structure(
        path: Path
) -> t.Union[t.Tuple[WrappedResult, str], t.Tuple[Exception, str]]:
    try:
        res = wrap_raw_pdb(path.read_text())
        return res, path.stem
    except Exception as e:
        return e, path.stem


def extract_sub_structure(
        structure: Structure,
        chain_id: t.Optional[str],
        res_start: t.Optional[int],
        res_end: t.Optional[int]) -> Structure:
    """
    Extract a specific region of a protein chain within structure.

    :param structure: biopython's ``Structure`` object.
    :param chain_id: a PDB chain identifier.
    :param res_start: a start of the desired segment.
    :param res_end: an end of the desired segment.
    :return: biopython's ``Structure`` object containing
        sub-structure resulting from the selection.
    """
    # the shortest way to subset structures is by making io
    # write a selection into a handle, and then use this handle
    # to setup a new `Structure` object
    selector = Selector(chain_id, res_start, res_end)
    handle = StringIO()
    io = PDBIO()
    io.set_structure(structure)
    io.save(handle, selector)
    parser = PDBParser(QUIET=True)
    handle.seek(0)
    new_id = f'{structure.id}:{chain_id}_{res_start}-{res_end}'
    sub_structure = parser.get_structure(new_id, handle)
    LOGGER.debug(f'selected new sub-structure {new_id} from {structure.id}')
    return sub_structure


def split_chains(structure: Structure) -> t.Iterator[t.Tuple[str, Structure]]:
    """
    Split ``Structure`` based on chain IDs.

    :param structure: biopython's ``Structure`` object.
    :return: an iterator over tuples ("chain_id", "sub-structure").
    """
    chain_ids = [c.id for c in structure.get_chains()]
    LOGGER.debug(f'Found {len(chain_ids)} chains in {structure.id}')
    for c in chain_ids:
        yield c, extract_sub_structure(structure, c, None, None)


def get_sequence(
        structure: Structure,
        convert: bool = True,
        filter_out: t.Collection[str] = ('HOH',),
        trim_het_tail: bool = True,
        numbering: bool = False,
) -> t.Union[str, t.Tuple[str, ...], t.Tuple[int, ...]]:
    """
    Extract structure's residues.

    Optionally convert the sequence into one-letter codes.
    Any unknown residue names are marked as "X".
    (Known residue codes are specified within :class:`lXtractor.base.AminoAcidDict`).

    :param structure: biopython's ``Structure`` object.
    :param convert: convert 3-letter codes into 1-letter codes.
    :param filter_out: a collection of 3-letter codes to filter out.
    :param trim_het_tail: cut discontinuous hetatoms ending a chain.
    :param get_numbering:
    :return: a one-letter code sequence as a string.
    """
    mapping = AminoAcidDict(any_unk='X')

    def trim_tail(residues: t.Iterable[Residue]):
        def pred(
                pair: t.Tuple[Residue, Residue]
        ) -> bool:
            r1, r2 = pair
            i1, i2 = r1.id[1], r2.id[2]
            return i1 + 1 != i2 and (len(r2.resname) == 1 or r2.resname not in mapping)

        pairs = pairwise(residues)
        pairs_stripped = rstrip(pairs, pred)
        pairs_marked = mark_ends(pairs_stripped)
        residues = chain.from_iterable(
            # if is the last element get both residues else get only the first one
            (x[2][0], x[2][1]) if x[1] else (x[2][0],)
            for x in pairs_marked)
        return residues

    return pipe(
        structure.get_residues(),
        filter(lambda r: r.get_resname() not in filter_out),
        trim_tail if trim_het_tail else identity,
        lambda residues: tuple(
            (r.get_id()[1] if numbering else r.get_resname())
            for r in residues),
        (lambda resnames: "".join(mapping[name] for name in resnames))
        if convert and not numbering else identity
    )


class Selector(Select):
    """
    Biopython's way of sub-setting structures.
    `None` attributes are ommitted during the selection.
    """

    def __init__(
            self, chain_id: t.Optional[str],
            res_start: t.Optional[int],
            res_end: t.Optional[int]):
        self.chain_id = chain_id
        self.res_start = res_start
        self.res_end = res_end

    def accept_residue(self, residue):
        full_id = residue.full_id
        match_chain = self.chain_id is None or full_id[2] == self.chain_id
        match_lower = self.res_start is None or full_id[3][1] >= self.res_start
        match_upper = self.res_end is None or full_id[3][1] <= self.res_end
        return all([match_chain, match_lower, match_upper])

    def accept_chain(self, _chain):
        return self.chain_id is None or _chain.id == self.chain_id


if __name__ == '__main__':
    raise RuntimeError
