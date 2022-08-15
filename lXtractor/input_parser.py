import logging
import typing as t
from itertools import product, chain
from pathlib import Path

from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from more_itertools import flatten

from lXtractor.base import SeqRec, FormatError, MissingData, InputSeparators
from lXtractor.protein import Protein
from lXtractor.sifts import SIFTS

LOGGER = logging.getLogger(__name__)
ProteinAttributes = t.NamedTuple(
    'ProteinAttributes', [
        ('pdb_id', t.Optional[str]),
        ('chain_id', t.Optional[str]),
        ('structure', t.Optional[Structure]),
        ('uniprot_id', t.Optional[str]),
        ('uniprot_seq', t.Optional[SeqRec]),
        ('domains', t.Optional[t.Sequence[str]])
    ])
Sep = InputSeparators(',', ':', '::', '_')


def parse_pdb_input(inp: str) -> t.Tuple[str, t.Optional[Structure]]:

    if inp.endswith('.pdb'):
        LOGGER.debug(f'Path to a PDB file expected in {inp}')
        path = Path(inp)
        if not (path.exists() or path.is_file()):
            raise FormatError(f'Invalid path {inp}')
        pdb_code = path.stem
        structure = PDBParser(QUIET=True).get_structure(pdb_code.upper(), path)
    else:
        pdb_code, structure = inp, None

    pdb_code = pdb_code.upper()
    LOGGER.debug(f'Parsed {inp} into PDB={pdb_code},structure={structure}')
    return pdb_code.upper(), structure


def parse_uniprot_input(inp: str) -> t.Tuple[str, SeqRec]:
    # No special name parsing. We instead assume the provided ID
    # to be a valid UniProt accession

    if inp.endswith('.fasta'):
        LOGGER.debug(f'Expecting a path to a fasta file in {inp}')
        path = Path(inp)
        if not (path.exists() or path.is_file()):
            raise FormatError(f'Invalid path {inp}')
        seqs = list(SeqIO.parse(path, 'fasta'))
        if len(seqs) > 1:
            LOGGER.warning(
                f'Found {len(seqs)} sequences in {inp}. '
                f'Assuming the first one to be a target seq.')
        uniprot_id, seq = path.stem, seqs[0]
    else:
        uniprot_id, seq = inp, None
    LOGGER.debug(f'Parsed {inp} into UniProt_ID:{uniprot_id},'
                 f'Sequence={seq.id if seq is not None else None}')
    return uniprot_id, seq


def parse_protein(
        inp: str, sep: InputSeparators = Sep
) -> t.Tuple[t.Optional[str], t.Optional[SeqRec], t.Optional[str], t.Optional[Structure]]:
    if sep.uni_pdb in inp:
        uni, pdb = inp.split(sep.uni_pdb)
    else:
        if len(inp) == 4 or inp.endswith('.pdb'):
            uni, pdb = None, inp
        else:
            uni, pdb = inp, None
    uni_id, uni_seq = parse_uniprot_input(uni) if uni else (None, None)
    pdb_id, pdb_str = parse_pdb_input(pdb) if pdb else (None, None)
    return uni_id, uni_seq, pdb_id, pdb_str


def convert_to_attributes(inp: str, sep: InputSeparators = Sep) -> t.Iterator[ProteinAttributes]:

    def safe_split(_inp: str, _sep: str) -> t.Tuple[str, t.List[t.Optional[str]]]:
        if _sep in _inp:
            inp_split = _inp.split(_sep)
            if len(inp_split) > 2:
                raise FormatError(f'>1 {sep.dom} in {_inp} is not allowed')
            left, right = inp_split
            return left, right.split(sep.list)
        return _inp, [None]

    prot, domains = safe_split(inp, sep.dom)    # protein_inp:chain_inp, [optional domains]
    prot, chains = safe_split(prot, sep.chain)  # protein_inp, [optional chains]
    prot = (parse_protein(x, sep) for x in prot.split(sep.list))

    for p, c in product(prot, chains):
        uni_id, uni_seq, pdb_id, pdb_str = p
        yield ProteinAttributes(pdb_id, c, pdb_str, uni_id, uni_seq, domains)


def infer_and_explode_attributes(
        var: ProteinAttributes, sifts: SIFTS
) -> t.Iterator[ProteinAttributes]:
    """
    Try inferring missing data from SIFTS.

    :param var: a (possibly incomplete) set of attributes.
    :param sifts: initialized :class:`lXtractor.sifts.SIFTS` mapping.
    :return: iterator over variable sets
        (sets of :class:`lXtractor.protein.Protein` attributes),
        uniquely specifying a protein.
    """

    def get_chains(pdb_id: str) -> t.List[str]:
        assert len(pdb_id) == 4
        _chains = sifts.id_mapping[pdb_id]
        LOGGER.debug(f'Found {len(_chains)} chains for {pdb_id}: {_chains}')
        if not _chains:
            raise MissingData(
                f'Failed to extract chain IDs from SIFTS '
                f'for variable set {var}')
        return _chains

    if var.pdb_id is None and var.uniprot_id is None:
        raise MissingData(
            f'Expected to have either UniProt ID or PDB ID, but got neither '
            f'for variable set {var}')

    elif var.uniprot_id is None:

        chains = [var.chain_id]

        if var.chain_id is None:
            chains = get_chains(var.pdb_id)

        pdb_chains = [f'{var.pdb_id}:{c}' for c in chains]
        uniprot_ids = set(flatten(sifts.map_id(x) for x in pdb_chains))
        LOGGER.debug(f'Found {len(uniprot_ids)} UniProt IDs for PDB chains {pdb_chains}')

    elif var.pdb_id is None:
        uniprot_ids = [var.uniprot_id]
        pdb_chains = sifts.id_mapping[var.uniprot_id]

        if var.chain_id is not None:
            pdb_chains = sorted({f"{x.split(':')[0]}:{var.chain_id}" for x in pdb_chains})

        LOGGER.debug(f'Found {len(pdb_chains)} chains for {var.uniprot_id}: {pdb_chains}')

    else:
        uniprot_ids = [var.uniprot_id]
        chains = get_chains(var.pdb_id) if var.chain_id is None else [var.chain_id]
        pdb_chains = sorted({f"{var.pdb_id}:{с}" for с in chains})

    # else:
    #     chains = get_chains(var.pdb_id) if var.chain_id is None else [var.chain_id]
    #     pdb_chains = [f'{var.pdb_id}:{c}' for c in chains]
    #     uniprot_ids = [var.uniprot_id]

    # At this point we have:
    # (1) A list of PDB:Chains (or a single one)
    # (2) A list of UnIProt IDs (or a single one)
    # Each combination is guaranteed to uniquely specify a `Protein` instance
    for uni_id in uniprot_ids:
        for pdb_chain in pdb_chains:
            _pdb_id, _chain_id = pdb_chain.split(':')
            # TODO: ! I decided not to deepcopy structure/seq to save space. Sane?
            # We deepcopy these objects so that manipulations within `Protein`
            # do not change other `Protein` objects due to mutability or the
            # `Protein`'s attributes.
            # structure = deepcopy(var.structure)
            # uniprot_seq = deepcopy(var.uniprot_seq)
            yield ProteinAttributes(
                _pdb_id, _chain_id, var.structure, uni_id,
                var.uniprot_seq, var.domains)


def init(inp: str, sifts: t.Optional[SIFTS]) -> t.Iterator[Protein]:
    # Parse initial input into a (merged) attributes
    attributes = convert_to_attributes(inp)
    LOGGER.debug(f'Parsed {inp} into attributes')

    if sifts is not None:
        attributes = chain.from_iterable(
            infer_and_explode_attributes(a, sifts) for a in attributes)
        LOGGER.debug('Exploded attributes using SIFTS')

    # for each variable set -- initialize `Protein` object
    for att in attributes:
        _id = f'{att.uniprot_id}_{att.pdb_id}:{att.chain_id}'
        LOGGER.debug(f'Initializing protein {_id} by attributes {att}')
        yield Protein(
            _id=_id, dir_name=_id, pdb=att.pdb_id, chain=att.chain_id,
            uniprot_id=att.uniprot_id, uniprot_seq=att.uniprot_seq,
            expected_domains=att.domains, structure=att.structure,
            variables={})


if __name__ == '__main__':
    raise RuntimeError
