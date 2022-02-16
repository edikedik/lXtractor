import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from Bio import SeqRecord, Seq
from Bio.PDB.Structure import Structure

SeqRec = SeqRecord.SeqRecord
Seq = Seq.Seq
T = t.TypeVar('T')
_Fetcher = t.Callable[[t.Iterable[str]], T]
_Getter = t.Callable[[T, t.Sequence[str]], t.Sequence[str]]
_StrSep = '--'
_DomSep = '::'


class AminoAcidDict:
    # TODO: consider morphing into a proper dict subclass
    """
    Complete and flexible amino acid dictionary, mapping between
    3->1 and 1->3-letter codes.
    """

    def __init__(
            self,
            aa1_unk: str = 'X',
            aa3_unk: str = 'UNK',
            any_unk: t.Optional[str] = None):
        """
        :param aa1_unk: unknown character when mapping 3->1
        :param aa3_unk: unknown character when mapping 1->3
        :param any_unk: unknown character when a key doesn't
            meet 1 or 3 length requirements.
        """
        self.aa1_unk = aa1_unk
        self.aa3_unk = aa3_unk
        self.any_unk = any_unk
        self._aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'THR': 'T', 'GLU': 'E',
            'GLH': 'e', 'ASP': 'D', 'ASH': 'd', 'PHE': 'F',
            'TRP': 'W', 'ILE': 'I', 'VAL': 'V', 'LEU': 'L',
            'LYS': 'K', 'LYN': 'k', 'MET': 'M', 'ASN': 'N',
            'GLN': 'Q', 'SER': 'S', 'ARG': 'R', 'TYR': 'Y',
            'TYD': 'y', 'HID': 'h', 'HIE': 'j', 'HIP': 'H',
            'HIS': 'H', 'PRO': 'P', 'GLY': 'G', 'A': 'ALA',
            'C': 'CYS', 'T': 'THR', 'E': 'GLU', 'e': 'GLH',
            'D': 'ASP', 'd': 'ASH', 'F': 'PHE', 'W': 'TRP',
            'I': 'ILE', 'V': 'VAL', 'L': 'LEU', 'K': 'LYS',
            'k': 'LYN', 'M': 'MET', 'N': 'ASN', 'Q': 'GLN',
            'S': 'SER', 'R': 'ARG', 'Y': 'TYR', 'y': 'TYD',
            'h': 'HID', 'j': 'HIE', 'H': 'HIS', 'P': 'PRO',
            'G': 'GLY'}

    @property
    def aa_dict(self) -> t.Dict[str, str]:
        return self._aa_dict

    @property
    def proto_mapping(self) -> t.Dict[str, str]:
        """
        :return: unprotonated version of an amino acid code
        """
        return {'e': 'E', 'd': 'D', 'k': 'K', 'y': 'Y', 'j': 'H', 'h': 'H',
                'GLH': 'GLU', 'ASH': 'ASP', 'LYN': 'LYS',
                'TYD': 'TYR', 'HID': 'HIP', 'HIE': 'HIP'}

    @property
    def three_letter_codes(self) -> t.List[str]:
        """
        :return: list of available 3-letter codes
        """
        return list(filter(lambda x: len(x) == 3, self._aa_dict))

    @property
    def one_letter_codes(self) -> t.List[str]:
        """
        :return: list of available 1-letter codes
        """
        return list(filter(lambda x: len(x) == 1, self._aa_dict))

    def __getitem__(self, item):
        if item in self._aa_dict:
            return self._aa_dict[item]
        if len(item) == 3:
            return self.aa1_unk
        elif len(item) == 1:
            return self.aa3_unk
        else:
            if self.any_unk is not None:
                return self.any_unk
            raise KeyError(
                f'Expected 3-sized or 1-sized item, '
                f'got {len(item)}-sized {item}')

    def __contains__(self, item):
        return item in self.aa_dict


class AbstractResource(metaclass=ABCMeta):
    """
    Abstract base class defyning basic interface any resource must provide.
    """

    def __init__(self, resource_path: t.Optional[Path],
                 resource_name: t.Optional[str]):
        self.name = resource_name
        self.path = resource_path

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        raise NotImplementedError

    @abstractmethod
    def dump(self, path: Path):
        raise NotImplementedError


class AbstractVariable(metaclass=ABCMeta):
    """
    Abstract base class for variables.

    A variable is any quantity that can be calculated given a
    :class:`Bio.PDB.Structure.Structure` object and a mapping
    between alignment numbering and structure's numbering.

    During ``__init__`` each a variable encapsulates alignment-related data,
    so the mapping is necessary for the calculation of any variable.
    """

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.__str__()

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Variable identifier. Must be a unique string value.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate(
            self, structure: Structure,
            mapping: t.Optional[t.Mapping[int, int]] = None):
        """
        Calculate the variable. Each variable defines its own calculation
        strategy within this method.

        :param structure:
        :param mapping:
        :return:
        """
        raise NotImplementedError


# class Variables(UserDict):
#     """
#     A custom dictionary subclass to encapsulate variables.
#     Redefines setting items based on a given :attr:`AbstractVariable.id`.
#     """
#
#     def __setitem__(self, key: str, value: t):
#         if key.id not in self.ids:
#             self.ids[key.id] = value
#             super().__setitem__(key, value)
#         elif self.ids[key.id] is None:
#             self.ids[key.id] = value
#             super().__delitem__(key)
#             super().__setitem__(key, value)


Variables = t.Dict[str, t.Tuple[AbstractVariable, t.Union[str, float, None]]]


@dataclass
class VariableResult:
    Level: str
    Result: t.Any


@dataclass
class _ProteinDumpNames:
    """
    Dataclass encapsulating names of files
    used for dumping data.
    """
    uniprot_seq: str = 'sequence.fasta'
    pdb_seq: str = 'structure.fasta'
    pdb_seq_raw: str = 'structure.txt'
    pdb_structure: str = 'structure.pdb'
    pdb_meta: str = 'meta.tsv'
    variables: str = 'variables.tsv'
    aln_mapping: str = 'aln_mapping.tsv'
    uni_pdb_map: str = 'uni_pdb_map.tsv'
    distance_map_base: str = 'DM'
    uni_pdb_aln: str = 'uni_pdb_aln.fasta'


ProteinDumpNames = _ProteinDumpNames()


@dataclass
class Segment:
    """
    Dataclass holding a definition of an arbitrary segment.

    Minimum data required is ``start`` and ``and``.
    It can be helpful to define both
     name of the segment and
    where it comes from (via ``parent_name``).

    A dictionary with arbitrary data resides within ``data``
    attribute.
    """
    start: int
    end: int
    name: t.Optional[str] = None
    parent_name: t.Optional[str] = None
    data: t.Optional[t.Dict[str, t.Any]] = field(
        default_factory=dict)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        parent = f'(<-{self.parent_name})' if self.parent_name else ''
        return f'{self.name}{parent}:{self.start}-{self.end}'

    def __iter__(self):
        return iter(range(self.start, self.end + 1))

    def __len__(self):
        if self.end is None or self.end is None or self.start >= self.end:
            return 0
        return self.end - self.start + 1

    def do_overlap(self, other: 'Segment') -> bool:
        """
        Check whether a segment overlaps with the other segment.
        Use :meth:`overlap` to find an acual overlap.

        :param other: other :class:`Segment` instance.
        :return: ``True`` if segments overlap and ``False`` otherwise.
        """
        return not (other.start > self.end or self.start > other.end)

    def overlap(self, other: 'Segment') -> t.Optional['Segment']:
        """
        If segments overlap, create a "child" segment with
        overlapping boundaries, ``name`` and a ``parent_name``
        of a given ``other``.

        :param other: other :class:`Segment` instance.
        :return: new :class:`Segment` instance.
        """

        if not self.do_overlap(other):
            return None

        return Segment(
            max(self.start, other.start),
            min(self.end, other.end),
            name=other.name,
            parent_name=other.parent_name,
            data=other.data)


@dataclass
class Domain(Segment):
    uniprot_seq: t.Optional[SeqRec] = None
    pdb_seq: t.Optional[SeqRec] = None
    pdb_seq_raw: t.Optional[t.Tuple[str, ...]] = None
    pdb_sub_structure: t.Optional[Structure] = None
    uni_pdb_map: t.Optional[t.Dict[int, t.Optional[int]]] = None
    uni_pdb_aln: t.Optional[t.Tuple[SeqRec, SeqRec]] = None
    aln_mapping: t.Optional[t.Dict[int, int]] = None
    pdb_segment_boundaries: t.Optional[t.Tuple[int, int]] = None
    metadata: t.Optional[t.List[t.Tuple[str, t.Any]]] = field(default_factory=list)
    dir_name: t.Optional[Path] = None
    variables: t.Optional[Variables] = field(default_factory=dict)

    @property
    def id(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        parent = f'(<-{self.parent_name})' if self.parent_name else ''
        return f'{self.name}{parent}:{self.start}-{self.end}'


Domains = t.Dict[str, Domain]


# DunbrackSegments: t.Tuple[Segment, ...] = (
#     Segment(1, 4),  # B1N
#     Segment(17, 23),  # B1C
#     Segment(37, 45),  # B2
#     Segment(82, 92),  # B3
#     Segment(125, 138),  # HC
#     Segment(161, 176),  # B4
#     Segment(398, 407),  # B5
#     Segment(419, 427),  # HD
#     Segment(909, 932),  # HE
#     Segment(974, 980),  # CL1
#     Segment(986, 994),  # CL2
#     Segment(1293, 1313),  # ALN
#     Segment(1862, 1878),  # ALC
#     Segment(1907, 1929),  # HF
#     Segment(1944, 1949),  # FL
#     Segment(1996, 2008),  # HG
#     Segment(2118, 2137),  # HH
#     Segment(2148, 2157)  # HI
# )


class InitError(ValueError):
    """
    A broad category exception for problems with
    an object's initialization
    """
    pass


class MissingData(ValueError):
    pass


class AmbiguousData(ValueError):
    pass


class AmbiguousMapping(ValueError):
    pass


class NoOverlap(ValueError):
    pass


class FormatError(ValueError):
    pass


class FailedCalculation(RuntimeError):
    pass


class LengthMismatch(ValueError):
    pass


class OverlapError(ValueError):
    pass


class ParsingError(ValueError):
    pass


MetaColumns = (
    # Taken from https://bioservices.readthedocs.io/en/master/_modules/bioservices/uniprot.html

    # Names & Taxonomy
    'id', 'entry name', 'genes', 'genes(PREFERRED)', 'genes(ALTERNATIVE)',
    'genes(OLN)', 'genes(ORF)', 'organism', 'organism-id', 'protein names',
    'proteome', 'lineage(ALL)', 'lineage-id', 'virus hosts',
    # Sequences
    'fragment', 'sequence', 'length', 'mass', 'encodedon',
    'comment(ALTERNATIVE PRODUCTS)', 'comment(ERRONEOUS GENE MODEL PREDICTION)',
    'comment(ERRONEOUS INITIATION)', 'comment(ERRONEOUS TERMINATION)',
    'comment(ERRONEOUS TRANSLATION)', 'comment(FRAMESHIFT)',
    'comment(MASS SPECTROMETRY)', 'comment(POLYMORPHISM)',
    'comment(RNA EDITING)', 'comment(SEQUENCE CAUTION)',
    'feature(ALTERNATIVE SEQUENCE)', 'feature(NATURAL VARIANT)',
    'feature(NON ADJACENT RESIDUES)',
    'feature(NON STANDARD RESIDUE)', 'feature(NON TERMINAL RESIDUE)',
    'feature(SEQUENCE CONFLICT)', 'feature(SEQUENCE UNCERTAINTY)',
    'version(sequence)',
    # Family and Domains
    'domains', 'domain', 'comment(DOMAIN)', 'comment(SIMILARITY)',
    'feature(COILED COIL)', 'feature(COMPOSITIONAL BIAS)',
    'feature(DOMAIN EXTENT)', 'feature(MOTIF)', 'feature(REGION)',
    'feature(REPEAT)', 'feature(ZINC FINGER)',
    # Function
    'ec', 'comment(ABSORPTION)', 'comment(CATALYTIC ACTIVITY)',
    'comment(COFACTOR)', 'comment(ENZYME REGULATION)', 'comment(FUNCTION)',
    'comment(KINETICS)', 'comment(PATHWAY)', 'comment(REDOX POTENTIAL)',
    'comment(TEMPERATURE DEPENDENCE)', 'comment(PH DEPENDENCE)',
    'feature(ACTIVE SITE)', 'feature(BINDING SITE)', 'feature(DNA BINDING)',
    'feature(METAL BINDING)', 'feature(NP BIND)', 'feature(SITE)',
    # Gene Ontologys
    'go', 'go(biological process)', 'go(molecular function)',
    'go(cellular component)', 'go-id',
    # InterPro
    'interpro',
    # Interaction
    'interactor', 'comment(SUBUNIT)',
    # Publications
    'citation', 'citationmapping',
    # Date of
    'created', 'last-modified', 'sequence-modified', 'version(entry)',
    # Structure
    '3d', 'feature(BETA STRAND)', 'feature(HELIX)', 'feature(TURN)',
    # Subcellular location
    'comment(SUBCELLULAR LOCATION)',
    'feature(INTRAMEMBRANE)',
    'feature(TOPOLOGICAL DOMAIN)',
    'feature(TRANSMEMBRANE)',
    # Miscellaneous
    'annotation score', 'score', 'features', 'comment(CAUTION)',
    'comment(TISSUE SPECIFICITY)',
    'comment(GENERAL)', 'keywords', 'context', 'existence', 'tools',
    'reviewed', 'feature', 'families', 'subcellular locations', 'taxonomy',
    'version', 'clusters', 'comments', 'database', 'keyword-id', 'pathway',
    'score',
    # Pathology & Biotech
    'comment(ALLERGEN)', 'comment(BIOTECHNOLOGY)', 'comment(DISRUPTION PHENOTYPE)',
    'comment(DISEASE)', 'comment(PHARMACEUTICAL)', 'comment(TOXIC DOSE)',
    # PTM / Processsing
    'comment(PTM)', 'feature(CHAIN)', 'feature(CROSS LINK)', 'feature(DISULFIDE BOND)',
    'feature(GLYCOSYLATION)', 'feature(INITIATOR METHIONINE)', 'feature(LIPIDATION)',
    'feature(MODIFIED RESIDUE)', 'feature(PEPTIDE)', 'feature(PROPEPTIDE)',
    'feature(SIGNAL)', 'feature(TRANSIT)',
    # Taxonomic lineage
    'lineage(all)', 'lineage(SUPERKINGDOM)', 'lineage(KINGDOM)', 'lineage(SUBKINGDOM)',
    'lineage(SUPERPHYLUM)', 'lineage(PHYLUM)', 'lineage(SUBPHYLUM)', 'lineage(SUPERCLASS)',
    'lineage(CLASS)', 'lineage(SUBCLASS)', 'lineage(INFRACLASS)', 'lineage(SUPERORDER)',
    'lineage(ORDER)', 'lineage(SUBORDER)', 'lineage(INFRAORDER)', 'lineage(PARVORDER)',
    'lineage(SUPERFAMILY)', 'lineage(FAMILY)', 'lineage(SUBFAMILY)', 'lineage(TRIBE)',
    'lineage(SUBTRIBE)', 'lineage(GENUS)', 'lineage(SUBGENUS)', 'lineage(SPECIES GROUP)',
    'lineage(SPECIES SUBGROUP)', 'lineage(SPECIES)', 'lineage(SUBSPECIES)', 'lineage(VARIETAS)',
    'lineage(FORMA)',
    # Taxonomic identifier
    'lineage-id(all)', 'lineage-id(SUPERKINGDOM)', 'lineage-id(KINGDOM)', 'lineage-id(SUBKINGDOM)',
    'lineage-id(SUPERPHYLUM)', 'lineage-id(PHYLUM)', 'lineage-id(SUBPHYLUM)', 'lineage-id(SUPERCLASS)',
    'lineage-id(CLASS)', 'lineage-id(SUBCLASS)', 'lineage-id(INFRACLASS)', 'lineage-id(SUPERORDER)',
    'lineage-id(ORDER)', 'lineage-id(SUBORDER)', 'lineage-id(INFRAORDER)', 'lineage-id(PARVORDER)',
    'lineage-id(SUPERFAMILY)', 'lineage-id(FAMILY)', 'lineage-id(SUBFAMILY)', 'lineage-id(TRIBE)',
    'lineage-id(SUBTRIBE)', 'lineage-id(GENUS)', 'lineage-id(SUBGENUS)', 'lineage-id(SPECIES GROUP)',
    'lineage-id(SPECIES SUBGROUP)', 'lineage-id(SPECIES)', 'lineage-id(SUBSPECIES)', 'lineage-id(VARIETAS)',
    'lineage-id(FORMA)',
    # Cross-references
    'database(db_abbrev)', 'database(EMBL)')

if __name__ == '__main__':
    raise RuntimeError
