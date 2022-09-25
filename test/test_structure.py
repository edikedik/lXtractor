from lXtractor.core.structure import Structure


def test_init(simple_structure_path):
    s = Structure.read(simple_structure_path)
    assert len(s.array) > 0


def test_split(simple_structure):
    s = simple_structure
    chains = list(s.split_chains())
    assert len(chains) == 1
    assert len(chains[0].array) == len(s.array)
    assert isinstance(chains.pop(), Structure)


def test_sequence(simple_structure):
    seq = list(simple_structure.get_sequence())
    assert len(seq) == 207


def test_subsetting(simple_structure):
    sub = simple_structure.sub_structure(1, 2)
    seq = list(sub.get_sequence())
    assert len(seq) == 2


def test_write(simple_structure):
    # TODO: implement when providing paths is fixed in biotite
    pass
