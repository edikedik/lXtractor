import pytest

from lXtractor.chain import Chain, ChainStructure, ChainSequence
from lXtractor.core.config import DefaultConfig
from lXtractor.core.exceptions import NoOverlap, InitError
from lXtractor.util import biotite_align
from test.common import sample_chain, mark_meta


def test_basic(four_chain_structure_seq_path, four_chain_structure):
    p = Chain.from_seq(four_chain_structure_seq_path)[0]
    assert len(p.seq) == 1657
    assert p.id == f"Chain({p.seq.id})"
    chains = list(four_chain_structure.split_chains())
    chain_a = ChainStructure(chains[0])
    p.add_structure(chain_a)
    assert len(p.structures) == 1
    start, end = chain_a.seq.start, chain_a.seq.end
    sep = DefaultConfig["separators"]
    assert f"3i6x{sep['chain']}A{sep['start_end']}{start}-{end}"

    with pytest.raises(ValueError):
        p.add_structure(chain_a)

    p.add_structure(chain_a, check_ids=False)
    assert len(p.structures) == 2


def test_add_structure(chicken_src_seq, src_str):
    c = Chain.from_seq(chicken_src_seq)
    c1 = c.spawn_child(c.seq.start + 1, c.seq.end - 1, "c1")
    c2 = c1.spawn_child(c1.seq.start + 1, c1.seq.end - 1, "c2")
    assert len(c.seq) == len(c1.seq) + 2 == len(c2.seq) + 4
    assert len(c.structures) == 0
    c.add_structure(src_str)
    assert len(c.structures) == 1
    c.structures.pop()
    c.add_structure(src_str, add_to_children=True, align_method=biotite_align)
    children = list(c.iter_children())
    assert len(children) == 2
    c1, c2 = children[0][0], children[1][0]
    assert len(c1.structures) == len(c2.structures) == 1


def test_spawn(chicken_src_seq, human_src_seq, chicken_src_str):
    mapnames = DefaultConfig["mapnames"]
    p = Chain.from_seq(chicken_src_seq)
    chains = chicken_src_str.split_chains()
    chain_a = ChainStructure(next(chains))
    p.add_structure(chain_a, map_name=mapnames["map_canonical"])

    # should work on any full protein chain
    child = p.spawn_child(1, 2, keep=False, subset_structures=False)
    assert not p.children
    assert not p.seq.children
    assert all(not s.children for s in p.structures)
    assert len(child.seq) == 2
    assert child.seq.seq1 == "MG"
    assert len(child.structures) == 0

    # Using segment's boundaries
    with pytest.raises((NoOverlap, InitError)):
        _ = p.spawn_child(1, 480, "child", keep=True)
    child = p.spawn_child(1, 260, "child", keep=True)
    assert len(child.seq) == 260
    assert len(child.structures) == 1
    assert len(child.structures[0].seq) == 260
    assert "child" in [c.seq.name for c in p.children]

    # Using canonical _seq numbering
    # +-----------------------|----|------------+
    # 1                       256  260
    #                         +----|------------+
    #                         1    5
    child = p.spawn_child(
        1, 260, str_map_from=mapnames["map_canonical"], str_map_closest=True
    )
    assert len(child.seq) == 260
    s = child.structures[0]
    assert len(s.seq) == 5
    assert s.seq.start == 1
    assert s.seq.end == 5
    assert child.seq.seq1[-5:] == s.seq.seq1
    s_num = s.seq[mapnames["map_canonical"]]
    assert s_num[0] == 256
    assert s_num[-1] == 260

    child_of_child = child.spawn_child(256, 260, str_map_from=mapnames["map_canonical"])
    children = list(p.iter_children())
    assert len(children) == 2
    assert child_of_child.seq.name in [c.seq.name for c in children[-1]]

    with pytest.raises(KeyError):
        _ = p.spawn_child(
            1,
            4,
            "child",
            keep=False,
            str_map_from=mapnames["enum"],
            str_map_closest=False,
        )


def test_iter():
    def get_name(_c):
        return _c.seq.name

    c = sample_chain()

    levels = list(c.iter_children())
    assert len(levels) == 3
    assert list(map(get_name, levels[0])) == ["c1", "c2"]
    assert list(map(get_name, levels[1])) == ["c1_1", "c1_2", "c2_1", "c2_2"]
    assert list(map(get_name, levels[2])) == ["c1_2_1"]


def test_filter_children(src_chain):
    c = src_chain
    c.spawn_child(270, 300, str_map_from="map_canonical", subset_structures=False)
    assert len(c.children) == 1 and len(c.children[0].seq) == 31
    c_new = c.filter_children(lambda x: len(x._seq) > 31)
    assert len(c_new.children) == 0
    assert len(c.children) == 1
    c.filter_children(lambda x: len(x._seq) > 31, inplace=True)
    assert len(c.children) == 0


def rm_structures(c: Chain) -> Chain:
    return Chain(c.seq)


def test_apply_children(src_chain):
    c = src_chain
    c.spawn_child(270, 300, str_map_from="map_canonical", subset_structures=True)
    assert len(c.children) == 1 and len(c.children[0].seq) == 31
    c_new = c.apply_children(rm_structures)
    assert len(c.children[0].structures) == 2
    assert len(c_new.children[0].structures) == 0
    c.apply_children(rm_structures, inplace=True)
    assert len(c.children[0].structures) == 0


def test_filter_structures(src_chain):
    c = src_chain
    s0len = len(c.structures[0].seq)
    c_new = c.filter_structures(lambda x: len(x._seq) == s0len)
    assert len(c.structures) == 2 and len(c_new.structures) == 1
    assert c_new.structures[0].id == c.structures[0].id
    c.filter_structures(lambda x: len(x._seq) == s0len, inplace=True)
    assert len(c.structures) == 1


def test_apply_structures(src_chain):
    c = src_chain
    c_new = c.apply_structures(mark_meta)
    assert all("X" not in s.meta for s in c.structures)
    assert all("X" in s.meta for s in c_new.structures)
    c.apply_structures(mark_meta, inplace=True)
    assert all("X" in s.meta for s in c.structures)


@pytest.mark.parametrize(
    "ref,cs,expected",
    [
        (
            ChainSequence.from_string("ABCDEG", name="R"),
            ChainStructure(
                None,
                seq=ChainSequence.from_string(
                    "XX", name="S", numbering=[1, 3], map_canonical=[2, 5]
                ),
            ),
            ChainSequence.from_string(
                "XCDX",
                name="S",
                numbering=[1, None, None, 3],
                map_canonical=[2, 3, 4, 5],
            ),
        ),
    ],
)
def test_patch_str_seqs(ref, cs, expected):
    c = Chain(ref, [cs])
    patched = next(c.generate_patched_seqs())
    assert patched == expected
