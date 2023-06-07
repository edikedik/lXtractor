import typing as t
from pathlib import Path
from tempfile import TemporaryDirectory

from more_itertools import consume

from lXtractor.core.chain import ChainStructure, ChainIO, Chain, ChainSequence
from lXtractor.core.chain.io import read_chains, ChainIOConfig
from lXtractor.core.config import DumpNames
from lXtractor.util.io import get_files, get_dirs

T = t.TypeVar("T")


def test_chainio(simple_structure, simple_chain_seq):
    fields, seq = simple_chain_seq
    struc = ChainStructure.from_structure(simple_structure)
    seq_child = seq.spawn_child(1, 2)
    ch = Chain(seq, [struc], children=[Chain(seq_child)])
    io = ChainIO(tolerate_failures=False)

    # chain seq io
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        consume(io.write(seq, tmp))
        files = get_files(tmp)
        assert DumpNames.sequence in files
        assert DumpNames.meta in files
        s_r = list(io.read_chain_seq(tmp))
        assert len(s_r) == 1
        s_r = s_r.pop()
        assert s_r.id == seq.id

        consume(io.write([seq, seq], tmp))
        dirs = get_dirs(tmp)
        assert seq.id in dirs

    # chain structure io
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        consume(io.write(struc, tmp))
        files = get_files(tmp)
        stems = [x.stem for x in files.values()]
        assert DumpNames.structure_base_name in stems
        assert DumpNames.sequence in files
        assert DumpNames.meta in files

        consume(io.write([struc, struc], tmp))
        dirs = get_dirs(tmp)
        assert struc.id in dirs

        objs = list(io.read_chain_str(tmp))
        assert len(objs) == 1
        s_r = objs.pop()
        assert s_r.id == struc.id
        assert s_r.pdb.id == struc.pdb.id
        assert s_r.pdb.chain == struc.pdb.chain
        assert s_r.pdb.structure is not None
        assert s_r.seq.seq1 == struc.seq.seq1

    # chain io
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        consume(io.write(ch, tmp))
        files = get_files(tmp)
        dirs = get_dirs(tmp)
        assert DumpNames.sequence in files
        assert DumpNames.meta in files
        assert DumpNames.structures_dir in dirs

        struc_paths = list(dirs[DumpNames.structures_dir].glob("*"))
        assert len(struc_paths) == 1
        struc_path = struc_paths.pop()
        assert struc_path.name == struc.id

        assert DumpNames.segments_dir in dirs
        segm_paths = list(dirs[DumpNames.segments_dir].glob("*"))
        assert len(segm_paths) == 1
        segm_dir = segm_paths.pop()
        assert segm_dir.name == f'Chain({seq_child.id})'

        c_r = list(io.read_chain(tmp, search_children=True))
        assert len(c_r) == 1
        c_r = c_r.pop()
        assert isinstance(c_r, Chain)
        assert c_r.id == ch.id
        assert c_r.seq.id == ch.seq.id
        assert len(c_r.structures) == 1
        assert len(c_r.children) == 1

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        consume(io.write([ch, ch], tmp))
        dirs = get_dirs(tmp)
        assert ch.id in dirs

        objs = list(io.read_chain(tmp))
        assert len(objs) == 1
        c_r = objs.pop()
        assert c_r.id == ch.id


def test_chainio_parallel(simple_structure, simple_chain_seq):
    fields, seq = simple_chain_seq

    io = ChainIO(num_proc=2)

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        res = list(io.write([seq, seq], tmp))
        assert len(res) == 2

        dirs = get_dirs(tmp)
        assert len(dirs) == 1

        c_r = list(io.read_chain_seq(tmp))

        assert len(c_r) == 1


def change_name(obj: T) -> T:
    if isinstance(obj, (Chain, ChainStructure)):
        obj.seq.name = "BAM"
    elif isinstance(obj, ChainSequence):
        obj.name = "DOUBLE BAM"
    return obj


def test_read_chains(simple_structure, simple_chain_seq):
    fields, seq = simple_chain_seq
    cs = ChainStructure.from_structure(simple_structure)
    c = Chain(seq, structures=[cs])
    child1 = c.spawn_child(1, 3, name="C1")
    _ = child1.spawn_child(1, 2, name="C2")

    io = ChainIO()

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        consume(io.write([c], tmp, write_children=True))
        assert (tmp / c.id).exists()

        chains = read_chains(
            [tmp / c.id],
            children=True,
            seq_cfg=ChainIOConfig(),
            str_cfg=ChainIOConfig(num_proc=2),
            seq_callbacks=[change_name],
            str_callbacks=[change_name],
        )
        assert len(chains) == 1
        c_ = chains.pop()
        assert len(c_.structures) == 1
        assert len(c_.children) == 1
        child1 = c_.children[0]
        assert len(child1.structures) == 1
        assert len(child1.children) == 1
        child2 = child1.children[0]
        assert len(child2.structures) == 1
        assert len(child2.children) == 0

        assert child2.structures[0].name == "BAM"
        assert child1.name == "DOUBLE BAM"
