from lXtractor.util.io import path_tree


def test_path_tree(fake_chain_dump):
    base, paths = fake_chain_dump
    print(base)

    base, X, x1, Y = paths["base"], paths["X"], paths["x1"], paths["Y"]

    pt = path_tree(base)
    assert set(pt.nodes) == {base, X, x1, Y}
    assert set(pt.edges) == {(base, X), (X, x1), (base, Y)}

    assert "structures" in pt.nodes[base]
    assert pt.nodes[base]["structures"] == [paths["s1"]]
