from collections import abc

import lXtractor.chain as lxc


def filter_chain_structures(
    c: lxc.Chain, fn: abc.Callable[[lxc.ChainStructure], bool]
) -> lxc.Chain:
    c.structures = c.structures.filter(fn)
    return c


if __name__ == "__main__":
    raise RuntimeError
