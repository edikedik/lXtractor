"""
The module defines :class:`Pocket`, representing an arbitrarily defined
binding pocket.
"""
import re

import numpy as np
from lXtractor.core import Ligand
from lXtractor.core.exceptions import ParsingError, FormatError

STATEMENT = re.compile(r"[dca]+:[,\d]+:[,\w]+ [>=<]+ \d+")


class Pocket:
    """
    A binding pocket.

    The pocket is defined via a single string following a particular syntax,
    such that, when applied to a ligand using :meth:`is_connected`, it allows
    deciding whether a ligand is connected or not. Consequently, it is tightly
    bound to :class:`lXtractor.core.ligand.Ligand`. Namely, the definition
    relies on two matrices:

    #. "c" = :attr:`lXtractor.core.ligand.Ligand.contact_mask` (boolean mask)
    #. "d" = :attr:`lXtractor.core.ligand.Ligand.dist` (distances)

    The definition comprises statements. Each statement involves the selection
    consisting of a matrix ("c" or "d"), residue positions, and residue atom
    names, formatted as::

        {matrix}:[pos]:[atom_names]

    where ``[pos]`` and ``[atom_names]`` can be comma-separated lists, a
    comparison operator, and a number (``int`` or ``float``) to compare to.
    Thus, a statement has the following format::

        {matrix}:[pos]:[atom_names] {sign} {number}

    For instance, selection ``c:1:CA,CB == 2`` translates into "must have
    exactly two contacts with atoms "CA" and "CB" at position 1. See more
    examples below.

    Comparing via ``sign`` and ``number`` have different meaning for matrices
    "c" and "d". In the former case, ``>= x`` means "at least x contacts".
    In the latter case, "<= x" means "have distance below x".

    Moreover, in case of the "d" matrix, applying selection and comparison will
    result in a ``bool`` vector still requiring an aggregation. Two aggregation
    types are supported: "da" (any) and "daa" (all). Thus, technically, three
    matrix-prefixes are supported: "c", "da", and "daa".

    Finally, statements can be bracketed and combined by boolean operators
    "AND" and "OR" (which one can abbreviate by "&" and "|").

    **Examples**:

    At least two contacts with any atom of residues 1 and 5::

        c:1,5:any >= 2

    Note that means this is a "cumulative" definition, i.e., it is applied to
    both residues at the same time. Thus, if a residue 1 has two atoms
    contacting a ligand while a residue 2 has none, this will still evaluate
    to ``True``. The following definition will ensure that each residue has
    at least two contacts::

        c:1:any >= 2 & c:2:any >= 2

    Any atoms farther than 10A from alpha-carbons of positions 1 and 10::

        da:1,10:CA > 10

    Any atoms with at least two contacts with any atoms at position 1 or
    all CA atoms closer than 6A of positions 2 and 3::

        c:1:any >= 2 | daa:2,3:CA < 6

    CA or CB atoms with a contact at position 1 but not 2, while position 3
    has any atoms below 10A threshold::

        c:1:CA,CB >= 1 & c:2:CA,CB == 0 & da:3:any <= 10

    Contact with positions 1 and 2 or positions 3 and 4::

        (c:1:any >= 1 & c:2:any >= 1) | (c:3:any >= 1 & c:4:any >= 1)

    .. seealso::
        :func:`translate_definition`.

    """

    __slots__ = ('definition', 'name')

    def __init__(self, definition: str, name: str = 'Pocket'):
        self.definition = definition
        self.name = name

    def is_connected(
        self,
        ligand: Ligand,
        mapping: dict[int, int] | None = None,
        *,
        skip_unmapped: bool = False,
    ) -> bool:
        """
        Check whether a ligand is connected to this pocket.

        .. warning::
            ``skip_unmapped=True`` may change the pocket's definition and lead
            to undesired conclusions. Caution advised!

        :param ligand: An arbitrary ligand.
        :param mapping: A mapping to the ligand's parent structure numbering.
        :param skip_unmapped: If the `mapping` is provided and some position
            is left unmapped, skip this position.
        :return: ``True`` if the ligand is bound within the pocket and ``False``
            otherwise.
        """

        translation = translate_definition(
            self.definition, mapping, skip_unmapped=skip_unmapped
        )
        a = ligand.parent.array
        c = ligand.contact_mask > 0
        d = ligand.dist
        try:
            return bool(eval(translation))
        except Exception as e:
            raise FormatError(
                f"Failed to apply definition {self.definition} translated into "
                f"{translation} to a ligand {ligand}"
            ) from e


def translate_definition(
    definition: str,
    mapping: dict[int, int] | None = None,
    *,
    skip_unmapped: bool = False,
) -> str:
    """
    Translates the :attr:`Pocket.definition` into a series of statements, such
    that, when applied to ligand matrices, evaluate to ``bool``.

    >>> translate_definition("c:1:any > 1")
    '(c[np.isin(a.res_id, [1])].sum() > 1)'
    >>> translate_definition("da:1,2:CA,CZ <= 6")
    "(d[np.isin(a.res_id, [1, 2]) & np.isin(a.atom_name, ['CA', 'CZ'])] <= 6).any()"
    >>> translate_definition("daa:1,2:any > 2", {1: 10}, skip_unmapped=True)
    '(d[np.isin(a.res_id, [10])] > 2).all()'

    :param definition: A string definition of a :class:`Pocket`.
    :param mapping: An optional mapping from the definition's numbering to
        a structure's numbering.
    :param skip_unmapped: Skip positions not present in `mapping`.
    :return: A new string with statements of the provided definition translated
        into a numpy syntax.
    """
    definition = definition.replace("AND", "&").replace("OR", "|")

    for s in STATEMENT.findall(definition):
        sel, sign, number = s.split()
        prefix, pos, atoms = sel.split(":")
        pos = list(map(int, pos.split(",")))

        if mapping:
            _pos = []
            for p in pos:
                if p in mapping:
                    _pos.append(mapping[p])
                else:
                    if skip_unmapped:
                        continue
                    else:
                        raise ParsingError(
                            f"Position {p} is not in the provided mapping"
                        )
            pos = _pos

        sel = f"np.isin(a.res_id, {pos})"

        if atoms != "any":
            atoms = atoms.split(",")
            sel = f"{sel} & np.isin(a.atom_name, {atoms})"

        if prefix == "c":
            sel = f"(c[{sel}].sum() {sign} {number})"
        elif prefix == "da":
            sel = f"(d[{sel}] {sign} {number}).any()"
        elif prefix == "daa":
            sel = f"(d[{sel}] {sign} {number}).all()"
        else:
            raise FormatError(
                f'Invalid prefix {prefix}. Supported prefixes are "c", "da", and "daa"'
            )

        definition = definition.replace(s, sel)

    return definition


if __name__ == "__main__":
    raise RuntimeError
