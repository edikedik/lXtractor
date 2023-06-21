"""
The module defines :class:`Pocket`, representing an arbitrarily defined
binding pocket.
"""
import re
from collections import abc

import numpy as np
from lXtractor.core import Ligand
from lXtractor.core.exceptions import ParsingError, FormatError

__all__ = ("Pocket", "translate_definition", "make_sel")

STATEMENT = re.compile(r"[dcsa]+:[,\d]+:[,\w]+ [>=<]+ \d+")


class Pocket:
    """
    A binding pocket.

    The pocket is defined via a single string following a particular syntax
    (a definition), such that, when applied to a ligand using
    :meth:`is_connected`, the latter outputs ``True`` if ligand is connected.
    Consequently, it is tightly bound to :class:`lXtractor.core.ligand.Ligand`.
    Namely, the definition relies on two matrices:

    #. "c" = :attr:`lXtractor.core.ligand.Ligand.contact_mask` (boolean mask)
    #. "d" = :attr:`lXtractor.core.ligand.Ligand.dist` (distances)

    The definition is a combination of statements. Each statement involves the
    selection consisting of a matrix ("c" or "d"), residue positions, and
    residue atom names, formatted as::

        {matrix-prefix}:[pos]:[atom_names] {sign} {number}

    where ``[pos]`` and ``[atom_names]`` can be comma-separated lists, ``sign``
    is` a comparison operator, and a ``number`` (``int`` or ``float``) is what
    to compare to. For instance, selection ``c:1:CA,CB == 2`` translates into
    "must have exactly two contacts with atoms "CA" and "CB" at position 1.
    See more examples below.

    Comparison meaning depends on the matrix type used: "c" or "d".

    In the former case, ``>= x`` means "at least x contacts".
    In the latter case, "<= x" means "have distance below x".

    In the case of the "d" matrix, applying selection and comparison will
    result in a vector of ``bool`` bool values, requiring an aggregation.
    Two aggregation types are supported: "da" (any) and "daa" (all).

    In the case of the "c" matrix, possible matrix prefixes are "c" and "cs".
    They have very different meanings! In the former case, the statements
    compares the total number of contacts when the selection is applied.
    In the latter case, the statement will select residues **separately** and,
    for each residue, decide whether the selected atoms form enough contact to
    extrapolate towards the full residue and mark it as "contacting"
    (controlled via `min_contacts`). These decisions are summed across each
    residue and this sum is compared to the number in the statement.
    See the example below.

    Finally, statements can be bracketed and combined by boolean operators
    "AND" and "OR" (which one can abbreviate by "&" and "|").

    **Examples**:

    At least two contacts with any atom of residues 1 and 5::

        c:1,5:any >= 2

    Note that the above is a "cumulative" statement, i.e., it is applied to
    both residues at the same time. Thus, if a residue 1 has two atoms
    contacting a ligand while a residue 2 has none, this will still evaluate
    to ``True``. The following definition will ensure that each residue has
    at least two contacts::

        c:1:any >= 2 & c:2:any >= 2

    In contrast, the following statement will translate "among residues 1, 2,
    and 3, there are at least two "contacting" residues::

        cs:1,2,3:any >= 2

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

    __slots__ = ("definition", "name")

    def __init__(self, definition: str, name: str = "Pocket"):
        self.definition = definition
        self.name = name

    def is_connected(
        self,
        ligand: Ligand,
        mapping: dict[int, int] | None = None,
        **kwargs,
    ) -> bool:
        """
        Check whether a ligand is connected.

        :param ligand: An arbitrary ligand.
        :param mapping: A mapping to the ligand's parent structure numbering.
        :param kwargs: Passed to :func:`translate_definition`.
        :return: ``True`` if the ligand is bound within the pocket and ``False``
            otherwise.
        """

        translation = translate_definition(self.definition, mapping, **kwargs)
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


def make_sel(pos: int | abc.Sequence[int], atoms: str) -> str:
    """
    Make a selection string from positions and atoms.

    >>> make_sel(1, 'any')
    '(a.res_id == 1)'
    >>> make_sel([1, 2], 'CA,CB')
    "np.isin(a.res_id, [1, 2]) & np.isin(a.atom_name, ['CA', 'CB'])"

    :param pos:
    :param atoms:
    :return:
    """
    if isinstance(pos, int):
        sel = f"(a.res_id == {pos})"
    else:
        sel = f"np.isin(a.res_id, {pos})"
    if atoms != "any":
        sel = f"{sel} & np.isin(a.atom_name, {atoms.split(',')})"
    return sel


def translate_definition(
    definition: str,
    mapping: dict[int, int] | None = None,
    *,
    skip_unmapped: bool = False,
    min_contacts: int = 1,
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
    >>> translate_definition("cs:1,2:any > 2")
    'sum([c[(a.res_id == 1)].sum() >= 1, c[(a.res_id == 2)].sum() >= 1]) > 2'

    .. warning::
        ``skip_unmapped=True`` may change the pocket's definition and lead
        to undesired conclusions. Caution advised!

    :param definition: A string definition of a :class:`Pocket`.
    :param mapping: An optional mapping from the definition's numbering to
        a structure's numbering.
    :param skip_unmapped: If the `mapping` is provided and some position
            is left unmapped, skip this position.
    :param min_contacts: If prefix is "cs", use this threshold to determine a
        minimum number of residue contacts required to consider it bound.
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

        if prefix == "cs":
            selections = ", ".join(
                f"c[{make_sel(p, atoms)}].sum() >= {min_contacts}" for p in pos
            )
            statement = f"sum([{selections}]) {sign} {number}"
        else:
            sel = make_sel(pos, atoms)
            if prefix == "c":
                statement = f"(c[{sel}].sum() {sign} {number})"
            elif prefix == "da":
                statement = f"(d[{sel}] {sign} {number}).any()"
            elif prefix == "daa":
                statement = f"(d[{sel}] {sign} {number}).all()"
            else:
                raise FormatError(
                    f"Invalid prefix {prefix}. Supported prefixes are: "
                    '"c", "cs", "da", and "daa"'
                )

        definition = definition.replace(s, statement)

    return definition


if __name__ == "__main__":
    raise RuntimeError
