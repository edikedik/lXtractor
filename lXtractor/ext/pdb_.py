"""
Utilities to interact with the RCSB PDB database.
"""
import json
from collections import abc
from itertools import repeat
from pathlib import Path

from lXtractor.core.base import UrlGetter
from lXtractor.ext.base import ApiBase
from lXtractor.util.io import fetch_files


def url_getters() -> dict[str, UrlGetter]:
    """
    :return: A dictionary with {name: getter} where getter is a function
        accepting string args and returning a valid URL.
    """

    def _url_getter_factory(name, *args):
        args_fn = ', '.join(args)
        args_url = '/'.join(f'{{{x}}}' for x in args)
        fn = f'lambda {args_fn}: f"{base}/{name}/{args_url}"'
        return eval(fn)  # pylint: disable=eval-used

    base = 'https://data.rcsb.org/rest/v1/core'

    staged = [
        # Single argument group
        ('chem_comp', 'comp_id'),
        ('drugbank', 'comp_id'),
        ('entry', 'entry_id'),
        ('pubmed', 'entry_id'),
        ('entry_groups', 'group_id'),
        ('polymer_entity_groups', 'group_id'),
        ('group_provenance', 'group_provenance_id'),
        # Two arguments group
        ('assembly', 'entry_id', 'assembly_id'),
        ('branched_entity', 'entry_id', 'entity_id'),
        ('nonpolymer_entity', 'entry_id', 'entity_id'),
        ('polymer_entity', 'entry_id', 'entity_id'),
        ('branched_entity_instance', 'entry_id', 'asym_id'),
        ('nonpolymer_entity_instance', 'entry_id', 'asym_id'),
        ('polymer_entity_instance', 'entry_id', 'asym_id'),
        ('uniprot', 'entry_id', 'entity_id'),
        # Three argument group
        ('interface', 'entry_id', 'assembly_id', 'interface_id'),
    ]

    result = {x[0]: _url_getter_factory(*x) for x in staged}
    result['files'] = lambda entry_id, fmt: (
        f'https://files.rcsb.org/download/{entry_id}.{fmt}'
    )

    return result


class PDB(ApiBase):
    """
    Basic RCSB PDB interface to fetch structures and information.

    Fetch structure files from RCSB PDB.

    >>> pdb = PDB()
    >>> fetched, failed = pdb.fetch_structures(['2src', '2oiq'],)
    >>> len(fetched) == 2 and len(failed) == 0
    True
    >>> (id1, res1), (id2, res2) = fetched
    >>> assert {id1, id2} == {'2src', '2oiq'}
    >>> isinstance(res1, str) and isinstance(res2, str)
    True
    """

    def __init__(
        self, max_trials: int = 1, num_threads: int | None = None, verbose: bool = False
    ):
        super().__init__(url_getters(), max_trials, num_threads, verbose)

    def fetch_structures(
        self,
        ids: abc.Iterable[str],
        dir_: Path | None,
        fmt: str = 'cif',
        *,
        overwrite: bool = False,
    ) -> tuple[list[tuple[tuple[str, str], Path | str]], list[tuple[str, str]]]:
        """
        Fetch structure files from RCSB PDB as text.

        >>> pdb = PDB()
        >>> fetched, failed = pdb.fetch_structures(['2src', '2oiq'], dir_=None)
        >>> len(fetched) == 2 and len(failed) == 0
        True
        >>> (args1, res1), (args2, res2) = fetched
        >>> assert {args1, args2} == {('2src', 'cif'), ('2oiq', 'cif')}
        >>> isinstance(res1, str) and isinstance(res2, str)
        True

        .. seealso::
            :func:`fetch_files lXtractor.ext.base.fetch_files`.

        :param ids: An iterable over PDB IDs.
        :param dir_: Dir to save files to. If ``None``, will keep downloaded
            files as strings.
        :param fmt: Structure format.
        :param overwrite: Overwrite existing files if `dir_` is provided.
        :return: A tuple with fetched results and the remaining IDs.
            The former is a list of tuples, where the first element
            is the original ID, and the second element is either the path to
            a downloaded file or downloaded data as string. The order
            may differ. The latter is a list of IDs that failed to fetch.
        """
        return fetch_files(
            self.url_getters['files'],
            zip(ids, repeat(fmt)),
            fmt,
            dir_,
            overwrite=overwrite,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )

    def fetch_info(
        self,
        service_name: str,
        url_args: abc.Iterable[tuple[str, ...]],
        dir_: Path | None,
        *,
        overwrite: bool = False,
    ) -> tuple[list[tuple[tuple[str, ...], dict]], list[tuple[str, ...]]]:
        """

        >>> pdb = PDB()
        >>> fetched, failed = pdb.fetch_info(
        ...     'entry', [('2SRC', ), ('2OIQ', )], dir_=None)
        >>> len(failed) == 0 and len(fetched) == 2
        True
        >>> (args1, res1), (args2, res2) = fetched
        >>> assert {args1, args2} == {('2SRC', ), ('2OIQ', )}
        >>> assert isinstance(res1, dict) and isinstance(res2, dict)

        :param service_name: The name of the service to use.
        :param dir_: Dir to save files to. If ``None``, will keep downloaded
            files as strings.
        :param url_args: Arguments to a `url_getter`. Check :meth:`url_args`
            to see which getters require which arguments.
        :param overwrite: Overwrite existing files if `dir_` is provided.
        :return: A tuple with fetched and remaining inputs.
            Fetched inputs are tuples, where the first element is the original
            arguments and the second argument is the dictionary with downloaded
            data. Remaining inputs are arguments that failed to fetch.
        """
        return fetch_files(
            self.url_getters[service_name],
            url_args,
            'json',
            dir_,
            callback=json.loads,
            overwrite=overwrite,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )


if __name__ == '__main__':
    raise RuntimeError
