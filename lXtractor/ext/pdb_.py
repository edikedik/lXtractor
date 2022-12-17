import json
from collections import abc
from itertools import chain
from pathlib import Path

from more_itertools import peekable, unzip

from lXtractor.core.base import UrlGetter
from lXtractor.ext.base import ApiBase, fetch_files
from lXtractor.util.io import fetch_max_trials, download_text, fetch_iterable


def url_getters() -> dict[str, UrlGetter]:
    def _url_getter_factory(name, *args):
        args_fn = ', '.join(args)
        args_url = '/'.join(f'{{{x}}}' for x in args)
        fn = f'lambda {args_fn}: f"{base}/{name}/{args_url}"'
        return eval(fn)

    base = 'https://data.rcsb.org/rest/v1/core'

    staged = [
        # Single argument group
        ('chem_comp', 'comp_id'), ('drugbank', 'comp_id'),
        ('entry', 'entry_id'), ('pubmed', 'entry_id'),
        ('entry_groups', 'group_id'), ('polymer_entity_groups', 'group_id'),
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
        ('interface', 'entry_id', 'assembly_id', 'interface_id')
    ]

    d = {x[0]: _url_getter_factory(*x) for x in staged}
    d['files'] = (lambda entry_id, fmt: f'https://files.rcsb.org/download/{entry_id}.{fmt}')

    return d


class PDB(ApiBase):
    """
    Basic RCSB PDB interface to fetch structures and information.

    Fetch structure files from RCSB PDB.

    >>> pdb = PDB()
    >>> fetched, failed = pdb.fetch_structures(['2src', '2oiq'])
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
            self, ids: abc.Iterable[str], fmt: str = 'cif',
            pdb_dir: Path | None = None, *, overwrite: bool = False,
    ) -> tuple[list[tuple[str, Path | str]], list[str]]:
        """
        Fetch structure files from RCSB PDB.

        >>> pdb = PDB()
        >>> fetched, failed = pdb.fetch_structures(['2src', '2oiq'])
        >>> len(fetched) == 2 and len(failed) == 0
        True
        >>> (id1, res1), (id2, res2) = fetched
        >>> assert {id1, id2} == {'2src', '2oiq'}
        >>> isinstance(res1, str) and isinstance(res2, str)
        True

        .. seealso::
            :func:`fetch_files lXtractor.ext.base.fetch_files`.

        :param ids: An iterable over PDB IDs.
        :param pdb_dir: Dir to save files to. If ``None``, will keep downloaded structures as strings.
        :param fmt: Structure format.
        :param overwrite: Overwrite existing files if `pdb_dir` is provided.
        :return: A tuple with fetched results and the remaining IDs. The former is a list of tuples,
            where the first element is the original ID, and the second element is either the path to
            a downloaded file or downloaded data as string. The order may differ.
            The latter is a list of IDs that failed to fetch.
        """
        return fetch_files(
            self.url_getters['files'], ids, fmt, pdb_dir,
            overwrite=overwrite, max_trials=self.max_trials,
            num_threads=self.num_threads, verbose=self.verbose
        )

    def fetch_info(
            self, service_name: str,
            url_args: abc.Iterable[tuple[str, ...]]
    ) -> tuple[list[tuple[tuple[str, ...], dict]], list[tuple[str, ...]]]:
        """

        >>> pdb = PDB()
        >>> fetched, failed = pdb.fetch_info('entry',[('2SRC', ), ('2OIQ', )])
        >>> len(failed) == 0 and len(fetched) == 2
        True
        >>> (args1, res1), (args2, res2) = fetched
        >>> assert {args1, args2} == {('2SRC', ), ('2OIQ', )}
        >>> assert isinstance(res1, dict) and isinstance(res2, dict)

        :param service_name: The name of the service to use. Check :meth:`url_args`.
        :param url_args: Arguments to a `url_getter`. Check :meth:`url_args` to see which getters
            require which arguments.
        :return: A tuple with fetched and remaining inputs. Fetched inputs are tuples, where the
            first element is the original arguments and the second argument is the dictionary
            with downloaded data. Remaining inputs are arguments that failed to fetch.
        """

        def fetch_one(args: tuple[str, ...]) -> dict:
            return json.loads(download_text(url_getter(*args)))

        def fetcher(chunk: abc.Iterable[tuple[str, ...]]) -> list[tuple[tuple[str, ...], dict]]:
            chunk = peekable(chunk)
            if not chunk.peek(False):
                return []

            return list(fetch_iterable(
                chunk, fetcher=fetch_one, num_threads=self.num_threads, verbose=self.verbose)
            )

        def get_remaining(
                fetched: abc.Iterable[tuple[tuple[str, ...], dict]],
                _remaining: list[tuple[str, ...]]
        ) -> list[tuple[str, ...]]:
            args, _ = unzip(fetched)
            return list(set(_remaining) - set(args))

        url_getter = self.url_getters[service_name]

        results, remaining = fetch_max_trials(
            url_args, fetcher=fetcher, get_remaining=get_remaining,
            max_trials=self.max_trials, verbose=self.verbose
        )

        results = list(chain.from_iterable(results))

        return results, remaining


if __name__ == '__main__':
    raise RuntimeError
