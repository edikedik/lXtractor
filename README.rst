lXtractor
=========

.. image:: https://coveralls.io/repos/github/edikedik/lXtractor/badge.svg
    :target: https://coveralls.io/github/edikedik/lXtractor
    :alt: Branch coverage

.. image:: https://readthedocs.org/projects/lxtractor/badge/?version=latest
    :target: https://lxtractor.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation status

.. image:: https://img.shields.io/pypi/v/lXtractor.svg
    :target: https://pypi.org/project/lXtractor
    :alt: PyPi status

.. image:: https://img.shields.io/pypi/pyversions/lXtractor.svg
    :target: https://pypi.org/project/lXtractor
    :alt: Python version

.. image:: https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg
   :alt: Hatch project
   :target: https://github.com/pypa/hatch


``lXtractor`` is a toolbox devoted to feature extraction from macromolecular
sequences and structures.
It's tailored towards creating shareable local data collections anchored to
a reference sequence-based object: a single sequence, MSA, or an HMM model.
Currently, it doesn't define any unique algorithms, aiming at simplicity and
transparency.
It simply provides a (hopefully) convenient interface simplifying mundane tasks,
such as fetching the data, extracting domains, mapping sequneces, and computing
sequential and structural variables.
Sequences and structures anchored to a single reference object have a benefit
of interpretability in downstream applications, such as fitting interpretable
ML models.

Installation
============

``lXtractor`` requires python>=3.10 installed on a Unix system and is
installable via pip:

.. code::
    pip install lXtractor

We encourage users to first create a virtual environment
via ``conda`` or ``mamba``.

Usage
=====

``lXtractor`` is designed to be flexible and its usage is defined by the initial
hypothesis or a reference object that one wants to extrapolate towards the
existing sequences or structures.
Below, we'll provide a very abstract description of what this package is
intended for.

In creating data collections, one could define the following steps::

1. Assemble the data.
2. Map reference object to assembled entries' sequences.
3. Filter hits.
4. Define and calculate variables -- sequence or structure descriptors.
5. Save the data for later usage or modifications.

``lXtractor`` defines objects and routines helpful throughout this process.
Namely, ``PDB``, ``SIFTS``, ``AlphaFold``, ``fetch_uniprot()``
can aid in the first step.
Then, ``Alignment`` and ``PyHMMer`` can facilitate step 2.
At the end of the step 2 one will get a collection of ``Chain*``-type objects.
If working with sequence-only collections, these are going to be
``ChainSequence`` objects.
For structure-only data, these are going to be ``ChainStructure`` containers,
embedding ``ChainSequence`` and ``GenericStructure`` objects.
Finally, dealing with mappings between canonical sequence associated with
a group of structures will result in ``Chain`` objects.

``ChainList`` wraps ``Chain*``-type objects into a list-like collection with
useful operations allowing to quickly filter and bulk-modify ``Chain*``-type
objects.
Thus, filtering typically comes down to using ``ChainList.filter()`` method that
accepts a ``Callable[Chain*, bool]`` and returns a filtered ``ChainList``.
One can save/load the collected objects using ``ChainIO`` and proceed
with the feature extraction.

``lXtractor`` defines various sequence and structure variables.
Variable-related operations are handled by ``GenericCalculator`` and
``Manager`` classes. The former defines the calculation strategy and how
the calculations are parallelized, while the latter handles the calculations
and aggregates the results into a pandas ``DataFrame``.

As a result, one is left with a collection of ``Chain*``-type objects and a
table with calculated variables. In addition, one can store the calculated
variables within the objects themselves, although we currently do not encourage
this practice.

``lXtractor`` is in the experimental stage and under active development.
Thus, objects' interfaces may change.

For the time being, one can check the examples of
(1) `finding sequence determinants <https://eboruta.readthedocs.io/en/latest/notebooks/sequence_determinants_tutorial.html>`_
of tyrosine and serine-threonine kinases and
(2) `a protocol <https://github.com/edikedik/kinactive/blob/abae9c8a1fca0754d02e3f117dee210b587e666b/kinactive/db.py#L142>`_
to build a complete structural collection of protein kinase domains.
