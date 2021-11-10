============
Astrochem ML
============


.. image:: https://img.shields.io/pypi/v/astrochem_ml.svg
        :target: https://pypi.python.org/pypi/astrochem_ml

.. image:: https://img.shields.io/travis/laserkelvin/astrochem_ml.svg
        :target: https://travis-ci.com/laserkelvin/astrochem_ml

.. image:: https://readthedocs.org/projects/astrochem-ml/badge/?version=latest
        :target: https://astrochem-ml.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Doing astrochemistry with robots.

The `astrochem_ml` package is designed for bringing accessible cheminformatics to
astrochemical discovery. The main features, some of which are currently active
development, are interfaces to common operations using RDKit that are relevant
to astrochemistry, and pre-trained embedding models ready for machine learning
projects that combine molecules and astrophysics.

The plan is to deliver a general purpose library, in addition to providing a
command line interface to several common tasks.


* Free software: MIT license
* Documentation: https://astrochem-ml.readthedocs.io.

Installation
------------

Not yet on PyPI, and so for now you can install `astrochem_ml` via:

```pip install git+https://github.com/laserkelvin/astrochem_ml```

Features
--------

Molecule generation
===================

A significant amount of functionality wraps the `rdkit` package, the main library
for doing cheminformatics in Python. For all molecule interactions, we go back
and forth between the native `rdkit` objects and SMILES/SMARTS strings.

* Exhaustive isotopologue generation in SMILES

.. code-block:: python

        >>> from astrochem_ml.smiles import isotopes
        # exhaustively enumerate all possible combinations isotopologues
        # user can set the threshold for natural abundance and whether
        # to include hydrogens
        >>> isotopes.generate_all_isos("c1ccccc1", explicit_h=False)
        ['c1[13cH]c[13cH][13cH][13cH]1', ... 'c1ccccc1', '[13cH]1[13cH][13cH][13cH][13cH][13cH]1','c1c[13cH][13cH][13cH]c1']

* Functional group substitutions

Replace substructures with other ones in a tree data structure!

.. code-block:: python

        >>> from astrochem_ml.smiles import MoleculeGenerator
        # randomly grow out possible structures starting from benzene,
        # and iteratively replace structures with other functional groups
        >>> benzene = MoleculeGenerator("c1ccccc1", substructs=["c", "cC#N", "cC=O", "cN"])
        >>> benzene.grow_tree(50)
        100%|██████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 237.44it/s]
        >>> print(benzene)
        c1ccccc1
        ├── Nc1ccccc1
        ├── N#Cc1ccccc1
        └── O=Cc1ccccc1
        ├── Nc1ccccc1C=O
        │   └── N#Cc1ccccc1C=O
        ├── Nc1cccc(C=O)c1
        │   ├── Nc1cccc(C=O)c1N
        │   │   ├── Nc1c(C=O)ccc(C=O)c1N
        │   │   ├── Nc1cc(C=O)cc(C=O)c1N
        ...

This provides a high level interface to view every structure generated,
and from which parent.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
