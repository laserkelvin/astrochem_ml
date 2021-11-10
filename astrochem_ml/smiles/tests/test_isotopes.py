
import pytest

from astrochem_ml.smiles import isotopes


def test_isotope_generation():
    # first off is benzene
    smiles = isotopes.generate_single_isos("c1ccccc1")
    # there should only be a hydrogen and a carbon subsititution
    assert len(smiles) == 2
    # try a more unique molecule
    smiles = isotopes.generate_single_isos("C(C(=O)O)N")
    # this should include 17 and 16 O
    assert len(smiles) == 10


def test_all_isotope_generation():
    smiles = isotopes.generate_all_isos("c1ccccc1", explicit_h=True)
    assert len(smiles) == 430
    smiles = isotopes.generate_all_isos("c1ccccc1", explicit_h=False)
    # without hydrogens, there should be 13 unique combinations
    assert len(smiles) == 13