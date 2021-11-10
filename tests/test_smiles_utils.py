
import pytest

from astrochem_ml import smiles_utils


def test_isotope_generation():
    # first off is benzene
    smiles = smiles_utils.generate_single_isos("c1ccccc1")
    # this value contains redundant subsitutions
    assert len(smiles) == 12


def test_all_isotope_generation():
    smiles = smiles_utils.generate_all_isos("c1ccccc1", explicit_h=False)
    # again, this contains redundant substitutions
    assert len(smiles) == 64