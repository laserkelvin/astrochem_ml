
import pytest

from astrochem_ml import smiles_utils


def test_isotope_generation():
    # first off is benzene
    smiles = smiles_utils.generate_single_isos("c1ccccc1")
    assert len(smiles) == 12