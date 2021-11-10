
import pytest

from rdkit import Chem

from astrochem_ml.smiles import functionals


def test_replace_base():
    molecule = Chem.MolFromSmiles("c1ccccc1")
    aromatic_c = Chem.MolFromSmarts("c")
    alcohol = Chem.MolFromSmarts("cCO")
    results = functionals.replace_substructure(molecule, aromatic_c, alcohol)
    assert len(results) == 6
    # now filter out the non-uniques
    smiles = [Chem.MolToSmiles(result) for result in results]
    unique_smiles = list(set(smiles))
    assert len(unique_smiles) == 1
    assert unique_smiles.pop() == "OCc1ccccc1"


def test_replace_aromatics():
    molecule = Chem.MolFromSmiles("c1ccccc1")
    cyanide = Chem.MolFromSmarts("cC#N")
    results = functionals.replace_aromatic_hydrogens(molecule, cyanide)
    assert len(results) == 1
    assert results.pop() == "N#Cc1ccccc1"