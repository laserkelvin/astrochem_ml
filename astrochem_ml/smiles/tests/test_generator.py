
import pytest

from astrochem_ml.smiles import MoleculeGenerator


def test_molecule_generation():
    benzene = MoleculeGenerator("c1ccccc1", ["c", "cC#N", "cC=O", "cN"], seed=120)
    benzene.grow_tree(50)
    assert len(benzene.nodes) == 35
    assert "Nc1cc(C#N)c(N)cc1C#N" in benzene.nodes
    assert "N#Cc1ccc(C=O)cc1" in benzene.nodes