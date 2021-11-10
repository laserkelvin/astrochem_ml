
from warnings import warn

import numpy as np
from rdkit import Chem
from mol2vec import features

def canonicize_smiles(smi: str) -> str:
    """
    Simple function to canonicize an input SMILES string.
    This is useful for homogenizing a dataset of SMILES, and
    getting rid of duplicate entries.

    Parameters
    ----------
    smi : str
        Input SMILES string

    Returns
    -------
    str
        Canonicized SMILES string
    """
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        warn(f"{smi} could not be converted to a `Molecule` object.")


def smi_to_vector(smi: str, model, radius: int = 1) -> np.ndarray:
    """
    Given an embedding model and SMILES string, generate the corresponding
    molecule vector.

    Parameters
    ----------
    smi : str
        Input SMILES string
    model : [type]
        mol2vec object
    radius : int, optional
        Radius used for Morgan FPs, by default 1

    Returns
    -------
    np.ndarray
        N-dimensional vector corresponding
        to the molecule embedding
    """
    # Molecule from SMILES will break on "bad" SMILES; this tries
    # to get around sanitization (which takes a while) if it can
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    # generate a sentence from rdkit molecule
    sentence = features.mol2alt_sentence(mol, radius)
    # generate vector embedding from sentence and model
    vector = features.sentences2vec([sentence], model)
    return vector