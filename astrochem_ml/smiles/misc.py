from warnings import warn

import numpy as np
from rdkit import Chem


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


def generate_cartesian_coords(molecule: Chem.Mol):
    """
    TODO

    Parameters
    ----------
    molecule : Chem.Mol
        [description]
    """
    pass
