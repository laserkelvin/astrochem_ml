"""
Utility functions for operating SMILES. Generally, just wraps
functionality in RDKIT, but abstracts those operations away
for people who are not familiar with working in RDKIT.
"""

from typing import List
from warnings import warn

import numpy as np
import periodictable as pt
from rdkit import Chem
from mol2vec import features


def generate_single_isos(
    smi: str, abundance_threshold: float = 0.01, explicit_h: bool = True
) -> List[str]:
    """
    Generate all singly substituted isotoplogues given an input SMILES
    string, given the isotope meets the specified threshold for Earth's
    natural abundance.

    Currently this function generates redundancies: symmetry is not
    recognized, and so there will be equivalent nuclei subsitutions.
    Similarly, we assume that the input SMILES contains the most
    common isotopologue, and with this assumption we skip over the
    most abundant isotope for each substitution.

    TODO: build in some sort of filter to remove redundancies.

    Parameters
    ----------
    smi : str
        Input SMILES string
    abundance_threshold : float, optional
        Minimum percentage natural abundance, by default 0.01.
        This value corresponds to the deuterium abundance.
    explicit_h : bool, optional
        Whether to generate D substitutions, by default True.
        Keeping in mind that this can blow up quickly!

    Returns
    -------
    List[str]
        List of SMILES isotopologues
    """
    molecule = Chem.MolFromSmiles(smi)
    if explicit_h:
        molecule = Chem.AddHs(molecule)
    isotopologues = []
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        element = getattr(pt.elements, symbol)
        # if this
        if element:
            isotopes = filter(lambda x: x.abundance >= abundance_threshold, element)
            # sort by abundance and then grab all the isotopes except the most commmon
            isotopes = sorted(isotopes, key=lambda x: x.abundance)
            masses = [int(isotope.mass) for isotope in isotopes[:-1]]
            for mass in masses:
                atom.SetIsotope(mass)
                isotopologues.append(Chem.MolToSmiles(molecule, canonical=True))
        else:
            warn(f"{symbol} not recognized by periodictable, skipping.")
            continue
        # clear isotope information
        atom.SetIsotope(0)
    return isotopologues


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
