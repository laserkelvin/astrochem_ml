"""
Utility functions for operating SMILES. Generally, just wraps
functionality in RDKIT, but abstracts those operations away
for people who are not familiar with working in RDKIT.
"""

from itertools import product
from typing import List, Dict
from warnings import warn

import numpy as np
import periodictable as pt
from rdkit import Chem
from mol2vec import features


def get_common_masses() -> Dict[str, int]:
    return {
        atom.symbol: int(atom.mass) for atom in pt.elements
    }

def get_isotopes(molecule: Chem.Mol, abundance_threshold: float = 0.01) -> List[List[float]]:
    """
    Get the masses of each atom with sufficient natural abundance.
    For each atom in the molecule, we 

    Parameters
    ----------
    molecule : `Chem.Mol`
        Instance of an RDKIT `Mol` object
    abundance_threshold : float, optional
        Minimum percentage natural abundance, by default 0.01.
        This value corresponds to the deuterium abundance.

    Returns
    -------
    List[List[float]]
        Nested list of possible isotopes for
        each atom in the `Mol` object
    """
    atoms = [atom for atom in molecule.GetAtoms()]
    symbols = [atom.GetSymbol() for atom in atoms]
    all_isotopes = []
    for symbol in symbols:
        element = getattr(pt.elements, symbol)
        isotopes = filter(lambda x: x.abundance >= abundance_threshold, element)
        isotopes = sorted(isotopes, key=lambda x: x.abundance)
        masses = [isotope.mass for isotope in isotopes]
        all_isotopes.append(masses)
    return all_isotopes


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
    common_masses = get_common_masses()
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
                if mass == common_masses.get(symbol):
                    mass = 0
                atom.SetIsotope(mass)
                isotopologues.append(Chem.MolToSmiles(Chem.RemoveHs(molecule)))
        else:
            warn(f"{symbol} not recognized by periodictable, skipping.")
            continue
        # clear isotope information
        atom.SetIsotope(0)
    return list(set(isotopologues))


def generate_all_isos(smi: str, abundance_threshold: float = 0.01, explicit_h: bool = False) -> List[str]:
    """
    Exhaustively generate all possible combinations of isotopologues.
    Naturally, this results in a _lot_ of isotopologues, and so
    by default we ignore the hydrogen substitutions however can be
    changed by the user.

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
        All possible combinations of isotopologue
        SMILES strings. The ordering of the SMILES
        is in ascending abundance.
    """
    molecule = Chem.MolFromSmiles(smi)
    if explicit_h:
        molecule = Chem.AddHs(molecule)
    output_smiles = []
    isotopes = product(*get_isotopes(molecule, abundance_threshold))
    common_masses = get_common_masses()
    for combination in isotopes:
        for atom, mass in zip(molecule.GetAtoms(), combination):
            # substitute out the default isotope so we don't
            # ugly out the SMILES
            if int(mass) == common_masses.get(atom.GetSymbol()):
                mass = 0
            atom.SetIsotope(int(mass))
        output_smiles.append(Chem.MolToSmiles(Chem.RemoveHs(molecule)))
    return list(set(output_smiles))


def isotopologues_from_file(filepath: Union[str, Path], **kwargs) -> List[str]:
    """
    Given a file containing line-by-line SMILES strings,
    generate all possible isotopic combinations. Kwargs
    are passed to the `generate_all_isos` function, allowing
    some control over the types of isotopologues generated.

    Parameters
    ----------
    filepath : Union[str, Path]
        Filepath to the SMILES as string or
        a `pathlib.Path` object

    Returns
    -------
    List[str]
        List of all isotopologues generated
    """
    kwargs.setdefault("abundance_threshold", 0.01)
    kwargs.setdefault("explicit_h", True)
    full_list = []
    with open(filepath, "r") as read_file:
        for line in read_file.readlines():
            smi = line.strip()
            isotopologues = generate_all_isos(smi, **kwargs)
            full_list.extend(isotopologues)
    return full_list


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
