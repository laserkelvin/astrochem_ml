from itertools import product
from typing import Dict, List, Union
from pathlib import Path
from warnings import warn

from rdkit import Chem
import periodictable as pt
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def get_common_masses() -> Dict[str, int]:
    return {atom.symbol: round(atom.mass) for atom in pt.elements}


def get_isotopes(
    molecule: Chem.Mol, abundance_threshold: float = 0.01
) -> List[List[float]]:
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
            masses = [round(isotope.mass) for isotope in isotopes[:-1]]
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


def generate_all_isos(
    smi: str, abundance_threshold: float = 0.01, explicit_h: bool = False
) -> List[str]:
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
            if round(mass) == common_masses.get(atom.GetSymbol()):
                mass = 0
            atom.SetIsotope(round(mass))
        output_smiles.append(Chem.MolToSmiles(Chem.RemoveHs(molecule)))
    return list(set(output_smiles))


def isotopologues_from_file(
    filepath: Union[str, Path], n_jobs: int = 1, verbose: int = 0, **kwargs
) -> List[str]:
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
    kwargs.setdefault("explicit_h", False)
    with open(filepath, "r") as read_file:
        smiles = [line.strip() for line in read_file.readlines()]
    with Parallel(n_jobs, verbose=verbose) as worker:
        isotopologues = worker(
            delayed(generate_all_isos)(smi, **kwargs) for smi in tqdm(smiles)
        )
    # flatten the list
    isotopologues = [val for sublist in isotopologues for val in sublist]
    return isotopologues
