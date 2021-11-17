from typing import Union, List

from rdkit import Chem


def replace_substructure(
    molecule: Chem.Mol, pattern: Chem.Mol, replace: Chem.Mol, **kwargs
) -> List[Chem.Mol]:
    """
    Simple wrapper function around the RDKIT function. This is
    mostly implemented for convenience and to homogenize other
    functions developed in `astrochem_ml`.

    Parameters
    ----------
    molecule : Chem.Mol
        An instance of a RDKIT `Mol` object
    pattern : Union[str, Chem.Mol]
        Either a SMARTS string, or an instance of
        an RDKIT `Mol` object to match with.
    replace : Union[str, Chem.Mol]
        Either a SMARTS string, or an instance of
        an RDKIT `Mol` object to replace with.

    Returns
    -------
    List[Chem.Mol]
        A list of all matched substitutions
    """
    return Chem.ReplaceSubstructs(molecule, pattern, replace, **kwargs)


def replace_aromatic_hydrogens(
    molecule: Chem.Mol, replace: Union[str, Chem.Mol], **kwargs
) -> List[Chem.Mol]:
    """
    Sequentially replaces all hydrogens bonded to an aromatic carbon
    or nitrogen with a specified functional group. The replacement
    can be specified with SMARTS as a string, or with a `Chem.Mol`
    object generated with `Chem.MolFromSmarts`.

    Keep in mind that because the matching is done with an aromatic
    carbon/nitrogen _and_ hydrogen, the SMARTS should include the
    aromatic atom we're removing.

    An example usage would be to replace hydrogens with cyanide
    groups, and the corresponding function call would be:

    ```
    > cyanide = Chem.MolFromSmarts("cC#N")
    > replace_aromatic_hydrogens(molecule, cyanide)
    ```

    Parameters
    ----------
    molecule : Chem.Mol
        An instance of a RDKIT `Mol` object
    replace : Union[str, Chem.Mol]
        Either a SMARTS string, or an instance of
        an RDKIT `Mol` object to replace with.

    Returns
    -------
    List[Chem.Mol]
        All unique combinations of substitutions as
        SMILES strings
    """
    if isinstance(replace, str):
        replace = Chem.MolFromSmarts(replace)
    pattern = Chem.MolFromSmarts("[c,n;H1]")
    molecules = replace_substructure(molecule, pattern, replace, **kwargs)
    return list(set([Chem.MolToSmiles(mol) for mol in molecules]))
