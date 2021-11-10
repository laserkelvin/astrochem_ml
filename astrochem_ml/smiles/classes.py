from typing import List, Union, Tuple, Type

from anytree import Node, RenderTree, NodeMixin
from rdkit import Chem
import numpy as np
from tqdm.auto import tqdm

from astrochem_ml.smiles.functionals import replace_substructure


class MoleculeNode(NodeMixin):
    def __init__(self, molecule: Chem.Mol, parent=None, children=None) -> None:
        super().__init__()
        self.molecule = molecule
        self.parent = parent
        if children:
            self.children = children

    def __str__(self) -> str:
        return Chem.MolToSmiles(self.molecule)

    def __repr__(self) -> str:
        return Chem.MolToSmiles(self.molecule)

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash((str(self)))


class MoleculeGenerator(object):
    def __init__(
        self,
        starting_smiles: str,
        substructs: List[Union[str, Chem.Mol]],
        seed: Union[None, int] = None,
    ):
        self.initial_smi = starting_smiles
        self._nodes = [
            MoleculeNode(self.initial_molecule),
        ]
        self.substructs = substructs
        self.rng = seed

    @property
    def initial_smi(self) -> str:
        return self._initial_smi

    @initial_smi.setter
    def initial_smi(self, value: str) -> None:
        # check if this is a valid SMILES string
        mol = Chem.MolFromSmiles(value)
        if not mol:
            raise ValueError(f"{value} is not a valid SMILES string!")
        self._initial_smi = value

    @property
    def initial_molecule(self) -> Chem.Mol:
        return Chem.MolFromSmiles(self.initial_smi)

    @property
    def substructs(self) -> List[Chem.Mol]:
        """
        Returns the list of substructures in the form of
        RDKIT `Mol` objects. When growing the tree, we will
        randomly sample from this list for the matching
        and the replacements.

        Returns
        -------
        List[Chem.Mol]
            Returns the current list of substructures
        """
        return self._substructs

    @substructs.setter
    def substructs(self, substructs: List[Union[str, Chem.Mol]]) -> None:
        if isinstance(substructs[0], str):
            substructs = [Chem.MolFromSmarts(sub) for sub in substructs]
        self._substructs = substructs

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, value: int):
        self._rng = np.random.default_rng(value)

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @property
    def node_ids(self) -> np.ndarray:
        return np.arange(len(self.nodes))

    def get_node(self, index: int) -> Node:
        return self.nodes[index]

    @property
    def smiles(self) -> List[str]:
        """
        Return the list of SMILES strings for each
        node in the tree.

        Returns
        -------
        List[str]
            Canonical SMILES strings for each node
        """
        return [str(node) for node in self.nodes]

    def add_nodes(self, molecules: List[Chem.Mol], parent: Node) -> None:
        """
        Function to add unique nodes to the current tree. We
        iterate through each of the new molecules, and check
        whether it's contained in the list of SMILES of all nodes
        currently in the tree.

        Parameters
        ----------
        molecules : List[Chem.Mol]
            List of Mol objects
        parent : Node
            The parent node corresponding to this molecule.
        """
        # filter out non-unique smiles that aren't already in our tree
        for molecule in molecules:
            if Chem.MolToSmiles(molecule) not in self.smiles:
                self.nodes.append(MoleculeNode(molecule, parent=parent))

    def find_node(self, node: MoleculeNode) -> Tuple[int, MoleculeNode]:
        """
        Traverses the entire list of nodes looking for a
        match to the node we wish to search for. This
        is controlled by the __eq__ method for `MoleculeNode`.

        Parameters
        ----------
        node : MoleculeNode
            Target `MoleculeNode` to search for

        Returns
        -------
        Tuple[int, MoleculeNode]
            The index of the node in the tree, as well as
            the node itself.
        """
        return tuple(filter(lambda x: x[1] == node, enumerate(self.nodes)))

    @property
    def random_node(self) -> MoleculeNode:
        """
        Grab a random node from the current tree.
        The random choice is dictated by the object `rng`
        object.

        Returns
        -------
        MoleculeNode
            A randomly chosen node from the tree
        """
        return self.rng.choice(self.nodes)

    @property
    def random_substructure(self) -> MoleculeNode:
        """
        Grab a random substructure from the list of
        substructures. The random choice is dictated by
        the object `rng` object.

        Returns
        -------
        MoleculeNode
            A randomly chosen node from the tree
        """
        return self.rng.choice(self.substructs)

    @property
    def rdkit_molecules(self) -> List[Chem.Mol]:
        """
        Return a list of RDKIT Mol objects from the current
        tree. This is intended to allow the user further
        manipulation of the resulting structures.

        Returns
        -------
        List[Chem.Mol]
            List of RDKIT Mol objects
        """
        return [node.molecule for node in self.nodes]

    def __repr__(self) -> str:
        return str(RenderTree(self.nodes[0]))

    def grow_tree(self, num_iterations: int):
        """
        Grow the current tree for `num_iterations`. For each iteration,
        we pick a random node to start with, and random substructures
        to match and replace with. If this operation is successful,
        we add the nodes that are currently not in the tree via SMILES
        matching.

        Parameters
        ----------
        num_iterations : int
            Number of iterations to generate for
        """
        for _ in tqdm(range(num_iterations)):
            # pick a random node to grow
            node = self.random_node
            pattern, replace = self.random_substructure, self.random_substructure
            # here name actually refers to the Chem.Mol object
            try:
                molecules = replace_substructure(node.molecule, pattern, replace)
                if molecules:
                    self.add_nodes(molecules, node)
            except Exception as error:
                print(error)
