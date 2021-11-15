from typing import Union, List, Dict
from pathlib import Path
from functools import cached_property

from joblib import dump, load
import selfies as sf
from tqdm.auto import tqdm
from joblib import Parallel, delayed


class Corpus(object):

    corpus_ext = {"smiles": "smi", "selfies": "sf"}

    def __init__(
        self,
        data: Union[str, Path],
        corpus_type: Union[None, str] = None,
        pad: Union[str, None] = "[nop]",
        n_jobs: Union[int, None] = None,
    ):
        if isinstance(corpus_type, str):
            assert corpus_type.lower() in ["smiles", "selfies"]
        # infer from the file extension
        else:
            for key, value in self.corpus_ext.items():
                if Path(data).suffix in value:
                    corpus_type = key
        if not corpus_type:
            raise KeyError(f"Unable to determine corpus type!")
        self.corpus_path = Path(data)
        self.corpus_type = corpus_type
        self.pad = pad
        if n_jobs:
            self._parallel = Parallel(n_jobs=n_jobs)

    @property
    def smiles_to_selfies(self) -> List[str]:
        return [sf.encoder(smi) for smi in tqdm(self.data)]

    @property
    def data(self) -> List[str]:
        """
        The dataset of strings that are read in from the
        data file. This caches the result, as it can be
        intensive to constantly read from disk.

        Returns
        -------
        List[str]
            List of all strings in the dataset.
        """
        if hasattr(self, "_data"):
            return tqdm(self._data)
        else:
            with open(self.corpus_path) as read_file:
                data = list(map(lambda x: x.strip(), read_file.readlines()))
            self._data = data
            return tqdm(data)

    @cached_property
    def selfies(self) -> List[str]:
        """
        Returns the all SELFIES in this data set; if
        the data set is encoded as SMILES, it will
        automatically do the transcribing into SELFIES.

        Returns
        -------
        str
            List of SELFIES strings
        """
        if self.corpus_type == "selfies":
            return self.data
        else:
            if hasattr(self, "_parallel"):
                return self._parallel(delayed(sf.encoder)(smi) for smi in self.data)
            else:
                return [sf.encoder(smi) for smi in self.data]

    @cached_property
    def vocabulary(self) -> List[str]:
        """
        Returns the vocabulary of the current dataset,
        which corresponds to all possible SELFIES tokens
        that have been observed.

        Returns
        -------
        List[str]
            List of SELFIES tokens
        """
        alphabet = sf.get_alphabet_from_selfies(self.selfies)
        if self.pad:
            alphabet.add(self.pad)
        return list(sorted(alphabet))

    @cached_property
    def max_length(self) -> int:
        """
        Return the maximum length of SELFIES in the current
        data set.

        Returns
        -------
        int

        """
        if hasattr(self, "_parallel"):
            counts = self._parallel(
                delayed(sf.len_selfies)(string) for string in self.selfies
            )
        else:
            counts = map(lambda x: sf.len_selfies(x), self.selfies)
        return max(counts)

    @property
    def token_mapping(self) -> Dict[str, int]:
        """
        Return the token mapping into index labels.

        Returns
        -------
        Dict[str, int]
            A dictionary with keys corresponding to
            tokens, and values to indices
        """
        return {token: index for index, token in enumerate(self.vocabulary)}

    def label_to_token(self, label: int) -> str:
        """
        Retrieve the token corresponding to a index label.

        For example:

        ```python
        >>> corpus.label_to_token(3)
        [C]
        ```

        Parameters
        ----------
        label : int
            Index label to retrieve

        Returns
        -------
        str
            Corresponding SELFIES token
        """
        return self.vocabulary.index(label)

    def token_to_label(self, token: str) -> int:
        """
        Retrieve the index label corresponding to a token.

        For example:

        ```python
        >>> corpus.token_to_label("[C]")
        3
        ```

        Parameters
        ----------
        token : str
            SELFIES token to retrieve

        Returns
        -------
        int
            Index label of the token
        """
        return self.token_mapping.get(token)

    def __len__(self) -> int:
        return self.max_length
