
from typing import Union, List
from pathlib import Path
from functools import cached_property

from joblib import dump, load
import selfies as sf
from tqdm.auto import tqdm


class Corpus(object):

    corpus_ext = {"smiles": "smi", "selfies": "sf"}

    def __init__(self, data: Union[str, Path], corpus_type: Union[None, str] = None, pad: Union[str, None] = "[nop]"):
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

    @property
    def smiles_to_selfies(self) -> List[str]:
        return [sf.encoder(smi) for smi in tqdm(self.data)]

    @property
    def data(self) -> List[str]:
        if hasattr(self, "_data"):
            return self._data
        else:
            with open(self.corpus_path) as read_file:
                data = list(map(lambda x: x.strip(), read_file.readlines()))
            self._data = data
            return data

    @cached_property
    def selfies(self) -> str:
        if self.corpus_type == "selfies":
            return self.data
        else:
            return [sf.encoder(smi) for smi in self.data]

    @cached_property
    def vocabulary(self):
        alphabet = sf.get_alphabet_from_selfies(self.selfies)
        if self.pad:
            alphabet.add(self.pad)
        return list(sorted(alphabet))

    @cached_property
    def max_length(self) -> int:
        return max(map(lambda x: sf.len_selfies(x), self.selfies))

    @property
    def token_mapping(self):
        return {token: index for index, token in enumerate(self.vocabulary)}

    def label_to_token(self, label: int) -> str:
        return self.vocabulary.index(label)

    def token_to_label(self, token: str) -> int:
        return self.token_mapping.get(token)