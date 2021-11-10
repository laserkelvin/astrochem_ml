
from typing import Union
from pathlib import Path

from joblib import dump, load

from astrochem_ml import smiles


default_model_path = Path(__file__).parents[0].absolute().joinpath("EmbeddingModel.pkl")


class EmbeddingModel(object):
    """
    A class that wraps both word2vec and scikit-learn pipelines.
    The main usage of this class is an abstract `vectorize` pipeline
    that will first generate molecule embeddings, followed by running
    it through any transformations with scikit-learn.
    """
    def __init__(self, w2vec_obj, transform=None, radius: int = 1) -> None:
        self._model = w2vec_obj
        self._transform = transform
        self._radius = radius
        self._covariance = None

    @property
    def model(self):
        return self._model

    @property
    def transform(self):
        return self._transform

    @property
    def radius(self):
        return self._radius

    def vectorize(self, smi: str):
        vector = smiles.misc.smi_to_vector(smi, self.model, self.radius)
        # 
        if self._transform is not None:
            # the clustering is always the last step, which we ignore
            for step in self.transform.steps[:len(self.transform.steps)-1]:
                vector = step[1].transform(vector)
        return vector[0]

    def __call__(self, smi: str):
        return self.vectorize(smi)

    @classmethod
    def from_pkl(cls, w2vec_path: str, transform_path: Union[str, None] = None, **kwargs):
        w2vec_obj = smiles.misc.load_model(w2vec_path)
        if transform_path:
            transform_obj = load(transform_path)
        else:
            transform_obj = None
        return cls(w2vec_obj, transform_obj, **kwargs)

    @classmethod
    def from_pretrained(cls):
        return load(default_model_path)

    def save(self, path: str):
        dump(self, path)
