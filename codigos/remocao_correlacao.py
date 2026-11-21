import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RemoverCorrelacionados(BaseEstimator, TransformerMixin):
    """Remove atributos muito correlacionados, usando só os dados de treino.

    Percorre os pares de atributos (i < j). Se |correlação de Pearson| > limiar,
    o atributo j (o segundo do par) é removido.
    """

    def __init__(self, limiar=0.95):
        self.limiar = limiar

    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(np.asarray(X), rowvar=False))
        superior = np.triu(corr, k=1)
        remover = (superior > self.limiar).any(axis=0)
        self.manter_ = np.where(~remover)[0]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.manter_]
