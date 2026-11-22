import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RemoverCorrelacionados(BaseEstimator, TransformerMixin):
    """Remove atributos muito correlacionados, usando só os dados de treino.

    Padrão (sequencial=False, usado na H2 e na H3): percorre os pares de atributos
    (i < j). Se |correlação de Pearson| > limiar, o atributo j (o segundo do par) é
    removido, mesmo que o atributo i também tenha sido removido. Isso pode causar
    remoções "em cadeia": j sai por ser parecido com i, embora i também saia e j não
    seja parecido com nenhum atributo que ficou.

    sequencial=True (variante usada só na H2): percorre os atributos na ordem e mantém
    cada um se a |correlação| com todos os atributos JÁ MANTIDOS for <= limiar.
    Evita a remoção em cadeia.
    """

    def __init__(self, limiar=0.95, sequencial=False):
        self.limiar = limiar
        self.sequencial = sequencial

    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(np.asarray(X), rowvar=False))
        if self.sequencial:
            manter = []
            for j in range(corr.shape[0]):
                if all(corr[i, j] <= self.limiar for i in manter):
                    manter.append(j)
            self.manter_ = np.array(manter)
        else:
            superior = np.triu(corr, k=1)
            remover = (superior > self.limiar).any(axis=0)
            self.manter_ = np.where(~remover)[0]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.manter_]
