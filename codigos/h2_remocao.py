"""
H2: Remover atributos redundantes (muito correlacionados) ajuda o Naive Bayes,
mas quase não muda o KNN e o SVM.

Representações avaliadas (todas com padronização antes, como na H3):
- sem remoção: os 16 atributos padronizados (mesma base da H1 "padronizados");
- remoção por correlação de Pearson com |r| > 0,99, 0,95 e 0,90
  (na base completa sobram 13, 9 e 7 atributos, respectivamente);
- variante "sequencial" da remoção em 0,95 e 0,90 (10 e 8 atributos), que compara
  cada atributo só com os já mantidos e evita remoções em cadeia (ver
  codigos/remocao_correlacao.py). Em 0,99 as duas regras mantêm os mesmos atributos.

Controle: remoção ALEATÓRIA do mesmo número de atributos (N_SORTEIOS conjuntos
sorteados para cada tamanho). Se o efeito vier da redundância, a remoção por
correlação deve ser melhor que a maioria dos sorteios; se vier apenas de
"ter menos atributos", ela deve ficar no meio dos sorteios.

Rodar a partir da raiz do repositório:
    python codigos/h2_remocao.py
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from remocao_correlacao import RemoverCorrelacionados


class SelecionarColunas(BaseEstimator, TransformerMixin):
    """Mantém um conjunto fixo de colunas (usado no controle aleatório)."""

    def __init__(self, colunas=None):
        self.colunas = colunas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.colunas]


# 1. Caminhos
DATASET_PATH = "dataset/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx"
RESULTS_PATH = os.path.join("resultados", "H2")
os.makedirs(RESULTS_PATH, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(RESULTS_PATH, "resultados_H2.csv")
CSV_ATRIBUTOS_PATH = os.path.join(RESULTS_PATH, "atributos_H2.csv")
CSV_CONTROLE_PATH = os.path.join(RESULTS_PATH, "controle_aleatorio_H2.csv")

# 2. Dados
df = pd.read_excel(DATASET_PATH)
X = df.drop(columns=['Class'])
y = df['Class']
nomes = np.array(X.columns)

# 3. Protocolo combinado: 5-fold estratificado, 3 repetições, random_state=42
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
divisoes = list(cv.split(X, y))

# 4. Classificadores (mesmos parâmetros padrão da H1 e da H3).
#    Com os dados padronizados, o var_smoothing do Naive Bayes não altera nada
#    (H1: 90,63% com e sem ele), então usamos o GaussianNB padrão.
classificadores = {
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC()
}

# 5. Representações: nome -> etapa de remoção
LIMIARES = [0.99, 0.95, 0.90]
representacoes = {'sem remocao': 'passthrough'}
for limiar in LIMIARES:
    representacoes[f'remocao r>{limiar:.2f}'] = RemoverCorrelacionados(limiar=limiar)
for limiar in [0.95, 0.90]:
    representacoes[f'remocao sequencial r>{limiar:.2f}'] = \
        RemoverCorrelacionados(limiar=limiar, sequencial=True)

N_SORTEIOS = 20


def avaliar(pipeline, registro, extra):
    """Roda o pipeline nas 15 divisões e acrescenta uma linha por fold em `registro`."""
    for i, (train_index, test_index) in enumerate(divisoes):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        inicio = time.time()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        tempo_s = time.time() - inicio

        registro.append({
            **extra,
            'repeticao': (i // 5) + 1,
            'fold': (i % 5) + 1,
            'acuracia': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'tempo_s': tempo_s
        })
        yield pipeline


# 6. Experimento principal
resultados = []
mantidos_por_fold = {}

print("Iniciando experimentos da H2...")
for nome_rep, remocao in representacoes.items():
    for nome_clf, clf in classificadores.items():
        print(f"Treinando: {nome_clf} com {nome_rep}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('remocao', remocao),
            ('clf', clf)
        ])
        extra = {'hipotese': 'H2', 'representacao': nome_rep, 'classificador': nome_clf}
        for pipe in avaliar(pipeline, resultados, extra):
            # Quais atributos a remoção manteve neste fold (pode variar entre folds)
            if isinstance(remocao, RemoverCorrelacionados) and nome_clf == 'KNN':
                manter = pipe.named_steps['remocao'].manter_
                mantidos_por_fold.setdefault(nome_rep, set()).add(tuple(nomes[manter]))

colunas = ['hipotese', 'representacao', 'classificador', 'repeticao', 'fold',
           'acuracia', 'f1_macro', 'tempo_s']
pd.DataFrame(resultados)[colunas].to_csv(CSV_OUTPUT_PATH, index=False)

# 7. Atributos mantidos e removidos em cada limiar
linhas = []
for nome_rep, conjuntos in mantidos_por_fold.items():
    if len(conjuntos) > 1:
        print(f"Aviso: {nome_rep} manteve conjuntos diferentes entre os folds: {conjuntos}")
    manter = sorted(conjuntos, key=len)[0]
    linhas.append({
        'representacao': nome_rep,
        'n_atributos': len(manter),
        'conjuntos_diferentes_entre_folds': len(conjuntos),
        'atributos_mantidos': ';'.join(manter),
        'atributos_removidos': ';'.join(c for c in nomes if c not in manter)
    })
atributos = pd.DataFrame(linhas)
atributos.to_csv(CSV_ATRIBUTOS_PATH, index=False)
print(atributos[['representacao', 'n_atributos', 'conjuntos_diferentes_entre_folds']].to_string(index=False))

# 8. Controle: remoção aleatória do mesmo número de atributos.
#    Os mesmos sorteios são usados para os três classificadores.
rng = np.random.default_rng(42)
sorteios = {}
for k in atributos['n_atributos'].drop_duplicates():
    escolhidos = set()
    while len(escolhidos) < N_SORTEIOS:
        escolhidos.add(tuple(sorted(rng.choice(len(nomes), size=k, replace=False))))
    sorteios[k] = sorted(escolhidos)

controle = []
for nome_clf, clf in classificadores.items():
    inicio_clf = time.time()
    for k, lista in sorteios.items():
        print(f"Controle aleatório: {nome_clf}, {k} atributos ({N_SORTEIOS} sorteios)...")
        for s, cols in enumerate(lista, start=1):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('remocao', SelecionarColunas(colunas=list(cols))),
                ('clf', clf)
            ])
            extra = {'classificador': nome_clf, 'n_atributos': k, 'sorteio': s,
                     'atributos_mantidos': ';'.join(nomes[list(cols)])}
            for _ in avaliar(pipeline, controle, extra):
                pass
    print(f"  ({time.time() - inicio_clf:.0f}s)")

pd.DataFrame(controle).to_csv(CSV_CONTROLE_PATH, index=False)

print(f"\nExperimentos concluídos! Arquivos salvos em '{RESULTS_PATH}':")
print(f"- {CSV_OUTPUT_PATH}")
print(f"- {CSV_ATRIBUTOS_PATH}")
print(f"- {CSV_CONTROLE_PATH}")
