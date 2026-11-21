"""
H3: Com o mesmo número de dimensões, o PCA mantém um desempenho melhor
do que a remoção de atributos correlacionados.

Representações avaliadas (todas com padronização antes):
- PCA com k = 2, 4, 6, 7, 8, 9, 10, 13 e 16 componentes;
- remoção de atributos correlacionados com limiares 0,99, 0,95 e 0,90
  (na base completa sobram 13, 9 e 7 atributos, respectivamente).
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from remocao_correlacao import RemoverCorrelacionados


# 1. Caminhos
DATASET_PATH = "dataset/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx"
RESULTS_PATH = os.path.join("resultados", "H3")
os.makedirs(RESULTS_PATH, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(RESULTS_PATH, "resultados_H3.csv")
CSV_ATRIBUTOS_PATH = os.path.join(RESULTS_PATH, "atributos_por_fold_H3.csv")
CSV_VARIANCIA_PATH = os.path.join(RESULTS_PATH, "variancia_pca_H3.csv")

# 2. Dados
df = pd.read_excel(DATASET_PATH)
X = df.drop(columns=['Class'])
y = df['Class']

# 3. Protocolo combinado: 5-fold estratificado, 3 repetições, random_state=42
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# 4. Classificadores (mesmos parâmetros padrão da H1)
classificadores = {
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC()
}

# 5. Representações: nome -> etapa de redução de dimensão
representacoes = {}
for k in [2, 4, 6, 7, 8, 9, 10, 13, 16]:
    representacoes[f'PCA k={k}'] = PCA(n_components=k, random_state=42)
for limiar in [0.99, 0.95, 0.90]:
    representacoes[f'remocao r>{limiar:.2f}'] = RemoverCorrelacionados(limiar=limiar)

resultados = []
atributos_por_fold = []

print("Iniciando experimentos da H3...")

# 6. Loop principal
for nome_rep, reducao in representacoes.items():
    for nome_clf, clf in classificadores.items():
        print(f"Treinando: {nome_clf} com {nome_rep}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('reducao', reducao),
            ('clf', clf)
        ])

        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            repeticao = (i // 5) + 1
            fold = (i % 5) + 1

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            inicio = time.time()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            tempo_s = time.time() - inicio

            resultados.append({
                'hipotese': 'H3',
                'representacao': nome_rep,
                'classificador': nome_clf,
                'repeticao': repeticao,
                'fold': fold,
                'acuracia': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'tempo_s': tempo_s
            })

            # Quantos atributos a remoção manteve neste fold (pode variar entre folds)
            if nome_rep.startswith('remocao') and nome_clf == 'KNN':
                manter = pipeline.named_steps['reducao'].manter_
                atributos_por_fold.append({
                    'representacao': nome_rep,
                    'repeticao': repeticao,
                    'fold': fold,
                    'n_atributos': len(manter),
                    'atributos_mantidos': ';'.join(X.columns[manter])
                })

# 7. Salvar resultados (colunas combinadas pelo grupo)
pd.DataFrame(resultados).to_csv(CSV_OUTPUT_PATH, index=False)
pd.DataFrame(atributos_por_fold).to_csv(CSV_ATRIBUTOS_PATH, index=False)

# 8. Variância explicada acumulada do PCA (descrição da base, usa todos os dados)
pca_total = PCA().fit(StandardScaler().fit_transform(X))
variancia = pd.DataFrame({
    'k': np.arange(1, X.shape[1] + 1),
    'variancia_explicada': pca_total.explained_variance_ratio_,
    'variancia_acumulada': pca_total.explained_variance_ratio_.cumsum()
})
variancia.to_csv(CSV_VARIANCIA_PATH, index=False)

print(f"\nExperimentos concluídos! Arquivos salvos em '{RESULTS_PATH}':")
print(f"- {CSV_OUTPUT_PATH}")
print(f"- {CSV_ATRIBUTOS_PATH}")
print(f"- {CSV_VARIANCIA_PATH}")
