import os
import time
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# 1. Configurar caminhos e garantir que o diretório de resultados exista
DATASET_PATH = "dataset/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx"
RESULTS_PATH = "resultados/"
os.makedirs(RESULTS_PATH, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(RESULTS_PATH, "resultados_H1.csv")

# 2. Carregar os dados
df = pd.read_excel(DATASET_PATH)
X = df.drop(columns=['Class'])
y = df['Class']

# 3. Configurar o protocolo de validação (5-fold estratificado, 3 repetições, random_state=42)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# 4. Definir classificadores e representações
classificadores = {
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC()
}

representacoes = {
    'dados brutos': 'passthrough',
    'MinMax': MinMaxScaler(),
    'padronizados': StandardScaler()
}

resultados = []

print("Iniciando experimentos da H1...")

# 5. Loop principal de avaliação
for nome_rep, scaler in representacoes.items():
    for nome_clf, clf in classificadores.items():
        print(f"Treinando: {nome_clf} com {nome_rep}...")
        
        pipeline = Pipeline([
            ('scaler', scaler), 
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
            fim = time.time()
            
            tempo_s = fim - inicio

            acuracia = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')

            resultados.append({
                'hipotese': 'H1',
                'representacao': nome_rep,
                'classificador': nome_clf,
                'repeticao': repeticao,
                'fold': fold,
                'acuracia': acuracia,
                'f1_macro': f1_macro,
                'tempo_s': tempo_s
            })

# 6. Gerar e salvar o CSV final no caminho configurado
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"\nExperimentos concluídos! Arquivo salvo em: {CSV_OUTPUT_PATH}")