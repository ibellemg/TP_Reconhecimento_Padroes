import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# 1. Configurar caminhos
DATASET_PATH = "dataset/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx"
RESULTS_PATH = "resultados/"
os.makedirs(RESULTS_PATH, exist_ok=True)

CSV_INPUT_PATH = os.path.join(RESULTS_PATH, "resultados_H1.csv")

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# 2. Carregar os resultados do CSV
df_res = pd.read_csv(CSV_INPUT_PATH)

ordem_rep = ['dados brutos', 'MinMax', 'padronizados']
ordem_clf = ['KNN', 'SVM', 'Naive Bayes']
df_res['representacao'] = pd.Categorical(df_res['representacao'], categories=ordem_rep, ordered=True)
df_res['classificador'] = pd.Categorical(df_res['classificador'], categories=ordem_clf, ordered=True)

# ==============================================================================
# 3. TABELAS DE RESUMO (Média ± Desvio Padrão e Ganho)
# ==============================================================================
resumo = df_res.groupby(['classificador', 'representacao'], observed=False).agg(
    f1_media=('f1_macro', 'mean'),
    f1_std=('f1_macro', 'std'),
    acc_media=('acuracia', 'mean'),
    acc_std=('acuracia', 'std'),
    tempo_medio=('tempo_s', 'mean'),
    tempo_std=('tempo_s', 'std')
).reset_index()

resumo['F1-Macro'] = resumo.apply(lambda r: f"{r['f1_media']*100:.2f}% ± {r['f1_std']*100:.2f}%", axis=1)
resumo['Acurácia'] = resumo.apply(lambda r: f"{r['acc_media']*100:.2f}% ± {r['acc_std']*100:.2f}%", axis=1)
resumo['Tempo (s)'] = resumo.apply(lambda r: f"{r['tempo_medio']:.3f}s ± {r['tempo_std']:.3f}s", axis=1)

tabela_formatada = resumo[['classificador', 'representacao', 'F1-Macro', 'Acurácia', 'Tempo (s)']]
print("\n=== TABELA RESUMO H1 (Média ± Desvio Padrão das 15 execuções) ===")
print(tabela_formatada.to_string(index=False))

pivot_f1 = resumo.pivot(index='classificador', columns='representacao', values='f1_media') * 100
pivot_f1['Ganho MinMax (p.p.)'] = pivot_f1['MinMax'] - pivot_f1['dados brutos']
pivot_f1['Ganho Padronizado (p.p.)'] = pivot_f1['padronizados'] - pivot_f1['dados brutos']

print("\n=== IMPACTO NO F1-MACRO (%) E GANHO EM PONTOS PERCENTUAIS ===")
print(pivot_f1.round(2))

# Salvar tabelas na pasta resultados/
caminho_tab_resumo = os.path.join(RESULTS_PATH, "tabela_resumo_H1.csv")
caminho_tab_ganho = os.path.join(RESULTS_PATH, "tabela_ganho_f1_H1.csv")
tabela_formatada.to_csv(caminho_tab_resumo, index=False)
pivot_f1.round(2).to_csv(caminho_tab_ganho)

# ==============================================================================
# 4. GRÁFICO 1: Barras Agrupadas (F1-Macro e Acurácia)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

sns.barplot(
    data=df_res, x='classificador', y='f1_macro', hue='representacao',
    ax=axes[0], capsize=.05, err_kws={'linewidth': 1.2}
)
axes[0].set_title('Impacto da Escala no F1-Macro (Principal)')
axes[0].set_xlabel('Classificador')
axes[0].set_ylabel('Pontuação (0 a 1)')
axes[0].set_ylim(0.5, 1.0)
axes[0].legend(title='Representação', loc='lower right')

sns.barplot(
    data=df_res, x='classificador', y='acuracia', hue='representacao',
    ax=axes[1], capsize=.05, err_kws={'linewidth': 1.2}
)
axes[1].set_title('Impacto da Escala na Acurácia')
axes[1].set_xlabel('Classificador')
axes[1].set_ylabel('')
axes[1].legend(title='Representação', loc='lower right')

plt.tight_layout()
caminho_fig1 = os.path.join(RESULTS_PATH, "h1_comparacao_metricas.png")
plt.savefig(caminho_fig1)
plt.close()

# ==============================================================================
# 5. GRÁFICO 2: Boxplot de Estabilidade e Tempo de Execução
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.boxplot(
    data=df_res, x='classificador', y='f1_macro', hue='representacao', ax=axes[0]
)
axes[0].set_title('Distribuição e Estabilidade do F1-Macro (15 execuções)')
axes[0].set_xlabel('Classificador')
axes[0].set_ylabel('F1-Macro')

sns.barplot(
    data=df_res, x='classificador', y='tempo_s', hue='representacao',
    ax=axes[1], capsize=.05, err_kws={'linewidth': 1.2}
)
axes[1].set_title('Tempo Médio de Execução (Treino + Predição)')
axes[1].set_xlabel('Classificador')
axes[1].set_ylabel('Tempo (segundos - escala log)')
axes[1].set_yscale('log')

plt.tight_layout()
caminho_fig2 = os.path.join(RESULTS_PATH, "h1_boxplot_e_tempo.png")
plt.savefig(caminho_fig2)
plt.close()

# ==============================================================================
# 6. GRÁFICO 3: Matrizes de Confusão (Bruto vs Padronizado)
# ==============================================================================
print("\nGerando Matrizes de Confusão agregadas (5-Fold)...")
df_beans = pd.read_excel(DATASET_PATH)
X = df_beans.drop(columns=['Class'])
y = df_beans['Class']
classes = sorted(y.unique())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB()
}

cenarios_cm = {
    'Dados Brutos': 'passthrough',
    'Padronizados': StandardScaler()
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for row_idx, (nome_cenario, scaler) in enumerate(cenarios_cm.items()):
    for col_idx, (nome_clf, clf) in enumerate(modelos.items()):
        cm_total = np.zeros((len(classes), len(classes)), dtype=int)
        pipe = Pipeline([('scaler', scaler), ('clf', clf)])
        
        for train_idx, test_idx in skf.split(X, y):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = pipe.predict(X.iloc[test_idx])
            cm_total += confusion_matrix(y.iloc[test_idx], preds, labels=classes)
        
        cm_norm = cm_total.astype('float') / cm_total.sum(axis=1)[:, np.newaxis] * 100
        
        ax = axes[row_idx, col_idx]
        sns.heatmap(
            cm_norm, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=classes, yticklabels=classes, cbar=False, ax=ax
        )
        ax.set_title(f"{nome_clf} — {nome_cenario} (%)")
        ax.set_xlabel('Predito' if row_idx == 1 else '')
        ax.set_ylabel('Real' if col_idx == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
caminho_fig3 = os.path.join(RESULTS_PATH, "h1_matrizes_confusao.png")
plt.savefig(caminho_fig3)
plt.close()

print(f"\nAnálise concluída! Todos os arquivos foram salvos em '{RESULTS_PATH}':")
print(f"- {caminho_tab_resumo}")
print(f"- {caminho_tab_ganho}")
print(f"- {caminho_fig1}")
print(f"- {caminho_fig2}")
print(f"- {caminho_fig3}")