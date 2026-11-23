"""
Análise da H3: PCA × remoção de atributos correlacionados com o mesmo número de dimensões.

Lê resultados/H3/resultados_H3.csv (gerado por codigos/h3_pca.py) e gera tabelas e gráficos.
Rodar a partir da raiz do repositório:
    python codigos/analise_h3.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, t as dist_t
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from remocao_correlacao import RemoverCorrelacionados

# 1. Caminhos
DATASET_PATH = "dataset/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx"
RESULTS_PATH = os.path.join("resultados", "H3")
H1_CSV_PATH = os.path.join("resultados", "H1", "resultados_H1.csv")

df_res = pd.read_csv(os.path.join(RESULTS_PATH, "resultados_H3.csv"))
df_atrib = pd.read_csv(os.path.join(RESULTS_PATH, "atributos_por_fold_H3.csv"))
df_var = pd.read_csv(os.path.join(RESULTS_PATH, "variancia_pca_H3.csv"))

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

ordem_clf = ['KNN', 'Naive Bayes', 'SVM']

# 2. Número de dimensões de cada representação
#    PCA: o próprio k. Remoção: número de atributos mais frequente entre os folds.
n_remocao = df_atrib.groupby('representacao')['n_atributos'].agg(lambda s: s.mode().iloc[0])
variacao = df_atrib.groupby('representacao')['n_atributos'].agg(['min', 'max'])
print("\n=== ATRIBUTOS MANTIDOS PELA REMOÇÃO (mín. e máx. entre os 15 folds) ===")
print(variacao.to_string())


def dimensoes(rep):
    if rep.startswith('PCA'):
        return int(rep.split('=')[1])
    return int(n_remocao[rep])


df_res['tecnica'] = np.where(df_res['representacao'].str.startswith('PCA'), 'PCA', 'Remoção')
df_res['dimensoes'] = df_res['representacao'].map(dimensoes)

# ==============================================================================
# 3. TABELA RESUMO (média ± desvio das 15 execuções)
# ==============================================================================
resumo = df_res.groupby(['classificador', 'tecnica', 'representacao', 'dimensoes']).agg(
    f1_media=('f1_macro', 'mean'), f1_std=('f1_macro', 'std'),
    acc_media=('acuracia', 'mean'), acc_std=('acuracia', 'std'),
    tempo_medio=('tempo_s', 'mean'), tempo_std=('tempo_s', 'std')
).reset_index().sort_values(['classificador', 'tecnica', 'dimensoes'])

resumo['F1-Macro'] = resumo.apply(lambda r: f"{r['f1_media']*100:.2f}% ± {r['f1_std']*100:.2f}%", axis=1)
resumo['Acurácia'] = resumo.apply(lambda r: f"{r['acc_media']*100:.2f}% ± {r['acc_std']*100:.2f}%", axis=1)
resumo['Tempo (s)'] = resumo.apply(lambda r: f"{r['tempo_medio']:.3f}s ± {r['tempo_std']:.3f}s", axis=1)

tabela_resumo = resumo[['classificador', 'representacao', 'dimensoes', 'F1-Macro', 'Acurácia', 'Tempo (s)']]
print("\n=== TABELA RESUMO H3 (Média ± Desvio Padrão das 15 execuções) ===")
print(tabela_resumo.to_string(index=False))
tabela_resumo.to_csv(os.path.join(RESULTS_PATH, "tabela_resumo_H3.csv"), index=False)

# ==============================================================================
# 4. TABELA PRINCIPAL: PCA × remoção com o mesmo número de dimensões
#    Comparação pareada fold a fold (mesmas divisões, random_state=42).
#    Teste principal: t corrigido de Nadeau e Bengio (2003), o mesmo da H2. Ele
#    considera que os folds não são independentes (os treinos se sobrepõem).
#    O Wilcoxon fica só como referência (ignora essa dependência, p otimista).
# ==============================================================================
N_TESTE_SOBRE_TREINO = 1 / 4  # 5 folds: teste = 1/5 da base, treino = 4/5


def t_corrigido(dif):
    n = len(dif)
    var_corrigida = (1 / n + N_TESTE_SOBRE_TREINO) * dif.var(ddof=1)
    if var_corrigida == 0:
        return np.nan, np.nan
    t = dif.mean() / np.sqrt(var_corrigida)
    return t, 2 * dist_t.sf(abs(t), n - 1)


linhas = []
for rep_rem, k in n_remocao.items():
    rep_pca = f'PCA k={k}'
    if rep_pca not in set(df_res['representacao']):
        print(f"Aviso: {rep_rem} mantém {k} atributos, mas não há {rep_pca} nos resultados.")
        continue
    for clf in ordem_clf:
        chave = ['repeticao', 'fold']
        a = df_res[(df_res['representacao'] == rep_pca) & (df_res['classificador'] == clf)].set_index(chave)
        b = df_res[(df_res['representacao'] == rep_rem) & (df_res['classificador'] == clf)].set_index(chave)
        dif = (a['f1_macro'] - b['f1_macro']).dropna() * 100
        try:
            p = wilcoxon(dif).pvalue
        except ValueError:
            p = np.nan  # todas as diferenças iguais a zero
        t_c, p_c = t_corrigido(dif)
        linhas.append({
            'classificador': clf,
            'dimensoes': k,
            'remocao': rep_rem,
            'F1 PCA (%)': a['f1_macro'].mean() * 100,
            'F1 remoção (%)': b['f1_macro'].mean() * 100,
            'diferença PCA - remoção (p.p.)': dif.mean(),
            'desvio da diferença (p.p.)': dif.std(),
            'folds com PCA melhor': f"{(dif > 0).sum()}/{len(dif)}",
            't corrigido': t_c,
            'p t corrigido': p_c,
            'p Wilcoxon': p
        })

tabela_comp = pd.DataFrame(linhas).sort_values(['classificador', 'dimensoes'])
print("\n=== PCA × REMOÇÃO COM O MESMO NÚMERO DE DIMENSÕES (F1-macro) ===")
print(tabela_comp.round(4).to_string(index=False))
tabela_comp.round(4).to_csv(os.path.join(RESULTS_PATH, "tabela_pca_vs_remocao_H3.csv"), index=False)

# ==============================================================================
# 5. GRÁFICO 1: F1-macro × número de dimensões, um painel por classificador
# ==============================================================================
base_h1 = None
if os.path.exists(H1_CSV_PATH):
    h1 = pd.read_csv(H1_CSV_PATH)
    base_h1 = h1[h1['representacao'] == 'padronizados'].groupby('classificador')['f1_macro'].mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, clf in zip(axes, ordem_clf):
    r = resumo[resumo['classificador'] == clf]
    pca = r[r['tecnica'] == 'PCA']
    rem = r[r['tecnica'] == 'Remoção']
    ax.errorbar(pca['dimensoes'], pca['f1_media'], yerr=pca['f1_std'],
                marker='o', capsize=3, label='PCA')
    ax.errorbar(rem['dimensoes'], rem['f1_media'], yerr=rem['f1_std'],
                marker='s', linestyle='none', markersize=8, capsize=3, label='Remoção por correlação')
    for _, linha in rem.iterrows():
        ax.annotate(linha['representacao'].replace('remocao ', ''), (linha['dimensoes'], linha['f1_media']),
                    textcoords='offset points', xytext=(0, -16), ha='center', fontsize=8)
    if base_h1 is not None and clf in base_h1:
        ax.axhline(base_h1[clf], color='gray', linestyle='--', linewidth=1,
                   label='Padronizado, 16 atributos (H1)')
    ax.set_title(clf)
    ax.set_xlabel('Número de dimensões')
    ax.set_xticks(sorted(r['dimensoes'].unique()))
axes[0].set_ylabel('F1-macro (média ± desvio)')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.04))
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(RESULTS_PATH, "h3_f1_por_dimensao.png"), bbox_inches='tight')
plt.close()

# ==============================================================================
# 6. GRÁFICO 2: Variância explicada acumulada do PCA
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df_var['k'], df_var['variancia_acumulada'] * 100, marker='o')
for k in sorted(n_remocao.unique()):
    ax.axvline(k, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Número de componentes (k)')
ax.set_ylabel('Variância explicada acumulada (%)')
ax.set_title('PCA após padronização (linhas: dimensões da remoção)')
ax.set_xticks(df_var['k'])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "h3_variancia_pca.png"), bbox_inches='tight')
plt.close()

# ==============================================================================
# 7. GRÁFICO 3: Matrizes de confusão no menor número de dimensões comparado
#    (PCA × remoção mais agressiva), 5 folds agregados, como na H1
# ==============================================================================
print("\nGerando Matrizes de Confusão agregadas (5-Fold)...")
df_beans = pd.read_excel(DATASET_PATH)
X = df_beans.drop(columns=['Class'])
y = df_beans['Class']
classes = sorted(y.unique())
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rep_rem_menor = n_remocao.idxmin()
k_menor = int(n_remocao.min())
limiar_menor = float(rep_rem_menor.split('>')[1])

cenarios = {
    f'PCA k={k_menor}': PCA(n_components=k_menor, random_state=42),
    f'Remoção r>{limiar_menor:.2f} ({k_menor} atrib.)': RemoverCorrelacionados(limiar=limiar_menor)
}
modelos = {'KNN': KNeighborsClassifier(), 'Naive Bayes': GaussianNB(), 'SVM': SVC()}

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for i, (nome_cen, reducao) in enumerate(cenarios.items()):
    for j, (nome_clf, clf) in enumerate(modelos.items()):
        pipe = Pipeline([('scaler', StandardScaler()), ('reducao', reducao), ('clf', clf)])
        cm_total = np.zeros((len(classes), len(classes)), dtype=int)
        for tr, te in skf.split(X, y):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            cm_total += confusion_matrix(y.iloc[te], pipe.predict(X.iloc[te]), labels=classes)
        cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True) * 100

        ax = axes[i, j]
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)
        ax.set_title(f"{nome_clf}\n{nome_cen} (%)")
        ax.set_xlabel('Predito' if i == 1 else '')
        ax.set_ylabel('Real' if j == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "h3_matrizes_confusao.png"), bbox_inches='tight')
plt.close()

print(f"\nAnálise concluída! Arquivos salvos em '{RESULTS_PATH}':")
for nome in ["tabela_resumo_H3.csv", "tabela_pca_vs_remocao_H3.csv", "h3_f1_por_dimensao.png",
             "h3_variancia_pca.png", "h3_matrizes_confusao.png"]:
    print(f"- {os.path.join(RESULTS_PATH, nome)}")
