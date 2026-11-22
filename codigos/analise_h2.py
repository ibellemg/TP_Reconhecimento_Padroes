"""
Análise da H2: remoção de atributos redundantes (correlação) × 16 atributos padronizados.

Lê os CSVs de resultados/H2/ (gerados por codigos/h2_remocao.py) e gera tabelas e gráficos.
Rodar a partir da raiz do repositório:
    python codigos/analise_h2.py

Critério de veredito (por classificador e limiar), comparando cada remoção com a base
"sem remoção" fold a fold (mesmas divisões, random_state=42):
- "melhora"/"piora": t pareado corrigido de Nadeau e Bengio (2003) com p < 0,05.
  A correção existe porque os 15 folds não são independentes (os conjuntos de treino
  se sobrepõem), o que deixa o t pareado comum e o Wilcoxon otimistas demais.
- "não muda": p >= 0,05.
- Coluna "maior que DP entre folds": |diferença média| maior que o desvio padrão
  da base entre os folds (regra de magnitude combinada pelo grupo).

As representações "remocao sequencial" são a variante sem remoção em cadeia
(ver codigos/remocao_correlacao.py); servem para separar o efeito de remover
redundância do efeito de a regra descartar, por tabela, um atributo não redundante.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wilcoxon, t as dist_t
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from remocao_correlacao import RemoverCorrelacionados

# 1. Caminhos
DATASET_PATH = "dataset/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx"
RESULTS_PATH = os.path.join("resultados", "H2")

df_res = pd.read_csv(os.path.join(RESULTS_PATH, "resultados_H2.csv"))
df_ctrl = pd.read_csv(os.path.join(RESULTS_PATH, "controle_aleatorio_H2.csv"))
df_atrib = pd.read_csv(os.path.join(RESULTS_PATH, "atributos_H2.csv"))

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
AZUL = sns.color_palette("muted")[0]
LARANJA = sns.color_palette("muted")[1]
CINZA = '#9e9e9e'

ordem_clf = ['KNN', 'Naive Bayes', 'SVM']
BASE = 'sem remocao'
reps_regra = ['remocao r>0.99', 'remocao r>0.95', 'remocao r>0.90']
reps_remocao = ['remocao r>0.99', 'remocao r>0.95', 'remocao sequencial r>0.95',
                'remocao r>0.90', 'remocao sequencial r>0.90']
n_atrib = dict(zip(df_atrib['representacao'], df_atrib['n_atributos']))
_l = df_atrib.iloc[0]
n_atrib[BASE] = len(_l['atributos_mantidos'].split(';')) + len(_l['atributos_removidos'].split(';'))
CHAVE = ['repeticao', 'fold']
RAZAO_TESTE_TREINO = 1 / 4  # 5 folds: 20% teste, 80% treino


def rotulo(rep):
    if rep == BASE:
        return 'sem remoção\n16 atrib.'
    seq = ' seq.' if 'sequencial' in rep else ''
    return f"|r|>{rep.split('>')[1].replace('.', ',')}{seq}\n{n_atrib[rep]} atrib."


def t_corrigido(d):
    """t pareado corrigido (Nadeau e Bengio, 2003) para validação cruzada repetida."""
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / n + RAZAO_TESTE_TREINO) * var)
    return t, 2 * dist_t.sf(abs(t), n - 1)


def serie(df, rep, clf, col='f1_macro'):
    return df[(df['representacao'] == rep) & (df['classificador'] == clf)].set_index(CHAVE)[col]


print("\n=== ATRIBUTOS MANTIDOS EM CADA LIMIAR ===")
print(df_atrib[['representacao', 'n_atributos', 'atributos_removidos']].to_string(index=False))

# ==============================================================================
# 2. TABELA RESUMO (média ± desvio das 15 execuções)
# ==============================================================================
ordem_rep = [BASE] + reps_remocao
df_res['representacao'] = pd.Categorical(df_res['representacao'], categories=ordem_rep, ordered=True)
resumo = df_res.groupby(['classificador', 'representacao'], observed=True).agg(
    f1_media=('f1_macro', 'mean'), f1_std=('f1_macro', 'std'),
    acc_media=('acuracia', 'mean'), acc_std=('acuracia', 'std'),
    tempo_medio=('tempo_s', 'mean'), tempo_std=('tempo_s', 'std')
).reset_index()
resumo['n_atributos'] = resumo['representacao'].map(n_atrib)
resumo['F1-Macro'] = resumo.apply(lambda r: f"{r['f1_media']*100:.2f}% ± {r['f1_std']*100:.2f}%", axis=1)
resumo['Acurácia'] = resumo.apply(lambda r: f"{r['acc_media']*100:.2f}% ± {r['acc_std']*100:.2f}%", axis=1)
resumo['Tempo (s)'] = resumo.apply(lambda r: f"{r['tempo_medio']:.3f}s ± {r['tempo_std']:.3f}s", axis=1)
tabela_resumo = resumo[['classificador', 'representacao', 'n_atributos', 'F1-Macro', 'Acurácia', 'Tempo (s)']]
print("\n=== TABELA RESUMO H2 (Média ± Desvio Padrão das 15 execuções) ===")
print(tabela_resumo.to_string(index=False))
tabela_resumo.to_csv(os.path.join(RESULTS_PATH, "tabela_resumo_H2.csv"), index=False)
df_res['representacao'] = df_res['representacao'].astype(str)

# ==============================================================================
# 3. TABELA PRINCIPAL: efeito de cada remoção em relação à base (pareado por fold)
# ==============================================================================
linhas = []
for clf in ordem_clf:
    base = serie(df_res, BASE, clf)
    for rep in reps_remocao:
        dif = ((serie(df_res, rep, clf) - base) * 100).dropna()
        t, p_t = t_corrigido(dif)
        try:
            p_w = wilcoxon(dif).pvalue
        except ValueError:
            p_w = np.nan  # todas as diferenças iguais a zero
        if p_t < 0.05:
            veredito = 'melhora' if dif.mean() > 0 else 'piora'
        else:
            veredito = 'não muda'
        linhas.append({
            'classificador': clf,
            'remocao': rep,
            'n_atributos': n_atrib[rep],
            'F1 base (%)': base.mean() * 100,
            'DP base entre folds (p.p.)': base.std() * 100,
            'F1 remoção (%)': base.mean() * 100 + dif.mean(),
            'diferença (p.p.)': dif.mean(),
            'DP da diferença (p.p.)': dif.std(),
            'folds com remoção melhor': f"{(dif > 0).sum()}/{len(dif)}",
            't corrigido': t,
            'p t corrigido': p_t,
            'p Wilcoxon': p_w,
            'veredito': veredito,
            'maior que DP entre folds': abs(dif.mean()) > base.std() * 100,
        })
tabela_efeito = pd.DataFrame(linhas)
print("\n=== EFEITO DA REMOÇÃO EM RELAÇÃO AOS 16 ATRIBUTOS PADRONIZADOS (F1-macro) ===")
print(tabela_efeito.round(4).to_string(index=False))
tabela_efeito.round(4).to_csv(os.path.join(RESULTS_PATH, "tabela_efeito_H2.csv"), index=False)

# ==============================================================================
# 4. CONTROLE: remoção por correlação × remoção aleatória do mesmo número de atributos
# ==============================================================================
# Diferença média (p.p.) de cada sorteio em relação à base, pareada por fold
base_por_clf = {clf: serie(df_res, BASE, clf) for clf in ordem_clf}
ctrl = df_ctrl.copy()
ctrl['base'] = [base_por_clf[c].loc[(r, f)] for c, r, f in
                zip(ctrl['classificador'], ctrl['repeticao'], ctrl['fold'])]
ctrl['dif'] = (ctrl['f1_macro'] - ctrl['base']) * 100
dif_sorteio = ctrl.groupby(['classificador', 'n_atributos', 'sorteio'])['dif'].mean().reset_index()

linhas = []
for clf in ordem_clf:
    for rep in reps_remocao:
        k = n_atrib[rep]
        d_corr = tabela_efeito[(tabela_efeito['classificador'] == clf) &
                               (tabela_efeito['remocao'] == rep)]['diferença (p.p.)'].iloc[0]
        d_rand = dif_sorteio[(dif_sorteio['classificador'] == clf) &
                             (dif_sorteio['n_atributos'] == k)]['dif']
        linhas.append({
            'classificador': clf,
            'n_atributos': k,
            'remocao por correlacao (p.p.)': d_corr,
            'aleatoria media (p.p.)': d_rand.mean(),
            'aleatoria mediana (p.p.)': d_rand.median(),
            'aleatoria min (p.p.)': d_rand.min(),
            'aleatoria max (p.p.)': d_rand.max(),
            'sorteios': len(d_rand),
            '% sorteios piores que a correlacao': (d_rand < d_corr).mean() * 100,
        })
tabela_controle = pd.DataFrame(linhas)
print("\n=== CONTROLE: CORRELAÇÃO × REMOÇÃO ALEATÓRIA (diferença em relação à base, p.p.) ===")
print(tabela_controle.round(2).to_string(index=False))
tabela_controle.round(4).to_csv(os.path.join(RESULTS_PATH, "tabela_controle_H2.csv"), index=False)

# ==============================================================================
# 5. GRÁFICO 1: diferença no F1-macro por limiar, um painel por classificador,
#    com a distribuição dos sorteios aleatórios ao fundo
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
x = np.arange(len(reps_remocao))
for ax, clf in zip(axes, ordem_clf):
    te = tabela_efeito[tabela_efeito['classificador'] == clf].set_index('remocao').loc[reps_remocao]
    dp_base = te['DP base entre folds (p.p.)'].iloc[0]
    ax.axhspan(-dp_base, dp_base, color=CINZA, alpha=0.15, lw=0,
               label='±1 DP da base entre folds')
    ax.axhline(0, color='black', lw=0.8)

    dados = [dif_sorteio[(dif_sorteio['classificador'] == clf) &
                         (dif_sorteio['n_atributos'] == n_atrib[r])]['dif'].values for r in reps_remocao]
    bp = ax.boxplot(dados, positions=x, widths=0.3, whis=(5, 95), showfliers=False,
                    patch_artist=True, medianprops={'color': '#555555', 'lw': 1.2},
                    whiskerprops={'color': CINZA}, capprops={'color': CINZA},
                    boxprops={'facecolor': '#e6e6e6', 'edgecolor': CINZA})
    bp['boxes'][0].set_label(f'Remoção aleatória, mesmo nº de atributos '
                             f'({df_ctrl["sorteio"].nunique()} sorteios)')

    for seq, cor, marcador, nome in [(False, AZUL, 'o', 'Remoção por correlação (regra da H2/H3)'),
                                     (True, LARANJA, 'D', 'Remoção sequencial (sem cadeia)')]:
        idx = [i for i, r in enumerate(reps_remocao) if ('sequencial' in r) == seq]
        sub = te.iloc[idx]
        ax.errorbar(x[idx], sub['diferença (p.p.)'], yerr=sub['DP da diferença (p.p.)'],
                    fmt=marcador, color=cor, ms=7, capsize=3, lw=1.8, zorder=5, label=nome)
    for xi, (_, l) in zip(x, te.iterrows()):
        marca = '*' if l['veredito'] != 'não muda' else ''
        ax.annotate(f"{l['diferença (p.p.)']:+.2f}{marca}".replace('.', ','),
                    (xi, l['diferença (p.p.)']), textcoords='offset points',
                    xytext=(9, 0), va='center', fontsize=8, color='#222222', zorder=6,
                    bbox={'boxstyle': 'round,pad=0.1', 'fc': 'white', 'ec': 'none', 'alpha': 0.8})
    ax.set_title(f"{clf} (base: {te['F1 base (%)'].iloc[0]:.2f}%)".replace('.', ','))
    ax.set_xticks(x)
    ax.set_xticklabels([rotulo(r) for r in reps_remocao], fontsize=8)
    ax.set_xlim(-0.5, len(x) - 0.3)
axes[0].set_ylabel('Diferença no F1-macro em relação\naos 16 atributos (p.p.)')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.03),
           fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig(os.path.join(RESULTS_PATH, "h2_efeito_remocao.png"), bbox_inches='tight')
plt.close()

# ==============================================================================
# 6. GRÁFICO 2: matriz de correlação (|r| de Pearson), marcando o limiar em que
#    cada atributo é removido. Descrição da base: usa todos os dados (como a
#    variância do PCA na H3); os conjuntos removidos são iguais em todos os folds.
# ==============================================================================
df_beans = pd.read_excel(DATASET_PATH)
X = df_beans.drop(columns=['Class'])
y = df_beans['Class']

atr = df_atrib.set_index('representacao')
sai_em = {}
for rep in reps_regra:  # do limiar mais alto para o mais baixo
    removidos = str(atr.loc[rep, 'atributos_removidos']).split(';')
    rep_seq = rep.replace('remocao', 'remocao sequencial')
    # removido pela regra, mas mantido pela variante sequencial = remoção em cadeia
    em_cadeia = set(atr.loc[rep_seq, 'atributos_mantidos'].split(';')) if rep_seq in atr.index else set()
    for nome in removidos:
        sai_em.setdefault(nome, rep.split('>')[1].replace('.', ',') + ('*' if nome in em_cadeia else ''))
rotulos_corr = [f"{c} (sai em {sai_em[c]})" if c in sai_em else c for c in X.columns]

corr = X.corr().abs()
fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, square=True,
            annot_kws={'fontsize': 7}, cbar_kws={'label': '|r| de Pearson', 'shrink': 0.8},
            xticklabels=X.columns, yticklabels=rotulos_corr, ax=ax)
ax.set_title('Correlação absoluta entre os atributos\n(rótulo: limiar em que o atributo é removido; '
             '* = removido em cadeia)')
ax.tick_params(axis='x', rotation=90, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "h2_correlacao.png"), bbox_inches='tight')
plt.close()

# ==============================================================================
# 7. GRÁFICO 3: matrizes de confusão (15 folds somados, mesmas divisões dos CSVs)
#    Naive Bayes: sem remoção × |r|>0,95 (maior ganho)
#    KNN: sem remoção × |r|>0,90 (maior perda)
# ==============================================================================
print("\nGerando Matrizes de Confusão agregadas (15 folds)...")
classes = sorted(y.unique())
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
divisoes = list(cv.split(X, y))


def matriz(clf, limiar):
    etapa = 'passthrough' if limiar is None else RemoverCorrelacionados(limiar=limiar)
    pipe = Pipeline([('scaler', StandardScaler()), ('remocao', etapa), ('clf', clf)])
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for tr, te in divisoes:
        pipe.fit(X.iloc[tr], y.iloc[tr])
        cm += confusion_matrix(y.iloc[te], pipe.predict(X.iloc[te]), labels=classes)
    return cm / cm.sum(axis=1, keepdims=True) * 100


casos = [('Naive Bayes', GaussianNB, 0.95), ('KNN', KNeighborsClassifier, 0.90)]
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
for i, (nome_clf, Clf, limiar) in enumerate(casos):
    cm_base = matriz(Clf(), None)
    cm_rem = matriz(Clf(), limiar)
    dif = cm_rem - cm_base
    print(f"\n{nome_clf}: recall por classe (%) — sem remoção → |r|>{limiar:.2f}")
    print(pd.DataFrame({'sem remoção': np.diag(cm_base), f'|r|>{limiar:.2f}': np.diag(cm_rem),
                        'diferença': np.diag(dif)}, index=classes).round(2).to_string())

    limiar_br = f"{limiar:.2f}".replace('.', ',')
    titulos = [f"{nome_clf}\nsem remoção, 16 atrib. (%)",
               f"{nome_clf}\n|r|>{limiar_br}, {n_atrib[f'remocao r>{limiar:.2f}']} atrib. (%)",
               f"{nome_clf}\ndiferença (p.p.)"]
    for j, (m, tit) in enumerate(zip([cm_base, cm_rem, dif], titulos)):
        ax = axes[i, j]
        if j < 2:
            sns.heatmap(m, annot=True, fmt='.1f', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=classes, yticklabels=classes)
        else:
            lim = max(np.abs(dif).max(), 0.1)
            sns.heatmap(m, annot=True, fmt='+.1f', cmap='RdBu', cbar=False, ax=ax,
                        norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                        xticklabels=classes, yticklabels=classes)
        ax.set_title(tit)
        ax.set_xlabel('Predito' if i == 1 else '')
        ax.set_ylabel('Real' if j == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "h2_matrizes_confusao.png"), bbox_inches='tight')
plt.close()

print(f"\nAnálise concluída! Arquivos salvos em '{RESULTS_PATH}':")
for nome in ["tabela_resumo_H2.csv", "tabela_efeito_H2.csv", "tabela_controle_H2.csv",
             "h2_efeito_remocao.png", "h2_correlacao.png", "h2_matrizes_confusao.png"]:
    print(f"- {os.path.join(RESULTS_PATH, nome)}")
