"""
Comparação geral: junta os resultados de H1, H2 e H3 e gera tabelas e gráficos gerais.

Lê (os que existirem):
    resultados/H1/resultados_H1.csv
    resultados/H2/resultados_H2.csv
    resultados/H3/resultados_H3.csv
Se algum CSV ainda não existir (ex.: H2), ele é pulado e o script roda com os demais.

Gera em resultados/geral/:
    resultados_todos.csv          -> todos os CSVs concatenados (por fold)
    tabela_geral.csv              -> média ± DP de F1, acurácia e tempo por hipótese/representação/classificador
    tabela_melhores.csv           -> por classificador: brutos, padronizados e melhor representação de H2/H3
    tabela_melhores.tex           -> mesma tabela, pronta para \\input no relatório
    geral_melhores_por_hipotese.png -> barras: brutos x padronizados x melhor de H2/H3
    geral_heatmap_f1.png          -> F1 médio de todas as representações x classificadores

Rodar a partir da raiz do repositório:
    python codigos/comparacao_geral.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Caminhos e configurações
# ==============================================================================
CSVS = {
    'H1': os.path.join("resultados", "H1", "resultados_H1.csv"),
    'H2': os.path.join("resultados", "H2", "resultados_H2.csv"),
    'H3': os.path.join("resultados", "H3", "resultados_H3.csv"),
}
OUT_PATH = os.path.join("resultados", "geral")
os.makedirs(OUT_PATH, exist_ok=True)

COLUNAS = ['hipotese', 'representacao', 'classificador', 'repeticao', 'fold',
           'acuracia', 'f1_macro', 'tempo_s']

# Classificadores da comparação geral (a variante "Naive Bayes (var_smoothing=0)"
# da H1 entra na tabela completa, mas não nos gráficos/tabela de melhores).
ORDEM_CLF = ['KNN', 'Naive Bayes', 'SVM']
REFERENCIA = ('H1', 'dados brutos')         # ponto de partida
REFERENCIA_PADR = ('H1', 'padronizados')  # base para medir o efeito de H2 e H3

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ==============================================================================
# 2. Leitura e junção dos CSVs
# ==============================================================================
partes = []
for hip, caminho in CSVS.items():
    if not os.path.exists(caminho):
        print(f"[aviso] {caminho} não encontrado -> {hip} será pulada.")
        continue
    df = pd.read_csv(caminho)
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"{caminho} não tem as colunas combinadas: {faltando}")
    partes.append(df[COLUNAS])
    print(f"[ok] {hip}: {len(df)} linhas, {df['representacao'].nunique()} representações.")

if not partes:
    raise SystemExit("Nenhum CSV encontrado. Rode a partir da raiz do repositório.")

todos = pd.concat(partes, ignore_index=True)
hipoteses = sorted(todos['hipotese'].unique())
todos.to_csv(os.path.join(OUT_PATH, "resultados_todos.csv"), index=False)

# ==============================================================================
# 3. Tabela geral (média ± DP das 15 execuções)
# ==============================================================================
# Mantém a ordem em que as representações aparecem em cada CSV
ordem_rep = {(h, r): i for i, (h, r) in
             enumerate(todos[['hipotese', 'representacao']].drop_duplicates().itertuples(index=False))}

geral = (todos.groupby(['hipotese', 'representacao', 'classificador'], sort=False)
         .agg(f1_media=('f1_macro', 'mean'), f1_std=('f1_macro', 'std'),
              acc_media=('acuracia', 'mean'), acc_std=('acuracia', 'std'),
              tempo_media=('tempo_s', 'mean'), tempo_std=('tempo_s', 'std'),
              n_execucoes=('f1_macro', 'size'))
         .reset_index())
for col in ['f1_media', 'f1_std', 'acc_media', 'acc_std']:
    geral[col] *= 100  # em %

geral['_ordem'] = [ordem_rep[(h, r)] for h, r in zip(geral['hipotese'], geral['representacao'])]
geral = geral.sort_values(['_ordem', 'classificador']).drop(columns='_ordem')

tabela_geral = pd.DataFrame({
    'hipotese': geral['hipotese'],
    'representacao': geral['representacao'],
    'classificador': geral['classificador'],
    'F1-Macro': [f"{m:.2f}% ± {s:.2f}%" for m, s in zip(geral['f1_media'], geral['f1_std'])],
    'Acurácia': [f"{m:.2f}% ± {s:.2f}%" for m, s in zip(geral['acc_media'], geral['acc_std'])],
    'Tempo (s)': [f"{m:.3f}s ± {s:.3f}s" for m, s in zip(geral['tempo_media'], geral['tempo_std'])],
    'execucoes': geral['n_execucoes'],
})
tabela_geral.to_csv(os.path.join(OUT_PATH, "tabela_geral.csv"), index=False)

# ==============================================================================
# 4. Melhores representações por hipótese (comparação geral)
# ==============================================================================
# Dados brutos e padronizados (H1) servem de referência. Como os pipelines de H2 e H3
# padronizam os dados antes da remoção/PCA, o ganho dessas hipóteses é medido em
# relação aos dados PADRONIZADOS (efeito próprio da técnica, sem o efeito da escala).
base = geral[geral['classificador'].isin(ORDEM_CLF)]
hips_tab = [h for h in hipoteses if h != 'H1']


def pega(g, hip, rep):
    r = g[(g['hipotese'] == hip) & (g['representacao'] == rep)]
    return (r['f1_media'].iloc[0], r['f1_std'].iloc[0]) if len(r) else (np.nan, np.nan)


linhas = []
for clf in ORDEM_CLF:
    g = base[base['classificador'] == clf]
    linha = {'classificador': clf}
    linha['brutos_f1'], linha['brutos_std'] = pega(g, *REFERENCIA)
    linha['padr_f1'], linha['padr_std'] = pega(g, *REFERENCIA_PADR)
    for hip in hips_tab:
        gh = g[g['hipotese'] == hip]
        if gh.empty:
            continue
        melhor = gh.loc[gh['f1_media'].idxmax()]
        linha[f'{hip}_rep'] = melhor['representacao']
        linha[f'{hip}_f1'] = melhor['f1_media']
        linha[f'{hip}_std'] = melhor['f1_std']
        linha[f'{hip}_ganho_pp'] = melhor['f1_media'] - linha['padr_f1']
    linhas.append(linha)
melhores = pd.DataFrame(linhas)
hips_tab = [h for h in hips_tab if f'{h}_rep' in melhores]


def fmt(m, s):
    return f"{m:.2f}% ± {s:.2f}%"


tab_melhores = pd.DataFrame({
    'Classificador': melhores['classificador'],
    'Dados brutos (F1)': [fmt(m, s) for m, s in zip(melhores['brutos_f1'], melhores['brutos_std'])],
    'Padronizados (F1)': [fmt(m, s) for m, s in zip(melhores['padr_f1'], melhores['padr_std'])],
    'Ganho da padronização (p.p.)': [f"{p - b:+.2f}" for p, b in zip(melhores['padr_f1'], melhores['brutos_f1'])],
})
for hip in hips_tab:
    tab_melhores[f'Melhor {hip}'] = melhores[f'{hip}_rep']
    tab_melhores[f'{hip} F1'] = [fmt(m, s) for m, s in zip(melhores[f'{hip}_f1'], melhores[f'{hip}_std'])]
    tab_melhores[f'{hip} vs padronizados (p.p.)'] = [f"{g:+.2f}" for g in melhores[f'{hip}_ganho_pp']]
tab_melhores.to_csv(os.path.join(OUT_PATH, "tabela_melhores.csv"), index=False)


# Tabela LaTeX (vírgula decimal, mesmo estilo booktabs do relatório)
def br(x, casas=2):
    return f"{x:.{casas}f}".replace('.', ',')


def sinal(x):
    return ('+' if x >= 0 else '') + br(x)


def tex_escape(s):
    return (str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_')
            .replace('%', r'\%').replace('>', r'$>$').replace('&', r'\&')
            .replace('remocao', 'remoção').replace('.', '{,}'))


tex = [r"\begin{table}[htbp]", r"    \centering",
       r"    \caption{F1-macro (\%) médio por classificador: dados brutos, dados padronizados e melhor "
       r"representação de " + " e ".join(hips_tab) + r". Entre parênteses: na coluna Padronizados, ganho "
       r"sobre os dados brutos; nas demais, diferença em relação aos padronizados (p.p.).}",
       r"    \label{tab:comparacao_geral}", r"    \footnotesize",
       r"    \begin{tabular}{l" + "c" * (2 + len(hips_tab)) + "}", r"        \toprule",
       "        \\textbf{Classificador} & \\textbf{Brutos} & \\textbf{Padronizados} & "
       + " & ".join(f"\\textbf{{Melhor {h}}}" for h in hips_tab) + r" \\",
       r"        \midrule"]
for _, l in melhores.iterrows():
    celulas = [l['classificador'], br(l['brutos_f1']),
               f"{br(l['padr_f1'])} ({sinal(l['padr_f1'] - l['brutos_f1'])})"]
    for h in hips_tab:
        celulas.append(f"{br(l[f'{h}_f1'])} ({sinal(l[f'{h}_ganho_pp'])}) "
                       f"\\scriptsize[{tex_escape(l[f'{h}_rep'])}]")
    tex.append("        " + " & ".join(celulas) + r" \\")
tex += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
with open(os.path.join(OUT_PATH, "tabela_melhores.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(tex) + "\n")

# ==============================================================================
# 5. Figura 1: dados brutos x padronizados x melhor representação de H2/H3
# ==============================================================================
grupos = [('Dados brutos', 'brutos'), ('Padronizados (H1)', 'padr')] + \
         [(f'Melhor {h}', h) for h in hips_tab]
cores = ['#9e9e9e', '#4c72b0', '#dd8452', '#55a868', '#c44e52'][:len(grupos)]
x = np.arange(len(ORDEM_CLF))
largura = 0.8 / len(grupos)

fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (rotulo, chave) in enumerate(grupos):
    medias = melhores[f'{chave}_f1'].values
    desvios = melhores[f'{chave}_std'].values
    pos = x - 0.4 + largura * (i + 0.5)
    barras = ax.bar(pos, medias, largura, yerr=desvios, capsize=3,
                    label=rotulo, color=cores[i], edgecolor='white')
    for j, b in enumerate(barras):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{medias[j]:.1f}",
                ha='center', va='bottom', fontsize=8)
        if chave in hips_tab:  # nome da representação escolhida dentro da barra
            ax.text(b.get_x() + b.get_width() / 2, 52,
                    melhores[f'{chave}_rep'].iloc[j].replace('remocao', 'remoção'),
                    ha='center', va='bottom', rotation=90, fontsize=7, color='white')

ax.set_xticks(x)
ax.set_xticklabels(ORDEM_CLF)
ax.set_ylabel('F1-macro (%)')
ax.set_ylim(50, 100)
ax.set_title('Dados brutos, padronizados e melhor representação de cada hipótese')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(grupos), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PATH, "geral_melhores_por_hipotese.png"), bbox_inches='tight')
plt.close(fig)

# ==============================================================================
# 6. Figura 2: heatmap com o F1 médio de todas as representações
# ==============================================================================
pivot = base.pivot_table(index=['hipotese', 'representacao'], columns='classificador',
                         values='f1_media', sort=False)[ORDEM_CLF]
pivot = pivot.loc[sorted(pivot.index, key=lambda k: ordem_rep[k])]
rotulos = [f"{h} | {r}" for h, r in pivot.index]

fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(pivot) + 1.5))
im = ax.imshow(pivot.values, cmap='viridis', aspect='auto',
               vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
ax.set_xticks(range(len(ORDEM_CLF)))
ax.set_xticklabels(ORDEM_CLF)
ax.set_yticks(range(len(rotulos)))
ax.set_yticklabels(rotulos, fontsize=8)
limiar_cor = np.nanmean([np.nanmin(pivot.values), np.nanmax(pivot.values)])
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha='center', va='center', fontsize=8,
                    color='black' if v > limiar_cor else 'white')
# linhas separando as hipóteses
hips_idx = [h for h, _ in pivot.index]
for i in range(1, len(hips_idx)):
    if hips_idx[i] != hips_idx[i - 1]:
        ax.axhline(i - 0.5, color='white', linewidth=2)
fig.colorbar(im, ax=ax, label='F1-macro (%)')
ax.set_title('F1-macro médio por representação e classificador')
fig.tight_layout()
fig.savefig(os.path.join(OUT_PATH, "geral_heatmap_f1.png"), bbox_inches='tight')
plt.close(fig)

# ==============================================================================
# 7. Resumo no terminal
# ==============================================================================
print("\n=== MELHOR REPRESENTAÇÃO POR CLASSIFICADOR EM CADA HIPÓTESE ===")
print(tab_melhores.to_string(index=False))
print(f"\nArquivos salvos em ./{OUT_PATH}/")
