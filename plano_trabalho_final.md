# Trabalho Final — Quem faz o quê

**Base:** Dry Bean Dataset (UCI) — 13.611 amostras, 16 atributos, 7 classes.

## Hipóteses

- **H1:** A padronização ajuda muito o KNN e o SVM, mas quase não muda o Naive Bayes.
- **H2:** Remover atributos redundantes (muito correlacionados) ajuda o Naive Bayes, mas quase não muda o KNN e o SVM.
- **H3:** O PCA ajuda mais o Naive Bayes do que o KNN e o SVM.

Cada hipótese roda **os 3 classificadores** (KNN, Naive Bayes, SVM).

## Combinado na reunião inicial (todos seguem igual)

- Validação: 5-fold estratificado, 3 repetições, `random_state=42`.
- Transformações sempre dentro do `Pipeline` do scikit-learn.
- Métricas: F1-macro (principal), acurácia, matriz de confusão.
- Cada um salva um CSV com as colunas: `hipotese, representacao, classificador, repeticao, fold, acuracia, f1_macro, tempo_s`.

## Divisão

| Pessoa | Código | Relatório | Slides |
|---|---|---|---|
| **1** | **H1:** dados brutos, MinMax e padronizados × 3 classificadores | Resultados da H1 (½ página) | 2 slides da H1 |
| **2** | **H2:** remoção por correlação (sem remoção, 0,99, 0,95, 0,90) × 3 classificadores | Resultados da H2 (½ página) | 2 slides da H2 |
| **3** | **H3:** PCA com 2, 4, 6, 8, 10 e 16 componentes × 3 classificadores | Resultados da H3 (½ página) | 2 slides da H3 |
| **4** | Análise exploratória da base (classes, correlação, escalas) | Introdução + Metodologia | Slides da base, hipóteses e protocolo |
| **5** | Script que junta os CSVs e gera a tabela e os gráficos gerais | Comparação geral + Conclusão + revisão final (máx. 4 páginas) | Monta o arquivo dos slides + conclusão |

Ninguém depende do código de outra pessoa. As Pessoas 4 e 5 começam no primeiro dia (a Pessoa 5 testa o script com um CSV de exemplo).

## Etapas

1. **Reunião inicial (todos):** fechar hipóteses e o combinado acima.
2. **Trabalho em paralelo:** cada um faz sua parte da tabela.
3. **Reunião de resultados (todos):** cada hipótese é confirmada, refutada ou inconclusiva.
4. **Fechamento:** a Pessoa 5 junta tudo. Todos revisam o relatório e os slides.
