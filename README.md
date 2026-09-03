# TabFM no BigQuery: demo de detecção de fraude com `AI.PREDICT`

Material de demonstração do **TabFM** (foundation model tabular do Google Research) dentro do
BigQuery, usando um caso de fraude em transações de cartão com dados sintéticos.

| Artefato | Descrição |
|---|---|
| [`notebooks/tabfm_fraud_demo.ipynb`](notebooks/tabfm_fraud_demo.ipynb) | Notebook para **Colab Enterprise**: gera dados, carrega no BigQuery, roda `AI.PREDICT` / `AI.EVALUATE`, analisa limiar e fila de alertas, demonstra adaptação sem re-treino e compara com `BOOSTED_TREE_CLASSIFIER`. |
| [`slides/tabfm_apresentacao.html`](slides/tabfm_apresentacao.html) | Apresentação em HTML (20 slides, navegação por teclado) para conduzir a conversa com o cliente: conceito, sintaxe, vantagens, quando usar TabFM × BQML treinado × Vertex AI, operacionalização, custos e limites. |
| [`data/generate_fraud_data.py`](data/generate_fraud_data.py) | Gerador determinístico dos dados sintéticos (também embutido no notebook). |

## Como usar

### Notebook (Colab Enterprise)

1. Abra o Colab Enterprise no console do GCP (BigQuery Studio > Notebooks ou Vertex AI > Colab Enterprise) e importe `notebooks/tabfm_fraud_demo.ipynb`.
2. Na primeira célula, informe `PROJECT_ID` (e, se quiser, `DATASET` e `LOCATION`).
3. Execute as células em ordem. Cada `AI.PREDICT` sobre ~100 mil linhas de treino leva alguns minutos.

Permissões necessárias: `roles/bigquery.user` no projeto e `roles/bigquery.dataEditor` no dataset da demo.
O TabFM está em Preview; se a função não estiver disponível no projeto, contate `bqml-feedback@google.com`.

### Apresentação

Abra `slides/tabfm_apresentacao.html` no navegador. Setas para navegar, `F` para tela cheia, Ctrl/Cmd+P para exportar PDF.

### Gerar os dados fora do notebook

```bash
pip install numpy pandas
python data/generate_fraud_data.py --rows 120000 --seed 42 --out transactions.csv
```

## Cenário dos dados sintéticos

- ~120 mil transações (jan a jul/2026), 20 mil clientes, taxa de fraude ≈ 3 %.
- 18 features (limite do TabFM: 20): valor, categoria, canal, cartão presente, internacional, distância de casa,
  dispositivo novo, velocidade 24 h, falhas de autenticação, idade do cliente e da conta, ticket médio,
  utilização de limite, chargebacks anteriores, hora e dia da semana.
- 30 % das fraudes são camufladas como legítimas e 3 % das legítimas parecem suspeitas.
- Um padrão de fraude novo surge em 01/06/2026, usado para demonstrar adaptação sem re-treino.

## Resultados da execução de referência

Split temporal: treino jan–jun (102,7 mil linhas), predição julho (18 mil linhas, 807 fraudes).

| Métrica | TabFM (`AI.PREDICT`, zero-shot) | `BOOSTED_TREE_CLASSIFIER` (BQML) |
|---|---|---|
| precision | 0,988 | 0,971 |
| recall | 0,824 | 0,830 |
| f1_score | 0,899 | 0,895 |
| AUC-ROC | 0,985 | 0,985 |
| AUC-PR | 0,928 | 0,925 |
| tempo até a primeira previsão | ≈ 4 min (uma query) | ≈ 20 min de treino + `ML.PREDICT` |

Adaptação sem re-treino (recall em julho sobre o padrão novo): 13 % com treino jan–mai, 96 % com treino jan–jun.

## Referências

- [Documentação `AI.PREDICT`](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-predict)
- [Documentação `AI.EVALUATE`](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-evaluate)
- [Blog: Introducing TabFM in BigQuery](https://cloud.google.com/blog/products/data-analytics/tabfm-adds-predictive-ml-to-bigquery)
- [Google Research: TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
