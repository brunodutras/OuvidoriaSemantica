# Ouvidoria Municipal — Busca Semântica de Manifestações

Sistema de apoio à triagem de manifestações de uma ouvidoria municipal: busca por
similaridade semântica, detecção de duplicatas e chunking de relatos longos.
Corpus de 40 manifestações em cinco categorias oficiais.

O relatório completo, com resultados, decisões de projeto e análise de erros, está em
**[`RELATORIO.pdf`](RELATORIO.pdf)** (fonte em [`RELATORIO.md`](RELATORIO.md)).

## Instalação

Testado em Python 3.14 (macOS/arm64).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Na primeira execução os modelos de embedding são baixados do HuggingFace Hub
(~1 GB, em cache a partir daí).

## Execução

```bash
# Aplicação Streamlit (Entrega 4)
.venv/bin/streamlit run app_ouvidoria.py

# Notebooks (já vêm com todas as saídas executadas)
.venv/bin/jupyter lab

# Regerar o PDF do relatório a partir do markdown
.venv/bin/python gerar_relatorio_pdf.py

# Regerar o corpus (valida por asserção as restrições do enunciado)
.venv/bin/python data/gerar_dataset.py
```

## Estrutura

| Arquivo | Entrega | Conteúdo |
|---|---|---|
| `análise_comparativa.ipynb` | 1 | BoW × TF-IDF × embeddings nos três pares do enunciado |
| `deteccao_duplicatas.ipynb` | 2 | `detectar_duplicatas()`, calibração do limiar, análise de erros |
| `chunking_manifestacoes.ipynb` | 3 | `RecursiveCharacterTextSplitter`, overlap, projeções 2D |
| `app_ouvidoria.py` | 4 | App Streamlit com as quatro abas |
| `utils_ouvidoria.py` | — | Carga do corpus, registro de modelos, paleta (compartilhado) |
| `gerar_relatorio_pdf.py` | — | Converte `RELATORIO.md` em PDF paginado |
| `data/manifestacoes.csv` | — | O corpus (40 manifestações) |
| `data/gabarito_duplicatas.csv` | — | Anotação manual: 6 duplicatas reais + 4 quase-duplicatas |
| `data/gerar_dataset.py` | — | Script que gera e valida o corpus |
| `figuras/` | — | Figuras usadas no relatório |

A lógica avaliada em cada entrega fica visível dentro dos notebooks;
`utils_ouvidoria.py` guarda apenas infraestrutura (caminhos, carga, registro de modelos).

## Principais resultados

- **Representações esparsas erram o caso central.** Em BoW, um par não relacionado
  (falta de médico × lâmpada queimada) marca 0,32 contra 0,12 de duas denúncias do
  mesmo buraco na mesma avenida — a ordem está invertida. Remover stopwords corrige o
  falso positivo (0,32 → 0,05) mas não cria a similaridade ausente (0,12 → 0,10).
- **O limiar de 0,85 não detecta nada** neste corpus: as seis duplicatas reais ficam
  entre 0,54 e 0,78. Calibrado contra o gabarito, o F1 máximo (0,67) ocorre em 0,68.
  O limiar varia por modelo (0,68 / 0,71 / 0,50 / 0,93) — **limiares pertencem ao
  modelo, não ao problema**.
- **O `chunk_overlap` pode ser silenciosamente ignorado.** Com separadores de frase, um
  orçamento de 90 caracteres produz 5,5 de overlap realizado, porque o splitter o gasta
  em unidades inteiras do separador. Configuração recomendada: `chunk_size` 300 com
  `chunk_overlap` 90.
- **Similaridade semântica não é identidade de ocorrência.** Os falsos positivos são
  problemas do mesmo tipo em locais diferentes. Daí a recomendação de uma faixa de
  triagem (>0,75 duplicata provável; 0,68–0,75 conferência humana) em vez de corte único.

## Modelos de embedding

O enunciado sugere `BAAI/bge-small-pt-v1.5`, mas esse repositório não existe no
HuggingFace Hub (404). O padrão adotado é `paraphrase-multilingual-MiniLM-L12-v2`.
Os quatro modelos do registro são selecionáveis na barra lateral do app:

| Modelo | Dim. | Observação |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Padrão. Leve e rápido (118M) |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | Maior (278M) e mais preciso, mais lento |
| `distiluse-base-multilingual-cased-v1` | 512 | Escala de similaridade mais comprimida |
| `multilingual-e5-small` | 384 | Exige prefixo `query:`, aplicado pelo código |
