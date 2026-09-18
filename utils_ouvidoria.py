"""Utilitários compartilhados pelos notebooks e pelo app Streamlit.

Mantém aqui apenas o que é infraestrutura (caminhos, carga do corpus, registro
de modelos). A lógica analisada nas entregas — vetorização, similaridade,
detecção de duplicatas e chunking — fica visível dentro dos notebooks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CORPUS_CSV = DATA_DIR / "manifestacoes.csv"
GABARITO_CSV = DATA_DIR / "gabarito_duplicatas.csv"

CATEGORIAS = ["infraestrutura", "saúde", "segurança", "educação", "meio ambiente"]

# O enunciado sugere "BAAI/bge-small-pt-v1.5", mas esse repositório não existe
# no HuggingFace Hub (retorna 404). Usamos o MiniLM multilíngue no lugar dele (ver
# MODELO_PADRAO abaixo); o mpnet fica disponível como alternativa mais precisa.
MODELOS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "prefixo": "",
        "descricao": "Leve e rápido (118M). Padrão do trabalho.",
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dim": 768,
        "prefixo": "",
        "descricao": "Maior (278M) e mais preciso, porém mais lento.",
    },
    "distiluse-base-multilingual-cased-v1": {
        "path": "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "dim": 512,
        "prefixo": "",
        "descricao": "Destilado; escala de similaridade mais comprimida.",
    },
    "multilingual-e5-small": {
        "path": "intfloat/multilingual-e5-small",
        "dim": 384,
        "prefixo": "query: ",
        "descricao": "Exige prefixo 'query:'; similaridades muito altas para todos os pares.",
    },
}

MODELO_PADRAO = "paraphrase-multilingual-MiniLM-L12-v2"

# Lista enxuta de stopwords do português; o sklearn não traz uma nativa.
STOPWORDS_PT = [
    "a", "à", "às", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "depois", "do", "dos", "e", "é", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "está", "estamos",
    "estão", "estas", "este", "estes", "estou", "eu", "foi", "fomos", "for", "foram",
    "há", "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu",
    "meus", "minha", "minhas", "muito", "na", "não", "nas", "nem", "no", "nos",
    "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou",
    "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem",
    "se", "sem", "ser", "seu", "seus", "só", "sob", "sobre", "sua", "suas", "são",
    "também", "te", "tem", "tém", "tenho", "ter", "teu", "teus", "tinha", "tive",
    "todos", "tua", "tuas", "tudo", "um", "uma", "umas", "uns", "vocês", "vos",
]


# --- Paleta -----------------------------------------------------------------
# Conjunto categórico validado para uso em dispersão (todos os pares): passa a
# faixa de luminosidade, o piso de croma e o piso de visão normal (ΔE 16,3).
# A separação para daltonismo fica em ΔE 6,1, o que exige codificação
# secundária: por isso todo gráfico de dispersão usa também o formato do
# marcador, e nunca só a cor, para indicar a categoria.
CORES_CATEGORIA = {
    "infraestrutura": "#2a78d6",
    "saúde": "#eda100",
    "segurança": "#1baf7a",
    "educação": "#4a3aa7",
    "meio ambiente": "#e87ba4",
}

MARCADORES_CATEGORIA = {
    "infraestrutura": "o",
    "saúde": "s",
    "segurança": "^",
    "educação": "D",
    "meio ambiente": "P",
}

# Rampa sequencial de um só tom (azul, claro -> escuro) para os heatmaps de
# similaridade, que codificam magnitude. Evita o arco-íris.
RAMPA_SEQUENCIAL = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]

TINTA = {
    "primaria": "#0b0b0b",
    "secundaria": "#52514e",
    "suave": "#898781",
    "grade": "#e1e0d9",
}


def cmap_sequencial():
    """Colormap sequencial de um tom para matrizes de similaridade."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ouvidoria_azul", RAMPA_SEQUENCIAL)


def carregar_corpus() -> pd.DataFrame:
    """Carrega as 40 manifestações com colunas derivadas de tamanho."""
    df = pd.read_csv(CORPUS_CSV)
    df["n_caracteres"] = df.texto.str.len()
    df["n_palavras"] = df.texto.str.split().str.len()
    df["texto_longo"] = df.n_caracteres > 500
    return df


def carregar_gabarito() -> pd.DataFrame:
    """Pares anotados manualmente: duplicatas reais e quase-duplicatas."""
    return pd.read_csv(GABARITO_CSV)


@lru_cache(maxsize=4)
def carregar_modelo(nome: str = MODELO_PADRAO):
    """Instancia (e memoriza) um SentenceTransformer do registro MODELOS."""
    from sentence_transformers import SentenceTransformer

    if nome not in MODELOS:
        raise KeyError(f"modelo desconhecido: {nome}. Opções: {list(MODELOS)}")
    return SentenceTransformer(MODELOS[nome]["path"])


def gerar_embeddings(textos: list[str], nome: str = MODELO_PADRAO) -> np.ndarray:
    """Codifica textos já aplicando o prefixo exigido pelo modelo e normalizando.

    Com vetores normalizados, o produto interno é a própria similaridade de cosseno.
    """
    modelo = carregar_modelo(nome)
    prefixo = MODELOS[nome]["prefixo"]
    return modelo.encode(
        [prefixo + t for t in textos],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def pares_superiores(matriz: np.ndarray, ids: list[str], limiar: float) -> list[tuple]:
    """Pares (i<j) da matriz de similaridade acima do limiar, do maior para o menor."""
    i, j = np.triu_indices(len(ids), k=1)
    mask = matriz[i, j] >= limiar
    pares = [(ids[a], ids[b], float(matriz[a, b])) for a, b in zip(i[mask], j[mask])]
    return sorted(pares, key=lambda p: p[2], reverse=True)
