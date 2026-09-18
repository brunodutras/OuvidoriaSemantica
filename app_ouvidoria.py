"""Ouvidoria Municipal — busca semântica de manifestações (Entrega 4).

Execute com:  streamlit run app_ouvidoria.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

from utils_ouvidoria import (
    CORES_CATEGORIA,
    MODELO_PADRAO,
    MODELOS,
    RAMPA_SEQUENCIAL,
    carregar_corpus,
    carregar_gabarito,
    carregar_modelo,
)

st.set_page_config(page_title="Ouvidoria — Busca Semântica", page_icon="🔎", layout="wide")

# Limiar calibrado na Entrega 2: o valor ótimo depende do modelo, não do problema.
LIMIAR_DUPLICATA = {
    "paraphrase-multilingual-MiniLM-L12-v2": 0.68,
    "paraphrase-multilingual-mpnet-base-v2": 0.71,
    "distiluse-base-multilingual-cased-v1": 0.50,
    "multilingual-e5-small": 0.93,
}

SIMBOLOS_CATEGORIA = {
    "infraestrutura": "circle",
    "saúde": "square",
    "segurança": "triangle-up",
    "educação": "diamond",
    "meio ambiente": "cross",
}

SEPARADORES = {
    "palavra (padrão do LangChain)": ["\n\n", "\n", " ", ""],
    "frase (quebra em pontuação)": ["\n\n", "\n", ". ", "; ", ", ", " ", ""],
}


# --- Cache -------------------------------------------------------------------
# O modelo é um recurso pesado e não serializável: cache_resource.
# Os embeddings são dados derivados: cache_data, indexado pelo nome do modelo.

@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def obter_modelo(nome: str):
    return carregar_modelo(nome)


@st.cache_data(show_spinner=False)
def obter_corpus() -> pd.DataFrame:
    return carregar_corpus()


@st.cache_data(show_spinner=False)
def obter_gabarito() -> pd.DataFrame:
    return carregar_gabarito()


@st.cache_data(show_spinner="Gerando embeddings...")
def codificar(textos: tuple[str, ...], nome_modelo: str) -> np.ndarray:
    modelo = obter_modelo(nome_modelo)
    prefixo = MODELOS[nome_modelo]["prefixo"]
    return modelo.encode(
        [prefixo + t for t in textos], normalize_embeddings=True, show_progress_bar=False
    )


@st.cache_data(show_spinner=False)
def projetar(embeddings: np.ndarray, metodo: str, semente: int = 0) -> np.ndarray:
    if metodo == "PCA":
        return PCA(n_components=2, random_state=semente).fit_transform(embeddings)
    perplexidade = min(30, max(2, (len(embeddings) - 1) // 3))
    return TSNE(
        n_components=2, perplexity=perplexidade, learning_rate="auto",
        init="pca", random_state=semente,
    ).fit_transform(embeddings)


def faixa_de_score(score: float) -> tuple[str, str]:
    """Rótulo e cor da faixa. O rótulo textual evita depender só da cor."""
    if score > 0.7:
        return "similaridade alta", "#0ca30c"
    if score > 0.5:
        return "similaridade média", "#fab219"
    return "similaridade baixa", "#898781"


# --- Barra lateral -----------------------------------------------------------
with st.sidebar:
    st.header("Configuração")

    nome_modelo = st.selectbox(
        "Modelo de embedding",
        options=list(MODELOS),
        index=list(MODELOS).index(MODELO_PADRAO),
        help="Modelos multilíngues do sentence-transformers.",
    )
    st.caption(
        f"{MODELOS[nome_modelo]['descricao']}  \n"
        f"Dimensões: **{MODELOS[nome_modelo]['dim']}**"
    )

    top_k = st.slider("Resultados na busca (top-k)", min_value=1, max_value=15, value=5)

    st.divider()
    st.caption(
        "**Limiar de duplicata**  \n"
        f"`{LIMIAR_DUPLICATA[nome_modelo]:.2f}` para este modelo — calibrado na Entrega 2. "
        "Cada modelo ocupa uma faixa própria da escala de cosseno, por isso o limiar "
        "acompanha a escolha do modelo."
    )

    if MODELOS[nome_modelo]["prefixo"]:
        st.warning(
            f"Este modelo exige o prefixo `{MODELOS[nome_modelo]['prefixo']}` nos textos, "
            "aplicado automaticamente."
        )

df = obter_corpus()
textos = tuple(df.texto)
embeddings = codificar(textos, nome_modelo)

st.title("Ouvidoria Municipal — Busca Semântica")
st.caption(
    f"{len(df)} manifestações · {df.categoria.nunique()} categorias · "
    f"modelo `{nome_modelo}`"
)

aba_busca, aba_base, aba_espaco, aba_chunking = st.tabs(
    ["🔎 Busca Semântica", "📋 Base Completa", "🗺️ Espaço Vetorial", "✂️ Chunking"]
)


# --- Aba 1: busca semântica --------------------------------------------------
with aba_busca:
    st.subheader("Descreva o problema com suas palavras")
    st.caption(
        "A busca compara o **sentido** da descrição com o das manifestações registradas — "
        "não é necessário acertar as mesmas palavras."
    )

    exemplos = [
        "não tem médico atendendo no posto do meu bairro",
        "a rua está cheia de buracos e estraga os carros",
        "lixo acumulado em terreno abandonado atraindo ratos",
        "minha filha está sem professor de matemática na escola",
        "está tudo escuro na praça porque a luz queimou",
    ]
    escolhido = st.selectbox("Exemplos prontos", ["(digitar manualmente)"] + exemplos)
    valor_inicial = "" if escolhido == "(digitar manualmente)" else escolhido

    consulta = st.text_area(
        "Sua manifestação", value=valor_inicial, height=90,
        placeholder="Ex.: faz semanas que a lâmpada do poste da minha rua está queimada",
    )

    if consulta.strip():
        vetor = codificar((consulta.strip(),), nome_modelo)
        scores = cosine_similarity(vetor, embeddings).ravel()
        melhores = scores.argsort()[::-1][:top_k]

        limiar = LIMIAR_DUPLICATA[nome_modelo]
        if scores[melhores[0]] >= limiar:
            st.info(
                f"**Possível duplicata.** A manifestação `{df.id.iloc[melhores[0]]}` tem "
                f"similaridade {scores[melhores[0]]:.2f}, acima do limiar {limiar:.2f} "
                "deste modelo. Confira se não é o mesmo problema já registrado."
            )

        st.write("")
        for posicao, i in enumerate(melhores, start=1):
            linha = df.iloc[i]
            score = float(scores[i])
            rotulo, cor = faixa_de_score(score)

            with st.container(border=True):
                cabecalho, medidor = st.columns([5, 1.6])
                with cabecalho:
                    st.markdown(
                        f"**{posicao}. {linha.id}** &nbsp;·&nbsp; "
                        f"<span style='color:{CORES_CATEGORIA[linha.categoria]}'>●</span> "
                        f"{linha.categoria}",
                        unsafe_allow_html=True,
                    )
                with medidor:
                    st.markdown(
                        f"<div style='text-align:right'>"
                        f"<span style='color:{cor};font-weight:700;font-size:1.25rem'>{score:.3f}</span><br>"
                        f"<span style='color:#52514e;font-size:0.72rem'>{rotulo}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.write(linha.texto)
                st.progress(max(0.0, min(1.0, score)))

        with st.expander("Como ler os scores"):
            st.markdown(
                """
                A similaridade de cosseno vai de -1 a 1; na prática, textos de ouvidoria
                ficam entre 0 e 0,9. As faixas usadas aqui:

                | faixa | leitura |
                |---|---|
                | **acima de 0,7** | quase certamente o mesmo assunto |
                | **0,5 a 0,7** | assunto relacionado; vale conferir |
                | **abaixo de 0,5** | provavelmente outro problema |

                As faixas **dependem do modelo**. O `multilingual-e5-small`, por exemplo,
                mantém quase todos os pares acima de 0,85 — nele, um score de 0,80 é
                *baixo*. Trocar o modelo na barra lateral e comparar os scores da mesma
                consulta deixa esse efeito visível.
                """
            )
    else:
        st.info("Digite uma descrição acima ou escolha um dos exemplos.")


# --- Aba 2: base completa ----------------------------------------------------
with aba_base:
    st.subheader("Todas as manifestações registradas")

    colunas = st.columns(4)
    colunas[0].metric("Manifestações", len(df))
    colunas[1].metric("Categorias", df.categoria.nunique())
    colunas[2].metric("Textos longos (>500)", int(df.texto_longo.sum()))
    colunas[3].metric("Média de caracteres", f"{df.n_caracteres.mean():.0f}")

    filtro = st.multiselect(
        "Filtrar por categoria", options=sorted(df.categoria.unique()), default=[]
    )
    visivel = df[df.categoria.isin(filtro)] if filtro else df

    st.dataframe(
        visivel[["id", "categoria", "texto", "n_caracteres", "n_palavras"]],
        width="stretch", hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "categoria": st.column_config.TextColumn("Categoria", width="medium"),
            "texto": st.column_config.TextColumn("Manifestação", width="large"),
            "n_caracteres": st.column_config.NumberColumn("Caracteres", width="small"),
            "n_palavras": st.column_config.NumberColumn("Palavras", width="small"),
        },
    )

    st.divider()
    st.markdown("#### Matriz de similaridade")
    st.caption(
        "Compara todas as manifestações entre si. Ordenada por categoria, os blocos "
        "claros ao longo da diagonal mostram a estrutura temática que o modelo recupera "
        "sem nunca ter visto os rótulos."
    )

    if st.button("Gerar matriz de similaridade", type="primary"):
        ordem = df.sort_values(["categoria", "id"]).index.to_numpy()
        matriz = cosine_similarity(embeddings[ordem])
        rotulos = df.id.iloc[ordem].tolist()
        categorias_ordenadas = df.categoria.iloc[ordem].tolist()

        np.fill_diagonal(matriz, np.nan)
        figura = px.imshow(
            matriz, x=rotulos, y=rotulos,
            color_continuous_scale=RAMPA_SEQUENCIAL, zmin=0, zmax=0.8, aspect="auto",
            labels=dict(color="similaridade"),
        )
        figura.update_traces(
            hovertemplate="%{y} × %{x}<br>similaridade: %{z:.3f}<extra></extra>"
        )
        limites = [k for k in range(1, len(categorias_ordenadas))
                   if categorias_ordenadas[k] != categorias_ordenadas[k - 1]]
        for k in limites:
            figura.add_hline(y=k - 0.5, line_width=1.5, line_color="white")
            figura.add_vline(x=k - 0.5, line_width=1.5, line_color="white")
        figura.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(figura, width="stretch")

        gabarito = obter_gabarito()
        limiar = LIMIAR_DUPLICATA[nome_modelo]
        matriz_sem_nan = cosine_similarity(embeddings)
        i, j = np.triu_indices(len(df), k=1)
        acima = matriz_sem_nan[i, j] >= limiar
        reais = {
            frozenset((r.id_a, r.id_b))
            for r in gabarito[gabarito.relacao == "duplicata"].itertuples()
        }
        pares = pd.DataFrame({
            "id_a": df.id.to_numpy()[i[acima]],
            "id_b": df.id.to_numpy()[j[acima]],
            "similaridade": matriz_sem_nan[i, j][acima].round(3),
        }).sort_values("similaridade", ascending=False)
        if len(pares):
            pares["no gabarito?"] = [
                "duplicata real" if frozenset((r.id_a, r.id_b)) in reais else "conferir"
                for r in pares.itertuples()
            ]

        st.markdown(f"##### Pares acima do limiar ({limiar:.2f})")
        if len(pares):
            st.dataframe(pares, width="stretch", hide_index=True)
        else:
            st.warning(
                f"Nenhum par atinge {limiar:.2f} com este modelo. "
                "Veja a Entrega 2: o limiar precisa ser calibrado por modelo."
            )


# --- Aba 3: espaço vetorial --------------------------------------------------
with aba_espaco:
    st.subheader("As manifestações no espaço semântico")

    controles = st.columns([1, 1, 2])
    metodo = controles[0].radio("Projeção", ["PCA", "t-SNE"], horizontal=True)
    mostrar_ids = controles[1].checkbox("Mostrar IDs", value=False)

    projecao = projetar(embeddings, metodo)
    plano = pd.DataFrame({
        "x": projecao[:, 0], "y": projecao[:, 1],
        "id": df.id, "categoria": df.categoria,
        "texto": df.texto.str.slice(0, 110) + "...",
    })

    figura = px.scatter(
        plano, x="x", y="y", color="categoria", symbol="categoria",
        color_discrete_map=CORES_CATEGORIA, symbol_map=SIMBOLOS_CATEGORIA,
        hover_data={"id": True, "texto": True, "x": False, "y": False},
        text="id" if mostrar_ids else None,
    )
    figura.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
    if mostrar_ids:
        figura.update_traces(textposition="top center", textfont=dict(size=9))
    figura.update_layout(
        height=560, xaxis_title=None, yaxis_title=None,
        legend_title_text="categoria oficial",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    figura.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    figura.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    st.plotly_chart(figura, width="stretch")

    # Os clusters semânticos coincidem com as categorias oficiais?
    from sklearn.cluster import KMeans

    agrupador = KMeans(n_clusters=df.categoria.nunique(), n_init=10, random_state=0)
    clusters = agrupador.fit_predict(embeddings)
    ari = adjusted_rand_score(df.categoria, clusters)

    S = cosine_similarity(embeddings)
    i, j = np.triu_indices(len(df), k=1)
    mesma = df.categoria.to_numpy()[i] == df.categoria.to_numpy()[j]
    intra, inter = S[i, j][mesma].mean(), S[i, j][~mesma].mean()

    metricas = st.columns(3)
    metricas[0].metric("Similaridade média dentro da categoria", f"{intra:.3f}")
    metricas[1].metric("Entre categorias diferentes", f"{inter:.3f}")
    metricas[2].metric("Concordância cluster × categoria (ARI)", f"{ari:.3f}")

    st.markdown(
        f"""
#### Os clusters semânticos coincidem com as categorias oficiais?

**Em parte — e as exceções são as mais informativas.**

O sinal existe e é consistente: manifestações da mesma categoria são mais parecidas entre
si ({intra:.3f}) do que de categorias diferentes ({inter:.3f}). O modelo recupera a
estrutura temática **sem nunca ter visto os rótulos**, apenas lendo os textos.

Mas a coincidência está longe de ser perfeita. Agrupando os embeddings em
{df.categoria.nunique()} clusters com K-Means e comparando com as categorias oficiais, o
índice Rand ajustado fica em **{ari:.3f}** — bem acima do acaso (0), bem abaixo de uma
correspondência exata (1).

As divergências têm explicação, e não são erro do modelo:

- **As categorias oficiais são administrativas, não semânticas.** Elas dizem qual
  secretaria atende o chamado. "Escola sem água" é *educação* e "posto de saúde sem água"
  é *saúde*, mas os dois textos descrevem o mesmo fato — falta de água em prédio público —
  e o modelo os aproxima, corretamente.
- **Um mesmo problema atravessa categorias.** A manifestação `M023` está em *segurança*,
  mas sua causa declarada é a iluminação queimada do beco, tema de *infraestrutura*; ela
  fica entre os dois grupos porque é, de fato, os dois.
- **Categorias amplas se espalham.** *Meio ambiente* reúne desde descarte de lixo até
  poda de árvore e esgoto a céu aberto — assuntos que o modelo, com razão, não coloca
  no mesmo ponto.

Para a ouvidoria isso é uma informação útil, não um defeito: quando um cidadão classifica
sua manifestação numa categoria e o vetor cai no meio de outra, vale conferir o
encaminhamento. A proximidade semântica é um bom detector de **classificação
administrativa duvidosa**.

Vale lembrar a diferença entre as duas projeções: o **PCA** preserva distâncias globais,
mas em duas dimensões explica pouca variância de um espaço de
{MODELOS[nome_modelo]['dim']} dimensões, então grupos distintos podem se sobrepor na
tela. O **t-SNE** separa melhor visualmente, porém só a vizinhança local é confiável —
a distância *entre* grupos não é interpretável. As métricas acima são calculadas no
espaço original, e é nelas que a conclusão se apoia.
"""
    )


# --- Aba 4: chunking ---------------------------------------------------------
with aba_chunking:
    st.subheader("Dividir uma manifestação longa em chunks")
    st.caption(
        "Relatos longos misturam vários problemas num só texto. Um embedding único vira "
        "a média de todos eles; o chunking dá um vetor a cada assunto."
    )

    longas = df[df.texto_longo].sort_values("n_caracteres", ascending=False)
    fonte = st.radio(
        "Texto de entrada",
        ["Escolher uma manifestação longa do corpus", "Colar um texto"],
        horizontal=True,
    )

    if fonte.startswith("Escolher"):
        escolha = st.selectbox(
            "Manifestação",
            options=longas.id.tolist(),
            format_func=lambda m: f"{m} — {longas.set_index('id').categoria[m]} "
                                  f"({longas.set_index('id').n_caracteres[m]} caracteres)",
        )
        texto_entrada = longas.set_index("id").texto[escolha]
    else:
        texto_entrada = st.text_area(
            "Cole a manifestação", height=180,
            placeholder="Cole aqui um relato longo para ver como ele é dividido...",
        )

    parametros = st.columns(3)
    estrategia = parametros[0].selectbox("Estratégia de separadores", list(SEPARADORES))
    chunk_size = parametros[1].slider("chunk_size", 100, 600, 300, step=50)
    chunk_overlap = parametros[2].slider("chunk_overlap", 0, 300, 90, step=10)

    if chunk_overlap >= chunk_size:
        st.error("O overlap precisa ser menor que o chunk_size.")
    elif texto_entrada.strip():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARADORES[estrategia],
            keep_separator="end" if estrategia.startswith("frase") else "start",
        )
        pedacos = splitter.split_text(texto_entrada.strip())

        def overlap_realizado(a: str, b: str) -> int:
            for k in range(min(len(a), len(b)), 0, -1):
                if a[-k:] == b[:k]:
                    return k
            return 0

        emendas = [overlap_realizado(a, b) for a, b in zip(pedacos, pedacos[1:])]
        media_emendas = float(np.mean(emendas)) if emendas else 0.0

        resumo = st.columns(4)
        resumo[0].metric("Chunks gerados", len(pedacos))
        resumo[1].metric("Média de caracteres", f"{np.mean([len(p) for p in pedacos]):.0f}")
        resumo[2].metric("Overlap pedido", chunk_overlap)
        resumo[3].metric(
            "Overlap realizado", f"{media_emendas:.0f}",
            delta=f"{media_emendas - chunk_overlap:.0f}",
            delta_color="normal" if media_emendas >= chunk_overlap * 0.8 else "inverse",
        )

        if chunk_overlap > 0 and media_emendas < chunk_overlap * 0.5:
            st.warning(
                f"**O overlap pedido não está acontecendo** ({media_emendas:.0f} de "
                f"{chunk_overlap} caracteres). O `RecursiveCharacterTextSplitter` gasta o "
                "orçamento de overlap em unidades inteiras do separador que usou: com "
                "separadores de frase, uma frase inteira não cabe em um orçamento pequeno "
                "e a sobreposição é descartada. Aumente o overlap ou use separadores de "
                "palavra."
            )

        st.markdown("#### Chunks gerados")
        for k, pedaco in enumerate(pedacos):
            sobreposicao = emendas[k - 1] if k else 0
            with st.container(border=True):
                st.markdown(
                    f"**chunk {k}** · {len(pedaco)} caracteres"
                    + (f" · {sobreposicao} herdados do anterior" if sobreposicao else "")
                )
                if sobreposicao:
                    st.markdown(
                        f"<span style='background:#cde2fb;padding:1px 3px;border-radius:3px'>"
                        f"{pedaco[:sobreposicao]}</span>{pedaco[sobreposicao:]}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(pedaco)

        if len(pedacos) > 1:
            st.markdown("#### Embeddings dos chunks")
            E_chunks = codificar(tuple(pedacos), nome_modelo)
            S_chunks = cosine_similarity(E_chunks)

            esquerda, direita = st.columns(2)

            with esquerda:
                rotulos = [f"chunk {k}" for k in range(len(pedacos))]
                figura = px.imshow(
                    S_chunks, x=rotulos, y=rotulos, zmin=0, zmax=1,
                    color_continuous_scale=RAMPA_SEQUENCIAL,
                    labels=dict(color="similaridade"), text_auto=".2f",
                )
                figura.update_layout(
                    height=380, margin=dict(l=10, r=10, t=40, b=10),
                    title=dict(text="Similaridade entre os chunks", font=dict(size=13)),
                )
                st.plotly_chart(figura, width="stretch")

            with direita:
                if len(pedacos) >= 3:
                    projecao_chunks = projetar(E_chunks, "PCA")
                    plano_chunks = pd.DataFrame({
                        "x": projecao_chunks[:, 0], "y": projecao_chunks[:, 1],
                        "chunk": rotulos,
                        "trecho": [p[:90] + "..." for p in pedacos],
                    })
                    figura = px.line(
                        plano_chunks, x="x", y="y", text="chunk",
                        hover_data={"trecho": True, "x": False, "y": False},
                    )
                    figura.update_traces(
                        line=dict(color="#c3c2b7", width=1),
                        mode="lines+markers+text",
                        marker=dict(size=13, color="#2a78d6", line=dict(width=1, color="white")),
                        textposition="top center", textfont=dict(size=9),
                    )
                    figura.update_layout(
                        height=380, xaxis_title=None, yaxis_title=None,
                        margin=dict(l=10, r=10, t=40, b=10),
                        title=dict(text="Percurso dos chunks (PCA)", font=dict(size=13)),
                    )
                    figura.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
                    figura.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
                    st.plotly_chart(figura, width="stretch")
                else:
                    st.info("A projeção 2D aparece a partir de 3 chunks.")

            consecutivos = [float(S_chunks[k, k + 1]) for k in range(len(pedacos) - 1)]
            st.caption(
                f"Coesão média entre chunks consecutivos: **{np.mean(consecutivos):.3f}**. "
                "Lembre que o overlap infla essa medida mecanicamente — vizinhos que "
                "compartilham texto literal são parecidos por repetição, não por "
                "continuidade de sentido (ver Entrega 3)."
            )
    else:
        st.info("Escolha uma manifestação ou cole um texto para ver os chunks.")
