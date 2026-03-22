import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import expon, kstest

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PASTA_DADOS = "dados"

st.set_page_config(
    page_title="Call Center · Análise Estatística",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILO
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}

.metric-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 8px;
}
.metric-label {
    color: #64748b;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    color: #38bdf8;
    font-size: 28px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 4px;
}
.metric-sub {
    color: #475569;
    font-size: 11px;
    margin-top: 2px;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-exp {
    background: #052e16;
    color: #4ade80;
    border: 1px solid #16a34a;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1px;
}
.badge-noexp {
    background: #431407;
    color: #fb923c;
    border: 1px solid #c2410c;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1px;
}
.section-divider {
    border-top: 1px solid #1e3a5f;
    margin: 32px 0 24px 0;
}
.teoria-box {
    background: #0c1a2e;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
    margin: 12px 0;
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.7;
}
.formula {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 14px 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px;
    color: #7dd3fc;
    text-align: center;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FUNÇÕES
# ─────────────────────────────────────────────

def ler_arquivo(caminho):
    with open(caminho, 'r') as f:
        dados = [float(l.strip()) for l in f if l.strip()]
    return np.array(dados)

def converter_acumulado(dados):
    return np.diff(dados, prepend=dados[0])

def extrair_horario(nome_arquivo):
    """Extrai a hora inicial do arquivo e gera rótulo 'Chamadas HH–HH+1'."""
    import re
    numeros = re.findall(r'\d+', nome_arquivo)
    if numeros:
        h = int(numeros[0])
        return f"Chamadas {h:02d}h–{(h+1):02d}h"
    # fallback
    return nome_arquivo.replace(".txt", "")

def hora_inicial(nome_arquivo):
    """Retorna a hora inicial para ordenação."""
    import re
    numeros = re.findall(r'\d+', nome_arquivo)
    return int(numeros[0]) if numeros else 99

def analisar(dados):
    media = float(np.mean(dados))
    variancia = float(np.var(dados))
    desvio = float(np.std(dados))
    lam = 1.0 / media if media > 0 else 0
    stat, p = kstest(dados, 'expon', args=expon.fit(dados))
    return dict(media=media, variancia=variancia, desvio=desvio, lam=lam, ks_stat=stat, ks_p=p)

def e_exponencial(m):
    """Critério duplo: razão média/desvio + KS test."""
    razao_ok = abs(m["media"] - m["desvio"]) < 0.2 * m["media"]
    ks_ok = m["ks_p"] > 0.05
    return razao_ok or ks_ok

# ─────────────────────────────────────────────
# CARREGAMENTO
# ─────────────────────────────────────────────

@st.cache_data
def carregar_todos():
    arquivos = sorted(
        [f for f in os.listdir(PASTA_DADOS) if f.endswith(".txt")],
        key=hora_inicial
    )
    registros = []
    for arq in arquivos:
        caminho = os.path.join(PASTA_DADOS, arq)
        dados = ler_arquivo(caminho)
        if "Horario" in arq:
            dados = converter_acumulado(dados)
        m = analisar(dados)
        registros.append({
            "arquivo": arq,
            "rotulo": extrair_horario(arq),
            "dados": dados,
            **m,
            "exponencial": e_exponencial(m),
        })
    return registros

registros = carregar_todos()
df_resumo = pd.DataFrame([{k: v for k, v in r.items() if k != "dados"} for r in registros])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Painel de Controle")
    st.markdown("---")

    opcoes = [r["rotulo"] for r in registros]
    rotulo_sel = st.selectbox("Intervalo para análise detalhada:", opcoes)
    reg_sel = next(r for r in registros if r["rotulo"] == rotulo_sel)

    st.markdown("---")
    st.markdown("### 📌 Sobre o projeto")
    st.markdown("""
Análise dos tempos entre chamadas em um call center ao longo de 24 intervalos horários. Os dados de 17h–20h estavam em formato acumulado e foram pré-processados. Foco em identificar aderência à distribuição exponencial e estimar λ por período.    """)

# ─────────────────────────────────────────────
# CABEÇALHO
# ─────────────────────────────────────────────
st.markdown("# 📞 Call Center · Análise de Tempos Entre Chamadas")

st.markdown(
    "Modelagem probabilística dos intervalos entre requisições ao longo do dia — "
    "identificação de distribuição exponencial e estimativa do parâmetro λ.<br><br>"
    "Os dados analisados representam os tempos entre requisições de um sistema ao longo de um dia de funcionamento, "
    "organizados em intervalos horários. Para a maior parte dos períodos, os arquivos já continham diretamente os "
    "tempos entre chamadas, o que permitiu uma análise estatística imediata.<br><br>"
    "No entanto, os dados referentes ao intervalo entre 17h e 20h apresentavam uma estrutura diferente, pois "
    "registravam os instantes de ocorrência das chamadas em formato acumulado, medidos em segundos. Como o sistema "
    "responsável pela coleta reinicia o relógio a cada hora cheia, foi necessário tratar cada intervalo separadamente, "
    "convertendo os dados acumulados em tempos entre requisições por meio da diferença entre valores consecutivos.<br><br>"
    "Após esse pré-processamento, todos os dados puderam ser analisados de maneira uniforme, possibilitando a comparação "
    "entre os diferentes períodos do dia. A análise estatística realizada indica que, em diversos intervalos, os dados "
    "apresentam comportamento compatível com uma distribuição exponencial, característica de processos de chegada aleatórios.",
    unsafe_allow_html=True
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SEÇÃO 1 — VISÃO GERAL DO DIA
# ─────────────────────────────────────────────

st.markdown("## 01 · Comportamento ao Longo do Dia")
st.caption("Cada ponto representa um intervalo horário. Observe como a média e o desvio padrão variam — intervalos com maior λ (menor média) indicam maior intensidade de chamadas.")

# Gráfico de tendência: média e desvio ao longo do dia
fig_trend = go.Figure()

x_labels = df_resumo["rotulo"]
x_idx = list(range(len(df_resumo)))

# Banda de desvio
fig_trend.add_trace(go.Scatter(
    x=x_idx + x_idx[::-1],
    y=list(df_resumo["media"] + df_resumo["desvio"]) + list((df_resumo["media"] - df_resumo["desvio"]).clip(0))[::-1],
    fill='toself',
    fillcolor='rgba(56,189,248,0.08)',
    line=dict(color='rgba(0,0,0,0)'),
    name='±1 Desvio Padrão',
    hoverinfo='skip',
))

# Linha de média
fig_trend.add_trace(go.Scatter(
    x=x_idx,
    y=df_resumo["media"],
    mode='lines+markers',
    name='Média (s)',
    line=dict(color='#38bdf8', width=2),
    marker=dict(
        size=9,
        color=['#4ade80' if e else '#fb923c' for e in df_resumo["exponencial"]],
        line=dict(color='#38bdf8', width=1.5)
    ),
    customdata=np.stack([df_resumo["rotulo"], df_resumo["lam"], df_resumo["desvio"]], axis=-1),
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "Média: %{y:.3f} s<br>"
        "Desvio: %{customdata[2]:.3f} s<br>"
        "λ: %{customdata[1]:.4f} chamadas/s<extra></extra>"
    )
))

fig_trend.update_layout(
    paper_bgcolor='#0a1628',
    plot_bgcolor='#0a1628',
    font=dict(color='#e2e8f0', family='IBM Plex Mono'),
    xaxis=dict(
        tickvals=x_idx, ticktext=x_labels,
        tickangle=-45, gridcolor='#1e3a5f', color='#64748b'
    ),
    yaxis=dict(title="Tempo médio entre chamadas (s)", gridcolor='#1e3a5f', color='#64748b'),
    legend=dict(bgcolor='#0f172a', bordercolor='#1e3a5f', borderwidth=1),
    height=380,
    margin=dict(t=20, b=80),
)

st.plotly_chart(fig_trend, use_container_width=True)

col_l, col_r = st.columns(2)
with col_l:
    n_exp = df_resumo["exponencial"].sum()
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Intervalos com aderência exponencial</div>
        <div class="metric-value">{n_exp} / {len(df_resumo)}</div>
    </div>
    """, unsafe_allow_html=True)
with col_r:
    idx_pico = df_resumo["lam"].idxmax()
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Intervalo de maior intensidade (λ máx)</div>
        <div class="metric-value">{df_resumo.loc[idx_pico, 'rotulo']}</div>
        <div class="metric-sub">λ = {df_resumo.loc[idx_pico, 'lam']:.4f} chamadas/s</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="teoria-box">
<b>Leitura do gráfico:</b> o pico de tempo médio entre chamadas ocorre nas primeiras horas da madrugada (03h–05h), 
o que pode induzir a uma leitura equivocada — de que esse é o período de maior atividade. 
Na realidade, ocorre o oposto: <b>um tempo médio alto significa que as chamadas chegam com menos frequência</b>, 
ou seja, o sistema está menos sobrecarregado nesses horários.<br><br>

O verdadeiro pico de intensidade está nos períodos em que a curva se aproxima de zero — 
especialmente entre 09h e 17h — onde os intervalos entre chamadas são muito curtos, 
indicando uma taxa de chegada λ elevada. 
<b>Alta frequência de chamadas e longos intervalos são fenômenos opostos</b>, 
e o gráfico deixa isso claro quando lido pelo eixo correto.<br><br>

Em termos estatísticos: quanto menor a média dos intervalos, maior o λ estimado — 
e portanto maior a demanda sobre o sistema naquele período.
</div>
""", unsafe_allow_html=True)

# Gráfico de λ por intervalo
st.markdown("### Estimativa de λ por intervalo")
st.caption("λ (lambda) é a taxa média de chegada de chamadas por segundo. Quanto maior o λ, mais frequentes as chamadas naquele período.")

fig_lambda = go.Figure(go.Bar(
    x=x_idx,
    y=df_resumo["lam"],
    marker_color=['#38bdf8' if e else '#475569' for e in df_resumo["exponencial"]],
    customdata=np.stack([df_resumo["rotulo"], df_resumo["media"]], axis=-1),
    hovertemplate="<b>%{customdata[0]}</b><br>λ = %{y:.4f}<br>Média = %{customdata[1]:.3f} s<extra></extra>",
    name="λ estimado"
))
fig_lambda.update_layout(
    paper_bgcolor='#0a1628',
    plot_bgcolor='#0a1628',
    font=dict(color='#e2e8f0', family='IBM Plex Mono'),
    xaxis=dict(tickvals=x_idx, ticktext=x_labels, tickangle=-45, gridcolor='#1e3a5f', color='#64748b'),
    yaxis=dict(title="λ (chamadas/s)", gridcolor='#1e3a5f', color='#64748b'),
    height=300,
    margin=dict(t=10, b=80),
    showlegend=False,
)
st.plotly_chart(fig_lambda, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SEÇÃO 2 — ANÁLISE INDIVIDUAL
# ─────────────────────────────────────────────

st.markdown(f"## 02 · Análise Detalhada — `{reg_sel['rotulo']}`")
st.caption("Selecione o intervalo no painel lateral para detalhar.")

m = reg_sel
dados = m["dados"]

# Métricas
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Média (μ)</div>
        <div class="metric-value">{m['media']:.4f} s</div>
        <div class="metric-sub">tempo médio entre chamadas</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Desvio padrão (σ)</div>
        <div class="metric-value">{m['desvio']:.4f} s</div>
        <div class="metric-sub">dispersão dos intervalos</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Taxa λ estimada</div>
        <div class="metric-value">{m['lam']:.4f}</div>
        <div class="metric-sub">chamadas por segundo</div>
    </div>""", unsafe_allow_html=True)
with c4:
    badge = '<span class="badge-exp">✔ EXPONENCIAL</span>' if m["exponencial"] else '<span class="badge-noexp">⚠ DIVERGE</span>'
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Aderência ao modelo</div>
        <div style="margin-top:10px">{badge}</div>
        <div class="metric-sub">KS p-valor: {m['ks_p']:.4f}</div>
    </div>""", unsafe_allow_html=True)

# Histograma com ajuste exponencial
loc_fit, scale_fit = expon.fit(dados)
x_fit = np.linspace(min(dados), max(dados), 300)
y_fit = expon.pdf(x_fit, loc_fit, scale_fit)

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=dados,
    nbinsx=40,
    histnorm='probability density',
    name='Dados observados',
    marker_color='#1d4ed8',
    opacity=0.65,
))
fig_hist.add_trace(go.Scatter(
    x=x_fit, y=y_fit,
    mode='lines',
    name=f'Exp(λ={m["lam"]:.4f})',
    line=dict(color='#38bdf8', width=2.5),
))
fig_hist.add_vline(x=m["media"], line_dash="dash", line_color="#f59e0b",
                   annotation_text=f"μ = {m['media']:.3f}", annotation_position="top right",
                   annotation_font_color="#f59e0b")

fig_hist.update_layout(
    paper_bgcolor='#0a1628',
    plot_bgcolor='#0a1628',
    font=dict(color='#e2e8f0', family='IBM Plex Mono'),
    xaxis=dict(title='Tempo entre chamadas (s)', gridcolor='#1e3a5f', color='#64748b'),
    yaxis=dict(title='Densidade de probabilidade', gridcolor='#1e3a5f', color='#64748b'),
    legend=dict(bgcolor='#0f172a', bordercolor='#1e3a5f', borderwidth=1),
    barmode='overlay',
    height=380,
    margin=dict(t=20),
)
st.plotly_chart(fig_hist, use_container_width=True)

# QQ-plot manual
st.markdown("### Q-Q Plot — Quantis Observados vs. Exponencial Teórica")
st.caption("Se os pontos seguirem a linha diagonal, os dados têm distribuição exponencial. Desvios nas caudas são comuns mesmo em dados exponenciais.")

n = len(dados)
quantis_teoricos = expon.ppf(np.linspace(0.01, 0.99, n), loc=loc_fit, scale=scale_fit)
quantis_obs = np.sort(dados)

fig_qq = go.Figure()
fig_qq.add_trace(go.Scatter(
    x=quantis_teoricos, y=quantis_obs,
    mode='markers',
    marker=dict(color='#38bdf8', size=5, opacity=0.7),
    name='Quantis observados',
))
lim = max(quantis_teoricos.max(), quantis_obs.max())
fig_qq.add_trace(go.Scatter(
    x=[0, lim], y=[0, lim],
    mode='lines',
    line=dict(color='#f59e0b', dash='dash'),
    name='Referência (ideal)',
))
fig_qq.update_layout(
    paper_bgcolor='#0a1628',
    plot_bgcolor='#0a1628',
    font=dict(color='#e2e8f0', family='IBM Plex Mono'),
    xaxis=dict(title='Quantis teóricos (Exponencial)', gridcolor='#1e3a5f', color='#64748b'),
    yaxis=dict(title='Quantis observados', gridcolor='#1e3a5f', color='#64748b'),
    legend=dict(bgcolor='#0f172a', bordercolor='#1e3a5f', borderwidth=1),
    height=360,
    margin=dict(t=20),
)
st.plotly_chart(fig_qq, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SEÇÃO 3 — DISTRIBUIÇÃO EXPONENCIAL
# ─────────────────────────────────────────────

st.markdown("## 03 · Modelagem com Distribuição Exponencial")

col_t1, col_t2 = st.columns([3, 2])

with col_t1:
    st.markdown("""
<div class="teoria-box">
<b>O que é a distribuição exponencial?</b><br><br>
A distribuição exponencial modela o <b>tempo de espera entre eventos</b> que ocorrem de forma aleatória e independente em uma taxa constante λ (lambda). Ela é a única distribuição contínua com a propriedade de <i>ausência de memória</i>: o tempo até a próxima chamada não depende de quanto tempo já se passou desde a última.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="teoria-box">
<b>Por que ela aparece em call centers?</b><br><br>
Em sistemas de filas (teoria de filas de Erlang), assume-se que chamadas chegam seguindo um <b>Processo de Poisson</b> — eventos independentes com taxa constante. O tempo entre dois eventos de Poisson consecutivos segue, matematicamente, uma distribuição exponencial com o mesmo parâmetro λ. Isso torna a exponencial o modelo natural para descrever intervalos entre chamadas.
</div>
""", unsafe_allow_html=True)

with col_t2:
    st.markdown('<div class="formula">f(x) = λ · e^(−λx),  x ≥ 0</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula">E[X] = 1/λ &nbsp;&nbsp;&nbsp; Var[X] = 1/λ²</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula">σ = 1/λ = E[X]</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="teoria-box" style="font-size:13px">
<b>Propriedade-chave:</b><br>
Na distribuição exponencial, <b>média = desvio padrão</b>. 
Por isso, quando μ ≈ σ em um intervalo horário, isso é um forte indício de aderência ao modelo exponencial.
</div>
""", unsafe_allow_html=True)

# Visualização interativa da exponencial
st.markdown("### Efeito de λ sobre a distribuição")
st.caption("Deslize para ver como λ altera a forma da curva. O λ estimado do intervalo selecionado está destacado.")

lambda_demo = st.slider("λ (chamadas/s)", min_value=0.01, max_value=5.0, value=round(m["lam"], 2), step=0.01)

x_demo = np.linspace(0, 10 / max(lambda_demo, 0.01), 300)

fig_demo = go.Figure()
for lv, cor, opac in [
    (lambda_demo * 0.5, '#64748b', 0.5),
    (lambda_demo, '#38bdf8', 1.0),
    (lambda_demo * 2.0, '#7c3aed', 0.5),
]:
    y_demo = expon.pdf(x_demo, scale=1/lv)
    fig_demo.add_trace(go.Scatter(
        x=x_demo, y=y_demo,
        mode='lines',
        name=f'λ = {lv:.3f}',
        line=dict(color=cor, width=2 if opac == 1 else 1.5),
        opacity=opac,
    ))

fig_demo.update_layout(
    paper_bgcolor='#0a1628',
    plot_bgcolor='#0a1628',
    font=dict(color='#e2e8f0', family='IBM Plex Mono'),
    xaxis=dict(title='Tempo entre chamadas (s)', gridcolor='#1e3a5f', color='#64748b'),
    yaxis=dict(title='f(x)', gridcolor='#1e3a5f', color='#64748b'),
    legend=dict(bgcolor='#0f172a', bordercolor='#1e3a5f', borderwidth=1),
    height=320,
    margin=dict(t=20),
)
st.plotly_chart(fig_demo, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SEÇÃO 4 — TABELA COMPARATIVA
# ─────────────────────────────────────────────

st.markdown("## 04 · Comparação de Todos os Intervalos")

df_exib = df_resumo[["rotulo", "media", "desvio", "variancia", "lam", "ks_p", "exponencial"]].copy()
df_exib.columns = ["Intervalo", "Média (s)", "Desvio Padrão (s)", "Variância", "λ estimado", "KS p-valor", "Exponencial?"]
df_exib["Exponencial?"] = df_exib["Exponencial?"].map({True: "✔ Sim", False: "✘ Não"})
df_exib = df_exib.round(4)

st.dataframe(
    df_exib,
    use_container_width=True,
    height=400,
    hide_index=True,
)

st.caption("KS p-valor: teste de Kolmogorov-Smirnov. p > 0.05 indica que não há evidência suficiente para rejeitar a hipótese de distribuição exponencial.")

# ─────────────────────────────────────────────
# SEÇÃO 5 — CONCLUSÃO
# ─────────────────────────────────────────────

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 05 · Conclusão")

st.markdown("""
<div class="teoria-box">
A análise estatística dos tempos entre chamadas ao longo dos 24 intervalos horários indica que, na maioria dos períodos,
os dados são compatíveis com uma distribuição exponencial — confirmada tanto pela proximidade entre média e desvio padrão
(propriedade teórica da exponencial) quanto pelo teste de Kolmogorov-Smirnov (KS).<br><br>

Esse resultado sustenta a hipótese de que o fluxo de chamadas segue um <b>Processo de Poisson</b>, com chegadas aleatórias
e independentes a uma taxa λ variável ao longo do dia. Os picos de λ identificados nos gráficos correspondem a períodos
de maior intensidade de uso, informação essencial para o dimensionamento de equipes em sistemas de filas.<br><br>

Nos intervalos que divergem do modelo exponencial, é possível que haja <b>sobreposição de subpopulações</b> (ex.: picos de
chamadas em horários específicos dentro do intervalo) ou variação não estacionária na taxa de chegada — situações que
demandariam modelos mais avançados, como distribuições hiper-exponenciais ou processos de Poisson não homogêneos.
</div>
""", unsafe_allow_html=True)