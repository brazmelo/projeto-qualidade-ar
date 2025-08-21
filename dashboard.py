import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title="Dashboard - Qualidade do Ar",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Dashboard - Qualidade do Ar")
st.markdown("---")

# Carregar dados e modelo com tratamento de erro
try:
    df = pd.read_csv("dados/dados_iot.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    model = joblib.load("dados/modelo_rf.pkl")
except FileNotFoundError as e:
    st.error(f"Erro ao carregar arquivos: {e}")
    st.stop()
except Exception as e:
    st.error(f"Erro inesperado: {e}")
    st.stop()

# Criar features
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday
df["day_name"] = df["timestamp"].dt.day_name()

# Função para classificar qualidade do ar
def classify_air_quality(pm25):
    if pm25 <= 12:
        return "Boa", "🟢"
    elif pm25 <= 35:
        return "Moderada", "🟡"
    elif pm25 <= 55:
        return "Insalubre para grupos sensíveis", "🟠"
    elif pm25 <= 150:
        return "Insalubre", "🔴"
    else:
        return "Muito Insalubre", "🟣"

df["quality"], df["quality_icon"] = zip(*df["PM2.5"].apply(classify_air_quality))

# Sidebar com filtros
st.sidebar.header("🔧 Filtros")

# Seleção de período
period_type = st.sidebar.selectbox(
    "Período de análise:",
    ["Dia específico", "Intervalo de datas", "Última semana"]
)

if period_type == "Dia específico":
    data_selecionada = st.sidebar.date_input(
        "Escolha a data:", 
        df["timestamp"].min().date(),
        min_value=df["timestamp"].min().date(),
        max_value=df["timestamp"].max().date()
    )
    df_filtrado = df[df["timestamp"].dt.date == data_selecionada]
elif period_type == "Intervalo de datas":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Data inicial:",
            df["timestamp"].min().date(),
            min_value=df["timestamp"].min().date(),
            max_value=df["timestamp"].max().date()
        )
    with col2:
        end_date = st.date_input(
            "Data final:",
            df["timestamp"].max().date(),
            min_value=df["timestamp"].min().date(),
            max_value=df["timestamp"].max().date()
        )
    df_filtrado = df[
        (df["timestamp"].dt.date >= start_date) & 
        (df["timestamp"].dt.date <= end_date)
    ]
else:  # Última semana
    end_date = df["timestamp"].max().date()
    start_date = end_date - timedelta(days=7)
    df_filtrado = df[
        (df["timestamp"].dt.date >= start_date) & 
        (df["timestamp"].dt.date <= end_date)
    ]

# Filtro por hora
hour_range = st.sidebar.slider(
    "Filtrar por horário:",
    min_value=0,
    max_value=23,
    value=(0, 23),
    step=1
)
df_filtrado = df_filtrado[
    (df_filtrado["hour"] >= hour_range[0]) & 
    (df_filtrado["hour"] <= hour_range[1])
]

if df_filtrado.empty:
    st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

# Métricas principais
st.header("📊 Métricas Principais")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_pm25 = df_filtrado["PM2.5"].mean()
    quality, icon = classify_air_quality(avg_pm25)
    st.metric(
        label="PM2.5 Médio",
        value=f"{avg_pm25:.1f} µg/m³",
        delta=f"{avg_pm25 - df['PM2.5'].mean():.1f}"
    )
    st.write(f"{icon} {quality}")

with col2:
    max_pm25 = df_filtrado["PM2.5"].max()
    st.metric(
        label="PM2.5 Máximo",
        value=f"{max_pm25:.1f} µg/m³"
    )

with col3:
    avg_temp = df_filtrado["Temperature"].mean()
    st.metric(
        label="Temperatura Média",
        value=f"{avg_temp:.1f}°C"
    )

with col4:
    avg_humidity = df_filtrado["Humidity"].mean()
    st.metric(
        label="Umidade Média",
        value=f"{avg_humidity:.1f}%"
    )

st.markdown("---")

# Previsões do modelo
if len(df_filtrado) > 0:
    X_novo = df_filtrado[
        ["PM10", "CO", "NO2", "Temperature", "Humidity", "hour", "weekday"]
    ]
    y_previsto = model.predict(X_novo)
    df_filtrado = df_filtrado.copy()
    df_filtrado["PM2.5_pred"] = y_previsto
    
    # Calcular métricas do modelo
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(df_filtrado["PM2.5"], y_previsto)
    r2 = r2_score(df_filtrado["PM2.5"], y_previsto)

# Layout em duas colunas
col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("📈 Análise Temporal")
    
    # Gráfico principal com Plotly
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("PM2.5: Real vs Previsto", "Outros Poluentes"),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # PM2.5 real vs previsto
    fig.add_trace(
        go.Scatter(
            x=df_filtrado["timestamp"],
            y=df_filtrado["PM2.5"],
            name="Real",
            line=dict(color="blue", width=2)
        ),
        row=1, col=1
    )
    
    if len(df_filtrado) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_filtrado["timestamp"],
                y=df_filtrado["PM2.5_pred"],
                name="Previsto",
                line=dict(color="red", width=2, dash="dash")
            ),
            row=1, col=1
        )
    
    # Outros poluentes
    fig.add_trace(
        go.Scatter(
            x=df_filtrado["timestamp"],
            y=df_filtrado["PM10"],
            name="PM10",
            line=dict(color="orange")
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_filtrado["timestamp"],
            y=df_filtrado["NO2"],
            name="NO2",
            line=dict(color="green"),
            yaxis="y3"
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Monitoramento da Qualidade do Ar",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Tempo", row=2, col=1)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=1)
    fig.update_yaxes(title_text="Concentração", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.header("🎯 Análises")
    
    # Distribuição da qualidade do ar
    st.subheader("Distribuição da Qualidade")
    quality_counts = df_filtrado["quality"].value_counts()
    
    fig_pie = px.pie(
        values=quality_counts.values,
        names=quality_counts.index,
        title="Classificação da Qualidade do Ar"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Métricas do modelo
    if len(df_filtrado) > 0:
        st.subheader("Precisão do Modelo")
        st.metric("MAE", f"{mae:.2f} µg/m³")
        st.metric("R² Score", f"{r2:.3f}")
        
        # Correlação
        st.subheader("Correlação com PM2.5")
        correlations = df_filtrado[["PM10", "CO", "NO2", "Temperature", "Humidity"]].corrwith(df_filtrado["PM2.5"]).sort_values(ascending=False)
        
        for param, corr in correlations.items():
            st.write(f"**{param}**: {corr:.3f}")

# Análise por período
st.header("📅 Análise por Período")

col1, col2 = st.columns(2)

with col1:
    # Análise por hora do dia
    st.subheader("Variação por Hora")
    hourly_avg = df_filtrado.groupby("hour")["PM2.5"].mean().reset_index()
    
    fig_hour = px.bar(
        hourly_avg,
        x="hour",
        y="PM2.5",
        title="PM2.5 Médio por Hora do Dia",
        labels={"hour": "Hora", "PM2.5": "PM2.5 (µg/m³)"}
    )
    st.plotly_chart(fig_hour, use_container_width=True)

with col2:
    # Análise por dia da semana
    if len(df_filtrado["day_name"].unique()) > 1:
        st.subheader("Variação por Dia da Semana")
        daily_avg = df_filtrado.groupby("day_name")["PM2.5"].mean().reset_index()
        
        fig_day = px.bar(
            daily_avg,
            x="day_name",
            y="PM2.5",
            title="PM2.5 Médio por Dia da Semana",
            labels={"day_name": "Dia", "PM2.5": "PM2.5 (µg/m³)"}
        )
        st.plotly_chart(fig_day, use_container_width=True)

# Mapa de calor de correlações
st.header("🔥 Mapa de Correlações")
corr_matrix = df_filtrado[["PM2.5", "PM10", "CO", "NO2", "Temperature", "Humidity"]].corr()

fig_heatmap = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title="Correlação entre Parâmetros",
    color_continuous_scale="RdBu_r"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Dados brutos (opcional)
if st.checkbox("📋 Mostrar dados brutos"):
    st.subheader("Dados Filtrados")
    st.dataframe(
        df_filtrado[["timestamp", "PM2.5", "PM10", "CO", "NO2", "Temperature", "Humidity", "quality"]],
        use_container_width=True
    )
    
    # Botão de download
    csv = df_filtrado.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados como CSV",
        data=csv,
        file_name=f"dados_qualidade_ar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
