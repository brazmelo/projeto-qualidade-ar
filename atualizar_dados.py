import pandas as pd
import numpy as np
from datetime import datetime, timedelta

arquivo_csv = "dados/dados_iot.csv"
data_hoje = datetime.now().date()

novos_dados = []
for hora in range(24):
    timestamp = datetime.combine(data_hoje, datetime.min.time()) + timedelta(hours=hora)
    linha = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        np.random.uniform(10, 70),  # PM2.5
        np.random.uniform(10, 80),  # PM10
        np.random.uniform(0.1, 1.0),  # CO
        np.random.uniform(10, 40),  # NO2
        np.random.uniform(15, 35),  # Temperature
        np.random.uniform(40, 80),  # Humidity
    ]
    novos_dados.append(linha)

df_novo = pd.DataFrame(
    novos_dados,
    columns=["timestamp", "PM2.5", "PM10", "CO", "NO2", "Temperature", "Humidity"],
)
df_novo.to_csv(arquivo_csv, mode="a", header=False, index=False)

print("Dados de hoje adicionados ao arquivo!")
