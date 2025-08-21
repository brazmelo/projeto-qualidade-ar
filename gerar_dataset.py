import pandas as pd
import numpy as np

np.random.seed(42)

# Gerar datas e horas (1 mês de leituras horárias)
timestamps = pd.date_range(start="2025-08-01", end="2025-08-31 23:00:00", freq="H")

# Simular sensores
pm2_5 = np.random.normal(loc=35, scale=15, size=len(timestamps))  # µg/m³
pm10 = pm2_5 + np.random.normal(0, 10, size=len(timestamps))
co = np.random.normal(0.5, 0.2, size=len(timestamps))  # ppm
no2 = np.random.normal(20, 5, size=len(timestamps))
temperature = np.random.normal(25, 5, size=len(timestamps))
humidity = np.random.normal(60, 10, size=len(timestamps))

df = pd.DataFrame(
    {
        "timestamp": timestamps,
        "PM2.5": np.clip(pm2_5, 0, None),
        "PM10": np.clip(pm10, 0, None),
        "CO": np.clip(co, 0, None),
        "NO2": np.clip(no2, 0, None),
        "Temperature": temperature,
        "Humidity": humidity,
    }
)

# Criar pasta dados se não existir
import os

os.makedirs("dados", exist_ok=True)

df.to_csv("dados/dados_iot.csv", index=False)
print("Dataset simulado criado: dados/dados_iot.csv")
