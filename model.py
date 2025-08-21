import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Carregar dados
df = pd.read_csv("dados/dados_iot.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Criar features
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

X = df[["PM10", "CO", "NO2", "Temperature", "Humidity", "hour", "weekday"]]
y = df["PM2.5"]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar modelo
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"RMSE do modelo: {rmse:.2f}")

# Salvar modelo
joblib.dump(model, "dados/modelo_rf.pkl")
print("Modelo salvo: dados/modelo_rf.pkl")
