import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos California Housing
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Comparar con y sin normalización
# KNN sin normalización
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# KNN con normalización
knn.fit(X_train_scaled, y_train)
y_pred_scaled = knn.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)

# Mostrar resultados
print(f'Error cuadrático medio sin normalización: {mse:.4f}')
print(f'Error cuadrático medio con normalización: {mse_scaled:.4f}')
