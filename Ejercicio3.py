import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carga los datos 
data = pd.read_csv('boston_housing.csv')
X = data.drop('MEDV', axis=1)  
y = data['MEDV']  

# Divide los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Itera sobre diferentes valores de k
k_values = range(1, 11)
mse_values = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse_values.append(mean_squared_error(y_test, y_pred))

# Visualiza los resultados
plt.plot(k_values, mse_values, marker='o')
plt.title('MSE vs K')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')
plt.show()
