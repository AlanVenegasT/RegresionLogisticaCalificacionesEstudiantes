import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar datos
data = pd.read_csv('datos.txt', delimiter='\t', header=None)
X = data.iloc[:, :-1]  # Calificaciones (2 primeras columnas)
y = data.iloc[:, -1]   # Etiqueta (aprobado/reprobado)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Obtener tres pares de calificaciones mínimas para aprobar
import numpy as np

# Generamos un rango de calificaciones hipotéticas para ver cuáles aprueban
min_grades = np.array([[70, 60], [60, 70], [50, 75]])
predictions = model.predict(min_grades)

for i, pred in enumerate(predictions):
    print(f"Calificaciones {min_grades[i]} = {'Aprobado' if pred == 1 else 'Reprobado'}")
