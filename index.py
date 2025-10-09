import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --- 1. Cargar los Datos ---
try:
    # Ajusta la ruta si tus archivos están en un subdirectorio como 'archive'
    train_df = pd.read_csv('./archive/fashion-mnist_train.csv')
    test_df = pd.read_csv('./archive/fashion-mnist_test.csv')
    print("✓ Archivos CSV cargados correctamente.")
except FileNotFoundError:
    print("Error: No se encontraron los archivos CSV. Asegúrate de que la ruta sea correcta.")
    exit()

# --- 2. Preparar los Datos ---
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Normalizamos los datos
X_train = X_train / 255.0
X_test = X_test / 255.0
print("Datos preparados y normalizados.")

# Mapeo de etiquetas a nombres para los reportes
label_map = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

# -----------------------------------------------------------------------------
# --- SECCIÓN OPCIONAL: RANDOM FOREST ---
# Para ejecutar esta sección, elimina las triples comillas de abajo y arriba.
# -----------------------------------------------------------------------------
"""
print("\n--- INICIANDO BÚSQUEDA PARA RANDOM FOREST ---")
param_grid_rf = {
    'n_estimators': [100, 150],
    'max_depth': [20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train) # Usando todos los datos
print("\n✓ Mejores hiperparámetros para Random Forest:", grid_search_rf.best_params_)

# Evaluar y visualizar RF
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("\n--- Reporte de Clasificación (Prueba) con Random Forest ---")
print(classification_report(y_test, y_pred_rf, target_names=label_map.values()))
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Etiqueta Verdadera'); plt.xlabel('Etiqueta Predicha')
plt.savefig('matrizDeConfusionRF.png')
print("\n✓ Matriz de confusión de Random Forest guardada.")
"""

# -----------------------------------------------------------------------------
# --- SECCIÓN OPCIONAL: SUPPORT VECTOR MACHINE (SVM) ---
# ¡ADVERTENCIA: EXTREMADAMENTE LENTO CON TODOS LOS DATOS!
# Para ejecutar esta sección, elimina las triples comillas de abajo y arriba.
# -----------------------------------------------------------------------------
"""
print("\n--- INICIANDO BÚSQUEDA PARA SVM ---")
param_grid_svm = {
    'C': [1, 10],
    'gamma': [0.1, 0.01],
    'kernel': ['rbf'] # aca tambien se deberian probar otros kernels como 'linear', 'poly', etc. pero me dio cosa pq tardaba demasiado 
}
svm = SVC(random_state=42)
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=3, n_jobs=-1, verbose=2)
grid_search_svm.fit(X_train, y_train) # Usando todos los datos
print("\n✓ Mejores hiperparámetros para SVM:", grid_search_svm.best_params_)

# Evaluar y visualizar SVM
best_svm_model = grid_search_svm.best_estimator_
y_pred_svm = best_svm_model.predict(X_test)
print("\n--- Reporte de Clasificación (Prueba) con SVM ---")
print(classification_report(y_test, y_pred_svm, target_names=label_map.values()))
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='viridis', xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title('Matriz de Confusión - SVM')
plt.ylabel('Etiqueta Verdadera'); plt.xlabel('Etiqueta Predicha')
plt.savefig('matrizDeConfusionSVM.png')
print("\n✓ Matriz de confusión de SVM guardada.")
"""
# -----------------------------------------------------------------------------
# --- SECCIÓN PRINCIPAL: K-NEAREST NEIGHBORS (K-NN) ---
# ¡ADVERTENCIA: MUY LENTO CON TODOS LOS DATOS!
# -----------------------------------------------------------------------------
#"""
print(f"\n--- INICIANDO BÚSQUEDA PARA K-NN ---")
# Definimos el rango de 'k' (n_neighbors) que queremos probar
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11] # Rango reducido para acelerar un poco
}

# Inicializamos el clasificador K-NN
knn = KNeighborsClassifier()

# Configuramos la búsqueda en cuadrícula
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=3, n_jobs=-1, verbose=2)
print("\nIniciando la búsqueda del mejor 'k' para K-NN... (Esto puede tardar varias horas)")

# --- Ejecutar la Búsqueda ---
grid_search_knn.fit(X_train, y_train) # Usando todos los datos

# Imprimimos el mejor parámetro encontrado
print("\n✓ ¡Búsqueda completada!")
print("El mejor número de vecinos (k) para K-NN es:")
print(grid_search_knn.best_params_)

# --- Evaluar el Mejor Modelo K-NN ---
best_knn_model = grid_search_knn.best_estimator_
y_pred = best_knn_model.predict(X_test)

# Imprimimos el reporte de clasificación
print("\n--- Reporte de Clasificación (Prueba) con K-NN ---")
print(classification_report(y_test, y_pred, target_names=label_map.values()))

# --- Visualizar la Matriz de Confusión para K-NN ---
cm = confusion_matrix(y_test, y_pred)
class_names = list(label_map.values())

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión - K-NN en Fashion-MNIST')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Guardamos la nueva imagen
plt.savefig('matrizDeConfusionKNN.png')
print("\n✓ Matriz de confusión de K-NN guardada como 'matrizDeConfusionKNN.png'")
#"""



""" SI es que quieren probar a usar todos los parametros de SVM esta seria la manera correcta
# Ejemplo de un param_grid para probar múltiples kernels
param_grid_svm_completo = [
    {
        'kernel': ['rbf'], 
        'C': [1, 10], 
        'gamma': [0.1, 0.01]
    },
    {
        'kernel': ['linear'], 
        'C': [1, 10]
    },
    {
        'kernel': ['poly'], 
        'C': [1, 10], 
        'degree': [2, 3] # Grado del polinomio
    }
]

# Luego lo pasarías a GridSearchCV
# grid_search = GridSearchCV(svm, param_grid_svm_completo, ...)

"""