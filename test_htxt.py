# Prueba de los hiperparametros encontrados en el conjunto de prueba

import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ruta del CSV de test
csv_path = './archive/fashion-mnist_test.csv'

# Cargar de datos
data = pd.read_csv(csv_path)

# Separar características y etiqueta
X = data.drop('label', axis=1).values
y = data['label'].values

# Escalador para KNN y SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Función para calcular métricas
def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

# Random Forest 
rf_params = {'n_estimators': 250, 'max_depth': None,
             'min_samples_split': 2, 'min_samples_leaf': 1,
             'random_state': 42}

rf = RandomForestClassifier(**rf_params)

start_time = time.time()
rf.fit(X, y)
y_pred_rf = rf.predict(X)
rf_time = time.time() - start_time
rf_metrics = compute_metrics(y, y_pred_rf)

print("\n\t-> RANDOM FOREST ")
print(f"Tiempo (s): {rf_time:.2f}")
print(f"Accuracy: {rf_metrics['accuracy']:.4f}")
print(f"Precision: {rf_metrics['precision']:.4f}")
print(f"Recall: {rf_metrics['recall']:.4f}")
print(f"F1: {rf_metrics['f1']:.4f}")

# KNN 
knn_params = {'n_neighbors': 5, 'weights': 'distance', 'metric': 'manhattan'}
knn = KNeighborsClassifier(**knn_params)

start_time = time.time()
knn.fit(X_scaled, y)
y_pred_knn = knn.predict(X_scaled)
knn_time = time.time() - start_time
knn_metrics = compute_metrics(y, y_pred_knn)

print("\n\t-> KNN ")
print(f"Tiempo (s): {knn_time:.2f}")
print(f"Accuracy: {knn_metrics['accuracy']:.4f}")
print(f"Precision: {knn_metrics['precision']:.4f}")
print(f"Recall: {knn_metrics['recall']:.4f}")
print(f"F1: {knn_metrics['f1']:.4f}")

# SVM 
svm_params = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
svm = SVC(**svm_params)

start_time = time.time()
svm.fit(X_scaled, y)
y_pred_svm = svm.predict(X_scaled)
svm_time = time.time() - start_time
svm_metrics = compute_metrics(y, y_pred_svm)

print("\n\t-> SVM ")
print(f"Tiempo (s): {svm_time:.2f}")
print(f"Accuracy: {svm_metrics['accuracy']:.4f}")
print(f"Precision: {svm_metrics['precision']:.4f}")
print(f"Recall: {svm_metrics['recall']:.4f}")
print(f"F1: {svm_metrics['f1']:.4f}")
