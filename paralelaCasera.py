# Librerías
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import time


# =========================== CONFIGURACIÓN ====================================
# Define la cantidad de procesadores que se usarán para el cómputo en paralelo
nProcesos = 8

# ======================= FUNCIONES DE EVALUACIÓN ==============================
def evaluarRf(args):
    """Evalúa una combinación de hiperparámetros para Random Forest."""
    parametros, xTrain, xTest, yTrain, yTest = args
    rf = RandomForestClassifier(random_state=42, n_jobs=1, **parametros)
    rf.fit(xTrain, yTrain)
    prediccion = rf.predict(xTest)
    
    # Devuelve un diccionario con todas las métricas
    metricas = {
        'accuracy': accuracy_score(yTest, prediccion),
        'f1_score': f1_score(yTest, prediccion, average='macro', zero_division=0),
        'precision': precision_score(yTest, prediccion, average='macro', zero_division=0),
        'recall': recall_score(yTest, prediccion, average='macro', zero_division=0)
    }
    return (parametros, metricas)

def evaluarSvm(args):
    """Evalúa una combinación de hiperparámetros para SVM."""
    parametros, xTrain, xTest, yTrain, yTest = args
    metricas_cero = {'accuracy': 0, 'f1_score': 0, 'precision': 0, 'recall': 0}
    try:
        svm = SVC(**parametros)
        svm.fit(xTrain, yTrain)
        prediccion = svm.predict(xTest)
        metricas = {
            'accuracy': accuracy_score(yTest, prediccion),
            'f1_score': f1_score(yTest, prediccion, average='macro', zero_division=0),
            'precision': precision_score(yTest, prediccion, average='macro', zero_division=0),
            'recall': recall_score(yTest, prediccion, average='macro', zero_division=0)
        }
        return (parametros, metricas)
    except Exception as e:
        return (parametros, metricas_cero)

def evaluarKnn(args):
    """Evalúa una combinación de hiperparámetros para KNN."""
    parametros, xTrain, xTest, yTrain, yTest = args
    knn = KNeighborsClassifier(n_jobs=1, **parametros)
    knn.fit(xTrain, yTrain)
    prediccion = knn.predict(xTest)
    metricas = {
        'accuracy': accuracy_score(yTest, prediccion),
        'f1_score': f1_score(yTest, prediccion, average='macro', zero_division=0),
        'precision': precision_score(yTest, prediccion, average='macro', zero_division=0),
        'recall': recall_score(yTest, prediccion, average='macro', zero_division=0)
    }
    return (parametros, metricas)

# ========================= INICIO DEL PROGRAMA ================================
if __name__ == '__main__':
    nombreArchivo = './archive/fashion-mnist_train.csv'
    
    print(f"Leyendo archivo: {nombreArchivo}...")
    try:
        df = pd.read_csv(f'./{nombreArchivo}')
        print("Archivo leído correctamente.\n")
    except FileNotFoundError:
        print(f"Error: El archivo '{nombreArchivo}' no se encontró.")
        exit()

    x = df.drop(columns='label')
    y = df['label']
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=0.8, random_state=42)
    
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0
    """
    # ========= 1. BÚSQUEDA PARA RANDOM FOREST =========
    print("Iniciando búsqueda para Random Forest...")
    print("Iniciando búsqueda para KNN...")

    hiperparametrosRf = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 4]
    }
    
    tareasRf = []
    print("Generando combinaciones de hiperparámetros para RF...")
    for nEstimators in hiperparametrosRf['n_estimators']:
        for maxDepth in hiperparametrosRf['max_depth']:
            for minSamplesSplit in hiperparametrosRf['min_samples_split']:
                for minSamplesLeaf in hiperparametrosRf['min_samples_leaf']:
                    parametros = {
                        'n_estimators': nEstimators,
                        'max_depth': maxDepth,
                        'min_samples_split': minSamplesSplit,
                        'min_samples_leaf': minSamplesLeaf
                    }
                    tareasRf.append((parametros, xTrain, xTest, yTrain, yTest))

    inicioRf = time.time()
    print(f"Iniciando cómputo en paralelo con {len(tareasRf)} tareas...")
    with Pool(processes=nProcesos) as pool:
        resultadosRf = list(tqdm(pool.imap(evaluarRf, tareasRf), total=len(tareasRf)))
    finRf = time.time()
    
    # Se elige el mejor modelo basado en el accuracy
    mejorRf = max(resultadosRf, key=lambda item: item[1]['accuracy'])
    
    print("\n--- RESULTADOS RANDOM FOREST ---")
    print(f"Mejores Hiperparámetros: {mejorRf[0]}")
    
    print("\nMétricas del Mejor Modelo:")
    for metrica, valor in mejorRf[1].items():
        print(f"  - {metrica.replace('_', ' ').capitalize()}: {valor:.4f}")
        
    print(f"\nTiempo total: {round(finRf - inicioRf, 2)} segundos")
    print("--------------------------------\n")

    """
    #"""
    # ========= 2. BÚSQUEDA PARA SVM =========
    print("Iniciando búsqueda para SVM...")

    hiperparametrosSvm = {
        'kernel': ['rbf', 'poly'],
        'C': [0.1, 1, 5,10],
        'gamma': ['scale', 'auto']
    }

    tareasSvm = []
    print("Generando combinaciones de hiperparámetros para SVM...")
    for kernel in hiperparametrosSvm['kernel']:
        for c in hiperparametrosSvm['C']:
            for gamma in hiperparametrosSvm['gamma']:
                parametros = {
                    'kernel': kernel,
                    'C': c,
                    'gamma': gamma
                }
                tareasSvm.append((parametros, xTrain, xTest, yTrain, yTest))

    inicioSvm = time.time()
    print(f"Iniciando cómputo en paralelo con {len(tareasSvm)} tareas...")
    with Pool(processes=nProcesos) as pool:
        resultadosSvm = list(tqdm(pool.imap(evaluarSvm, tareasSvm), total=len(tareasSvm)))
    finSvm = time.time()

    mejorSvm = max(resultadosSvm, key=lambda item: item[1]['accuracy'])
    print("\n--- RESULTADOS SVM ---")
    print(f"Mejores Hiperparámetros: {mejorSvm[0]}")
    
    print("\nMétricas del Mejor Modelo:")
    for metrica, valor in mejorSvm[1].items():
        print(f"  - {metrica.replace('_', ' ').capitalize()}: {valor:.4f}")
        
    print(f"\nTiempo total: {round(finSvm - inicioSvm, 2)} segundos")
    print("----------------------\n")
    #"""

    """
    # ========= 3. BÚSQUEDA PARA KNN =========
    print("Iniciando búsqueda para KNN...")

    hiperparametrosKnn = {
        'n_neighbors': [3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    tareasKnn = []
    print("Generando combinaciones de hiperparámetros para KNN...")
    for nNeighbors in hiperparametrosKnn['n_neighbors']:
        for weights in hiperparametrosKnn['weights']:
            for metric in hiperparametrosKnn['metric']:
                parametros = {
                    'n_neighbors': nNeighbors,
                    'weights': weights,
                    'metric': metric
                }
                tareasKnn.append((parametros, xTrain, xTest, yTrain, yTest))
    
    inicioKnn = time.time()
    print(f"Iniciando cómputo en paralelo con {len(tareasKnn)} tareas...")
    with Pool(processes=nProcesos) as pool:
        resultadosKnn = list(tqdm(pool.imap(evaluarKnn, tareasKnn), total=len(tareasKnn)))
    finKnn = time.time()

    mejorKnn = max(resultadosKnn, key=lambda item: item[1]['accuracy'])
    print("\n--- RESULTADOS KNN ---")
    print(f"Mejores Hiperparámetros: {mejorKnn[0]}")
    
    print("\nMétricas del Mejor Modelo:")
    for metrica, valor in mejorKnn[1].items():
        print(f"  - {metrica.replace('_', ' ').capitalize()}: {valor:.4f}")
        
    print(f"\nTiempo total: {round(finKnn - inicioKnn, 2)} segundos")
    print("----------------------\n")
    """