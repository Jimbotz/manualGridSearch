#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import time
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
NUM_WORKERS = 16        # núcleos a usar (procesos paralelos)
N_COMBINATIONS = 10      # combinaciones por modelo
SVM_TRAIN_SUBSET = 20000 # subset para entrenar SVM (para que no tarde horas)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Limitar hilos BLAS globalmente (cada proceso usará 1 hilo)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ----------------------------
# 1) CARGA Y PREPROCESAMIENTO
# ----------------------------
print("Cargando dataset Fashion-MNIST desde CSV...")
t0_global = time.time()

train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

X_train = train_df.drop('label', axis=1).values.astype(np.float32)
y_train = train_df['label'].values.astype(int)

X_temp = test_df.drop('label', axis=1).values.astype(np.float32)
y_temp = test_df['label'].values.astype(int)

# Dividir test en validación y prueba
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=RANDOM_SEED)

# Normalizar y escalar
X_train /= 255.0
X_val /= 255.0
X_test /= 255.0

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Subconjunto para SVM
if SVM_TRAIN_SUBSET < len(X_train):
    idx_small = np.random.RandomState(RANDOM_SEED).choice(len(X_train), SVM_TRAIN_SUBSET, replace=False)
    X_train_small = X_train[idx_small]
    y_train_small = y_train[idx_small]
else:
    X_train_small = X_train
    y_train_small = y_train

t_prep = time.time() - t0_global
print(f"Preparación de datos completada en {t_prep:.2f}s — subconjunto SVM: {len(X_train_small)} muestras")

# ----------------------------
# 2) GENERADORES DE HIPERPARÁMETROS
# ----------------------------
def sample_params_rf():
    return {
        'n_estimators': int(np.random.randint(250, 340)),   # más centrado cerca de 300
        'max_depth': np.random.choice([35, 40, 45, 50]),
        'min_samples_split': int(np.random.randint(4, 7)),  # rango estrecho
        'min_samples_leaf': int(np.random.randint(3, 6)),   # 3–5
        'max_features': 'sqrt'                              # ya comprobado el mejor
    }


def sample_params_svm():
    return {
        'C': float(10 ** np.random.uniform(0.0, 0.3)),   # ~1–2
        'kernel': 'poly',                                 # se fija el kernel ganador
        'gamma': float(10 ** np.random.uniform(-2.2, -1.5))  # ~0.006–0.03
    }


def sample_params_knn():
    return {
        'n_neighbors': int(np.random.choice([7, 9, 10, 12])),
        'weights': 'distance',
        'p': 1
    }

# ----------------------------
# 3) FUNCIÓN DE ENTRENAMIENTO
# ----------------------------
def train_evaluate(args):
    model_name, params, combo_idx = args

    # asegurar single-threading dentro del proceso
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if model_name == 'rf':
        model = RandomForestClassifier(**params, random_state=RANDOM_SEED, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = float(accuracy_score(y_val, preds))
        return {'model': 'rf', 'combo': combo_idx, 'acc_val': acc, 'params': params}

    elif model_name == 'svm':
        model = SVC(**params, random_state=RANDOM_SEED)
        model.fit(X_train_small, y_train_small)
        preds = model.predict(X_val)
        acc = float(accuracy_score(y_val, preds))
        return {'model': 'svm', 'combo': combo_idx, 'acc_val': acc, 'params': params}

    elif model_name == 'knn':
        model = KNeighborsClassifier(**params, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = float(accuracy_score(y_val, preds))
        return {'model': 'knn', 'combo': combo_idx, 'acc_val': acc, 'params': params}

# ----------------------------
# 4) EJECUCIÓN DE EXPERIMENTOS
# ----------------------------
def run_experiment(model_name, sampler):
    print(f"\n>>> Ejecutando {model_name.upper()} con {N_COMBINATIONS} combinaciones ({NUM_WORKERS} workers)...")
    params_list = [sampler() for _ in range(N_COMBINATIONS)]
    for i, p in enumerate(params_list, 1):
        print(f"  Combinación {i}: {p}")

    args_list = [(model_name, params_list[i], i + 1) for i in range(len(params_list))]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        results = list(ex.map(train_evaluate, args_list))
    t1 = time.time()

    results = sorted(results, key=lambda r: r['combo'])
    print(f"\nResultados {model_name.upper()}:")
    for r in results:
        print(f"  Combo {r['combo']:2d} -> acc_val = {r['acc_val']:.4f} | params: {r['params']}")

    best = max(results, key=lambda r: r['acc_val'])
    print(f"\n>>> Mejor combinación para {model_name.upper()}: acc_val={best['acc_val']:.4f}, params={best['params']}")
    print(f"Tiempo paralelo: {t1 - t0:.2f}s")
    return best, t1 - t0

# ----------------------------
# 5) SECUENCIA PRINCIPAL
# ----------------------------
if __name__ == "__main__":
    rf_best, t_rf = run_experiment('rf', sample_params_rf)
    svm_best, t_svm = run_experiment('svm', sample_params_svm)
    knn_best, t_knn = run_experiment('knn', sample_params_knn)

    # Evaluación final
    print("\nEvaluando mejores modelos en conjunto de prueba...")
    models_final = {
        'rf': RandomForestClassifier(**rf_best['params'], random_state=RANDOM_SEED, n_jobs=NUM_WORKERS),
        'svm': SVC(**svm_best['params'], random_state=RANDOM_SEED),
        'knn': KNeighborsClassifier(**knn_best['params'], n_jobs=NUM_WORKERS)
    }

    results_final = []
    for name, model in models_final.items():
        print(f"\nEntrenando {name.upper()} final...")
        t0 = time.time()
        if name == 'svm':
            model.fit(X_train_small, y_train_small)
        else:
            model.fit(X_train, y_train)
        t1 = time.time()
        preds = model.predict(X_test)
        acc_test = accuracy_score(y_test, preds)
        results_final.append((name, acc_test, t1 - t0))
        print(f"{name.upper()} -> Accuracy test: {acc_test:.4f} | Tiempo entrenamiento: {t1 - t0:.2f}s")

    best_overall = max(results_final, key=lambda x: x[1])
    print("\n================= RESUMEN FINAL =================")
    print(f"Random Forest -> acc_val={rf_best['acc_val']:.4f} | acc_test={results_final[0][1]:.4f} | t_par={t_rf:.2f}s")
    print(f"SVM           -> acc_val={svm_best['acc_val']:.4f} | acc_test={results_final[1][1]:.4f} | t_par={t_svm:.2f}s")
    print(f"KNN           -> acc_val={knn_best['acc_val']:.4f} | acc_test={results_final[2][1]:.4f} | t_par={t_knn:.2f}s")
    print("-------------------------------------------------")
    print(f"🏆 Mejor modelo global: {best_overall[0].upper()} con accuracy test = {best_overall[1]:.4f}")
    print("=================================================")
