import os
import sys
import time
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style, init
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

# Inicializar colorama
init(autoreset=True)

# ----------------------------
# Crear carpeta de salida
# ----------------------------
os.makedirs("salida", exist_ok=True)

# Redirigir salida de consola a archivo
log_file = open("salida/resultados_consola.txt", "w", encoding="utf-8")
sys.stdout = log_file

# ----------------------------
print(Fore.CYAN + "\n[+] " + Style.RESET_ALL, end='')
print("Leyendo archivo de entrenamiento...")

df = pd.read_csv('./archive/fashion-mnist_train.csv')
X = df.drop(columns='label')
y = df['label']

# Mejores hiperparámetros (sin n_jobs)
mejores_params_rf = {'max_depth': None, 'n_estimators': 250,
                     'min_samples_split': 2, 'min_samples_leaf': 1}
mejores_params_svm = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
mejores_params_knn = {'n_neighbors': 5, 'metric': 'manhattan'}

# Número de repeticiones
n_reps = 10

# Procesos por modelo
procs_rf = 12
procs_svm = 4
procs_knn = 8

# No pedir más procesos que CPUs disponibles
cpus = mp.cpu_count()
procs_rf = min(procs_rf, cpus)
procs_svm = min(procs_svm, cpus)
procs_knn = min(procs_knn, cpus)

print(Fore.MAGENTA + "\n[+] " + Style.RESET_ALL +
      f"Ejecutando {n_reps} repeticiones por modelo...\n")

resultados = {}
df_detalle = pd.DataFrame(columns=["modelo", "repeticion", "accuracy", "tiempo_seg"])

# ----------------------------
# Inicializador para pool
# ----------------------------
def init_pool_for_model(X_shared, y_shared, params):
    global X_global, y_global, params_global
    X_global = X_shared
    y_global = y_shared
    params_global = params

# ----------------------------
# Workers con medición de tiempo
# ----------------------------
def rf_worker(seed):
    start = time.time()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, shuffle=True, random_state=seed
    )
    model = RandomForestClassifier(**params_global, random_state=seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)
    acc = accuracy_score(y_holdout, pred)
    return acc, time.time() - start

def svm_worker(seed):
    start = time.time()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, shuffle=True, random_state=seed
    )
    model = make_pipeline(StandardScaler(), SVC(**params_global))
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)
    acc = accuracy_score(y_holdout, pred)
    return acc, time.time() - start

def knn_worker(seed):
    start = time.time()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, shuffle=True, random_state=seed
    )
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(**params_global))
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)
    acc = accuracy_score(y_holdout, pred)
    return acc, time.time() - start

# ----------------------------
# Ejecutar en paralelo
# ----------------------------
if __name__ == "__main__":
    for modelo, worker, procs, params in [
        ("Random Forest", rf_worker, procs_rf, mejores_params_rf),
        ("SVM", svm_worker, procs_svm, mejores_params_svm),
        ("KNN", knn_worker, procs_knn, mejores_params_knn),
    ]:
        print(Fore.CYAN + f"\n[+] Ejecutando {modelo}...\n" + Style.RESET_ALL)
        with mp.Pool(processes=procs,
                     initializer=init_pool_for_model,
                     initargs=(X, y, params)) as pool:
            results = list(tqdm(pool.imap(worker, range(n_reps)), total=n_reps, desc=modelo))
        
        # Separar accuracy y tiempo
        accs, tiempos = zip(*results)
        
        # Guardar resultados agregados
        resultados[modelo] = {
            'media': round(pd.Series(accs).mean(), 4),
            'varianza': round(pd.Series(accs).var(), 6),
            'desviacion': round(pd.Series(accs).std(), 6),
            'tiempo_total': round(sum(tiempos), 2)
        }
        
        # Guardar detalle por repetición
        for i, (acc, t) in enumerate(results):
            df_detalle = pd.concat([df_detalle, pd.DataFrame({
                "modelo":[modelo], "repeticion":[i], "accuracy":[acc], "tiempo_seg":[t]
            })], ignore_index=True)

    # Guardar CSV con detalle
    df_detalle.to_csv("salida/resultados_detalle.csv", index=False)

    # ----------------------------
    # Graficas Accuracy
    # ----------------------------
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,6))
    sns.boxplot(x="modelo", y="accuracy", data=df_detalle)
    plt.title("Distribución de Accuracy por Modelo")
    plt.savefig("salida/boxplot_accuracy.png")
    plt.close()

    plt.figure(figsize=(10,6))
    sns.lineplot(x="repeticion", y="accuracy", hue="modelo", marker="o", data=df_detalle)
    plt.title("Accuracy por Repetición y Modelo")
    plt.savefig("salida/lineplot_accuracy.png")
    plt.close()

    # ----------------------------
    # Graficas Tiempo
    # ----------------------------
    plt.figure(figsize=(10,6))
    sns.lineplot(x="repeticion", y="tiempo_seg", hue="modelo", marker="o", data=df_detalle)
    plt.title("Tiempo por Repetición y Modelo")
    plt.ylabel("Tiempo (segundos)")
    plt.savefig("salida/lineplot_tiempo.png")
    plt.close()

    # ----------------------------
    # Resultados finales impresos
    # ----------------------------
    print(Fore.GREEN + "\n=== Resultados Finales ===" + Style.RESET_ALL)
    for modelo, stats in resultados.items():
        print(f"\n{modelo}:")
        print(f"  Media Accuracy: {stats['media']}")
        print(f"  Varianza: {stats['varianza']}")
        print(f"  Desviación estándar: {stats['desviacion']}")
        print(f"  Tiempo total: {stats['tiempo_total']} seg")

    print(Fore.CYAN + "\n[+] Resultados guardados en la carpeta 'salida'.")

    # ----------------------------
    # Restaurar salida de consola
    # ----------------------------
    sys.stdout = sys.__stdout__
    log_file.close()
