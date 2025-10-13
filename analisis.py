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

# Inicializar colorama
init(autoreset=True)

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

# Número de repeticiones (cada repetición baraja y hace split distinto)
n_reps = 10

# Cuántos procesos usar por modelo (configurable)
procs_rf = 12
procs_svm = 4
procs_knn = 8

# No pedir más procesos que CPUs disponibles
cpus = mp.cpu_count()
procs_rf = min(procs_rf, cpus)
procs_svm = min(procs_svm, cpus)
procs_knn = min(procs_knn, cpus)

print(Fore.MAGENTA + "\n[+] " + Style.RESET_ALL +
      f"Ejecutando {n_reps} repeticiones por modelo (cada iteración baraja los datos)...\n")

resultados = {}

# ----------------------------
# Inicializador para cada proceso del pool
# ----------------------------
def init_pool_for_model(X_shared, y_shared, params):
    """
    Guarda X,y y params en variables globales del proceso.
    """
    global X_global, y_global, params_global
    X_global = X_shared
    y_global = y_shared
    params_global = params

# ----------------------------
# Workers: en cada iteración se baraja y se hace split distinto
# ----------------------------
def rf_worker(seed):
    # Barajar y split distinto en cada seed. shuffle=True es explícito.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, shuffle=True, random_state=seed
    )
    model = RandomForestClassifier(**params_global, random_state=seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)        # evaluar en el holdout de esa iteración
    return accuracy_score(y_holdout, pred)

def svm_worker(seed):
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, shuffle=True, random_state=seed
    )
    model = make_pipeline(StandardScaler(), SVC(**params_global))
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)
    return accuracy_score(y_holdout, pred)

def knn_worker(seed):
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, shuffle=True, random_state=seed
    )
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(**params_global))
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)
    return accuracy_score(y_holdout, pred)

# ----------------------------
# Ejecutar en paralelo con multiprocessing.Pool
# ----------------------------
if __name__ == "__main__":
    # Random Forest pool
    with mp.Pool(processes=procs_rf,
                 initializer=init_pool_for_model,
                 initargs=(X, y, mejores_params_rf)) as pool:
        accs_rf = list(tqdm(pool.imap(rf_worker, range(n_reps)), total=n_reps, desc="Random Forest"))
    resultados['Random Forest'] = {
        'media': round(pd.Series(accs_rf).mean(), 4),
        'varianza': round(pd.Series(accs_rf).var(), 6),
        'desviacion': round(pd.Series(accs_rf).std(), 6)
    }

    # SVM pool
    with mp.Pool(processes=procs_svm,
                 initializer=init_pool_for_model,
                 initargs=(X, y, mejores_params_svm)) as pool:
        accs_svm = list(tqdm(pool.imap(svm_worker, range(n_reps)), total=n_reps, desc="SVM"))
    resultados['SVM'] = {
        'media': round(pd.Series(accs_svm).mean(), 4),
        'varianza': round(pd.Series(accs_svm).var(), 6),
        'desviacion': round(pd.Series(accs_svm).std(), 6)
    }

    # KNN pool
    with mp.Pool(processes=procs_knn,
                 initializer=init_pool_for_model,
                 initargs=(X, y, mejores_params_knn)) as pool:
        accs_knn = list(tqdm(pool.imap(knn_worker, range(n_reps)), total=n_reps, desc="KNN"))
    resultados['KNN'] = {
        'media': round(pd.Series(accs_knn).mean(), 4),
        'varianza': round(pd.Series(accs_knn).var(), 6),
        'desviacion': round(pd.Series(accs_knn).std(), 6)
    }

    # Resultados finales
    print(Fore.GREEN + "\n=== Resultados Finales ===" + Style.RESET_ALL)
    for modelo, stats in resultados.items():
        print(f"\n{modelo}:")
        print(f"  Media: {stats['media']}")
        print(f"  Varianza: {stats['varianza']}")
        print(f"  Desviación estándar: {stats['desviacion']}")
