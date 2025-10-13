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
print("Leyendo archivo de prueba para evaluación final...")

df = pd.read_csv('./Fashion_MNIST/fashion-mnist_test.csv')
X = df.drop(columns='label')
y = df['label']

# Mejores hiperparámetros encontrados (sin n_jobs)
mejores_params_rf = {'max_depth': None, 'n_estimators': 250,
                     'min_samples_split': 2, 'min_samples_leaf': 1}
mejores_params_svm = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
mejores_params_knn = {'n_neighbors': 5, 'metric': 'manhattan'}

# Número de repeticiones
n_reps = 10

# Cuántos procesos usar por modelo (configurable)
procs_rf = 12
procs_svm = 4
procs_knn = 8

print(Fore.MAGENTA + "\n[+] " + Style.RESET_ALL +
      f"Ejecutando {n_reps} repeticiones por modelo...\n")

resultados = {}

# ----------------------------
# Helpers para pools (globales en cada proceso)
# ----------------------------
def init_pool_for_model(X_shared, y_shared, model_name, params):
    """
    Inicializador para cada proceso del pool.
    Guarda X, y, params y model_name en variables globales del proceso.
    """
    global X_global, y_global, model_name_global, params_global
    X_global = X_shared
    y_global = y_shared
    model_name_global = model_name
    params_global = params

# Worker para Random Forest
def rf_worker(seed):
    # usa X_global, y_global, params_global
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, random_state=seed
    )
    model = RandomForestClassifier(**params_global, random_state=seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)
    return accuracy_score(y_holdout, pred)

# Worker para SVM (pipeline con scaler)
def svm_worker(seed):
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, random_state=seed
    )
    model = make_pipeline(StandardScaler(), SVC(**params_global))
    model.fit(X_train, y_train)
    pred = model.predict(X_holdout)
    return accuracy_score(y_holdout, pred)

# Worker para KNN (pipeline con scaler)
def knn_worker(seed):
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_global, y_global, train_size=0.8, stratify=y_global, random_state=seed
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
                 initargs=(X, y, 'rf', mejores_params_rf)) as pool:
        accs_rf = list(tqdm(pool.imap(rf_worker, range(n_reps)), total=n_reps, desc="Random Forest"))
    resultados['Random Forest'] = {
        'media': round(pd.Series(accs_rf).mean(), 4),
        'varianza': round(pd.Series(accs_rf).var(), 6),
        'desviacion': round(pd.Series(accs_rf).std(), 6)
    }

    # SVM pool
    with mp.Pool(processes=procs_svm,
                 initializer=init_pool_for_model,
                 initargs=(X, y, 'svm', mejores_params_svm)) as pool:
        accs_svm = list(tqdm(pool.imap(svm_worker, range(n_reps)), total=n_reps, desc="SVM"))
    resultados['SVM'] = {
        'media': round(pd.Series(accs_svm).mean(), 4),
        'varianza': round(pd.Series(accs_svm).var(), 6),
        'desviacion': round(pd.Series(accs_svm).std(), 6)
    }

    # KNN pool
    with mp.Pool(processes=procs_knn,
                 initializer=init_pool_for_model,
                 initargs=(X, y, 'knn', mejores_params_knn)) as pool:
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
