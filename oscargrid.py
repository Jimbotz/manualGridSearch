# Librerías
from itertools import product
import pandas as pd
from colorama import Fore, Style
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import time


# Menú principal
def menu_principal(df):
    print("\n=========================")
    opcion_modelo = int(
        input("1. Random Forest\n2. SVM\n3. KNN\nModelo a usar: "))

    # Separar datos
    x = df.drop(columns='label')
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=42)

    # Búsqueda de mejores hiperparámetros de Random Forest
    if opcion_modelo == 1:
        algoritmo = 1
        print(Fore.GREEN + "\n[+] " + Style.RESET_ALL, end='')
        print("Búsqueda de los mejores hiperparámetros para Random Forest...")

        # Diccionario de hiperparámetros
        hiperparametros = {
            'max_depth': [None, 5, 10, 15, 20, 30],
            'n_estimators': [50, 100, 150, 200, 250],
            # Número dmínimo de muestras necesarias para dividir un nodo
            'min_samples_split': [2, 5, 10, 20],
            # Número mínimo de muestras requeridas en hoja final
            'min_samples_leaf': [1, 2, 3, 8]
        }

    elif opcion_modelo == 2:
        algoritmo = 2
        print(Fore.RED + "\n[+] " + Style.RESET_ALL, end='')
        print("Búsqueda de mejores hiperparámetros para SVM...")

        # Diccionario de hiperparámetros
        hiperparametros = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }

    elif opcion_modelo == 3:
        algoritmo = 3
        print(Fore.YELLOW + "\n[+] " + Style.RESET_ALL, end='')
        print("Búsqueda de mejores hiperparámetros para KNN...")

        # Diccionario de hiperparámetros
        hiperparametros = {
            'n_neighbors': [3, 5, 7, 9],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

    # Llamada a función de hiperparámetros
    diccionarios(x_train, x_test, y_train, y_test, algoritmo, hiperparametros)


# Función para desempaquetar diccionarios
def diccionarios(x_train, x_test, y_train, y_test, algoritmo, hiperparametros):
    print(Fore.RED + "[!] " + Style.RESET_ALL, end='')
    n_procesos = int(input("Número de procesos a ejecutar: "))

    # Inicio de toma de tiempo
    inicio = time.time()

    # Generar todas las combinaciones posibles
    claves = list(hiperparametros.keys())
    combinaciones = list(product(*hiperparametros.values()))

    # Convertir tuplas en diccionarios
    combinaciones_dict = [dict(zip(claves, c)) for c in combinaciones]

    # Crear lista de argumentos
    tareas = [(params, x_train, x_test, y_train, y_test)
              for params in combinaciones_dict]

    # Ejecución en paralelo de Pool
    resultados = []

    if algoritmo == 1:
        with Pool(processes=n_procesos) as pool:
            resultados = pool.map(evaluar_rf, tareas)
            resultados = list(
                tqdm(pool.imap(evaluar_rf, tareas), total=len(tareas)))
    elif algoritmo == 2:
        with Pool(processes=n_procesos) as pool:
            resultados = list(
                tqdm(pool.imap(evaluar_svm, tareas), total=len(tareas)))
    elif algoritmo == 3:
        with Pool(processes=n_procesos) as pool:
            resultados = list(
                tqdm(pool.imap(evaluar_knn, tareas), total=len(tareas)))

    fin = time.time()

    mejor = max(resultados, key=lambda x: x[1])
    print(f"Mejor: {mejor}")
    print(f"Tiempo total de ejecución: {round(fin - inicio, 2)}")


# Función para evaluar Random Forest con Pool
def evaluar_rf(args):
    parametros, x_train, x_test, y_train, y_test = args
    # Instancia de modelo con hiperparámetros actuales
    rf = RandomForestClassifier(
        random_state=42,
        **parametros
    )

    # Evaluar con accuracy
    rf.fit(x_train, y_train)
    prediccion = rf.predict(x_test)
    score = accuracy_score(y_test, prediccion)

    return (parametros, score)


# Función para evaluar SVM con Pool
def evaluar_svm(args):
    parametros, x_train, x_test, y_train, y_test = args
    # Instancia de modelo con hiperparámetros actuales
    svm = SVC(
        **parametros
    )

    # Evaluar con accuracy
    try:
        svm = SVC(**parametros)
        svm.fit(x_train, y_train)
        prediccion = svm.predict(x_test)
        score = accuracy_score(y_test, prediccion)
        return (parametros, score)
    except Exception as e:
        return (parametros, 0)


# Función para evaluar KNN con Pool
def evaluar_knn(args):
    parametros, x_train, x_test, y_train, y_test = args
    # Instancia de modelo con hiperparámetros actuales
    knn = KNeighborsClassifier(
        **parametros
    )

    # Evaluar con accuracy
    knn.fit(x_train, y_train)
    prediccion = knn.predict(x_test)
    score = accuracy_score(y_test, prediccion)

    return (parametros, score)


# ======================
# = Inicio de programa =
# ======================
if __name__ == '__main__':
    archivo = 'fashion-mnist_train.csv'
    # Lectura de archivo
    print(Fore.YELLOW + "\n[!] " + Style.RESET_ALL, end='')
    print("Leyendo archivo...")
    df = pd.read_csv(f'./Fashion_MNIST/{archivo}')
    print(Fore.GREEN + "\n[+] " + Style.RESET_ALL, end='')
    print(f"Archivo leído: {archivo}")

    # Llamada a función de menú
    # menu_principal(df)

    # ==============================
    print(Fore.CYAN + "\n[+] " + Style.RESET_ALL, end='')
    print("Leyendo archivo de prueba para evaluación final...")

    df_test = pd.read_csv('./Fashion_MNIST/fashion-mnist_test.csv')
    x_test = df_test.drop(columns='label')
    y_test = df_test['label']

    # Mejores hiperparámetros encontrados
    mejores_params_rf = {'max_depth': 30, 'n_estimators': 250,
                         'min_samples_split': 2, 'min_samples_leaf': 1}
    mejores_params_svm = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
    mejores_params_knn = {'n_neighbors': 5, 'metric': 'manhattan'}

    # Número de repeticiones
    n_reps = 10

    print(Fore.MAGENTA + "\n[+] " + Style.RESET_ALL +
          f"Ejecutando {n_reps} repeticiones por modelo...\n")

    resultados = {}

    # Random Forest
    accs_rf = []
    for i in tqdm(range(n_reps), desc="Random Forest"):
        x_train, _, y_train, _ = train_test_split(
            x_test, y_test, train_size=0.8, random_state=i)
        model = RandomForestClassifier(**mejores_params_rf, random_state=i)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accs_rf.append(accuracy_score(y_test, pred))

    resultados['Random Forest'] = {
        'media': round(pd.Series(accs_rf).mean(), 4),
        'varianza': round(pd.Series(accs_rf).var(), 6),
        'desviacion': round(pd.Series(accs_rf).std(), 6)
    }

    # SVM
    accs_svm = []
    for i in tqdm(range(n_reps), desc="SVM"):
        x_train, _, y_train, _ = train_test_split(
            x_test, y_test, train_size=0.8, random_state=i)
        model = SVC(**mejores_params_svm)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accs_svm.append(accuracy_score(y_test, pred))

    resultados['SVM'] = {
        'media': round(pd.Series(accs_svm).mean(), 4),
        'varianza': round(pd.Series(accs_svm).var(), 6),
        'desviacion': round(pd.Series(accs_svm).std(), 6)
    }

    # KNN
    accs_knn = []
    for i in tqdm(range(n_reps), desc="KNN"):
        x_train, _, y_train, _ = train_test_split(
            x_test, y_test, train_size=0.8, random_state=i)
        model = KNeighborsClassifier(**mejores_params_knn)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accs_knn.append(accuracy_score(y_test, pred))

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
