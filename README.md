# manualGridSearch

To use this project you must install the dataset with the following link:
__https://www.kaggle.com/datasets/zalando-research/fashionmnist__

And unzip into the "archive" folder

You also need to create a .gitignore to ignore the archive folder
.gitignore:
/archive/

Viendo el dataset al parecer las imagenes ya estan aplanadas, entonces no ocupas preprocesar algo antes
# fashion-mnist_test.csv 
10k de fotos
# fashion-mnist_train.csv
60k de fotos

Aca no se si debemos de juntar todo de madrazo porque el profesor dijo que hay que probar diferentes sets de train y test

# 28x28 en escala de grises
Son 10 clases:
0 	T-shirt/top
1 	Trouser
2 	Pullover
3 	Dress
4 	Coat
5 	Sandal
6 	Shirt
7 	Sneaker
8 	Bag
9 	Ankle boot


Los mejores hiperparametros que encontre con random forest fueron:
{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}

--- Reporte de Clasificación (Prueba) ---
              precision    recall  f1-score   support

 T-shirt/top       0.82      0.86      0.84      1000
     Trouser       0.99      0.97      0.98      1000
    Pullover       0.81      0.80      0.81      1000
       Dress       0.89      0.94      0.91      1000
        Coat       0.81      0.87      0.84      1000
      Sandal       0.97      0.95      0.96      1000
       Shirt       0.76      0.62      0.68      1000
     Sneaker       0.92      0.93      0.93      1000
         Bag       0.95      0.97      0.96      1000
  Ankle boot       0.94      0.95      0.94      1000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Los mejores hiperparámetros para SVM son:
{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}

--- Reporte de Clasificación (Prueba) con SVM ---
              precision    recall  f1-score   support

 T-shirt/top       0.84      0.88      0.86      1000
     Trouser       0.99      0.98      0.99      1000
    Pullover       0.85      0.83      0.84      1000
       Dress       0.91      0.92      0.91      1000
        Coat       0.85      0.88      0.86      1000
      Sandal       0.98      0.96      0.97      1000
       Shirt       0.79      0.73      0.76      1000
     Sneaker       0.94      0.96      0.95      1000
         Bag       0.98      0.98      0.98      1000
  Ankle boot       0.96      0.97      0.96      1000

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000


El mejor número de vecinos (k) para K-NN es:
{'n_neighbors': 5}

--- Reporte de Clasificación (Prueba) con K-NN ---
              precision    recall  f1-score   support

 T-shirt/top       0.77      0.87      0.82      1000
     Trouser       0.99      0.96      0.98      1000
    Pullover       0.75      0.81      0.78      1000
       Dress       0.91      0.88      0.90      1000
        Coat       0.79      0.80      0.79      1000
      Sandal       1.00      0.82      0.90      1000
       Shirt       0.68      0.58      0.63      1000
     Sneaker       0.87      0.94      0.91      1000
         Bag       0.98      0.95      0.97      1000
  Ankle boot       0.88      0.96      0.92      1000

    accuracy                           0.86     10000
   macro avg       0.86      0.86      0.86     10000
weighted avg       0.86      0.86      0.86     10000
