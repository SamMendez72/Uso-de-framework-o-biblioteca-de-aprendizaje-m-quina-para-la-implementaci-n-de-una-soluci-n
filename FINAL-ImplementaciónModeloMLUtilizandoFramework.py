## Samuel Méndez Villegas - A016522777
## Uso de un framework/librería para implementar un modelo de ML

'''
En este código se presenta la implementación de dos modelos de Machine Learning (ML), más concretamente de
un 'Random Forest Classifier' y un 'Random Forest Regressor'. Los modelos forman parte de la lista de algoritmos supervisados 
de ML y se caracterisan por crear 'n' subconjuntos de datos del set de entrenamiento y posteriormente, generar un árbol de 
decisión por cada subconjunto. Después de la creación de los árboles, se realiza una predicción de cada árbol, y se obtiene 
un resultado.
Dichos resultados pasan a un tipo de 'votación' en donde se escoge la clase más votada (en caso de clasificación), o 
el promedio de los resultados (en caso de regresión).

Para la implementación de esta librería, se utilizará la librería scikit-learn la cual incluye las funciones
'RandomForestClassifier()' y 'RandomForestRegressor()' que reciben el conjunto de entrenamiento y posteriormente
el conjunto de prueba para realizar predicciones.

En cuanto a la base de datos que se utilizará, es la base de datos de diabetes, en la cual se tratará de predecir
(en el caso del clasificador) si un individuo padece de diabetes dada una serie de características como el nivel de glucosa, el número embarazos,
la edad, entre otras.

En el caso del regresor, este tiene el objetivo de predecir el BMI de una persona a partir de una serie de características
similares a las del clasificador. 

Finalmente, se realiza la evaluación de los modelos implementados con ayuda de algunas métricas como el 'mean squared 
error', el 'mean absolute error', y la matriz de confusión. 
'''

## Librerías a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

## Librerías para implementar el modelo
from sklearn.model_selection import train_test_split # Separar el set de datos en entrenamiento y prueba
from sklearn.ensemble import RandomForestClassifier # Clasificador a utilizar
from sklearn.ensemble import RandomForestRegressor # Regresor a utilizar
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score # Medidas de evaluación

'''
EXPLORACIÓN DE LA BASE DE DATOS Y UN POCO DE LIMPIEZA

En esta pequeña sección se pasará a visualizar los datos y realizar un pequeño análisis exploratorio de éstos antes
de aplicar el modelo de machine learning.
'''
## Carga de los datos a utilizar
df = pd.read_csv('diabetes.csv')
df.head()

## Dimensiones del data frame
df.shape

## Evaluación de datos duplicados
df.duplicated().sum()

## Evaluación de datos nulos
df.isnull().sum()

## Análisis descriptivo de las variables numéricas
df.describe()

'''
Gracias al resumen de estadística descriptiva que ofrece `describe`, se observa que se tienen registros con
valores de 0 en algunas columnas como presión sanguinea, o en el ínidice de masa muscular, por lo que se pasa 
a identificar dichos registros
'''

df[df['BloodPressure'] == 0].head()
df[df['Insulin'] == 0].head()
df[df['BMI'] == 0].head()

'''
Se puede observar que la muchos de los registros tienen valores de 0, por lo que puede que en realidad se traten de
valores faltantes que se llenaron con ceros. Por lo tanto, para tener un modelo un poco más limpio, se pasará a
eliminar estos registros. 
'''

df = df[(df['BloodPressure'] > 0) & (df['Glucose'] > 0) & (df['SkinThickness'] > 0) & (df['Insulin'] > 0) & (df['BMI'] > 0)]
df.shape
df.describe()

'''
Como se puede observar en el nuevo describe, ya no se tienen valores mínimos en las variables en las no tenia sentido
tener un cero. Ya con este súper pequeño análisis de los datos, pasaremos a implementar el clasificador con bosques
aleatorios.
'''

print('\n-------------------------------')
print('    RANDOM FOREST CLASSIFIER     ')
print('-------------------------------\n')

## Selección de las variables predictoras y objetivo
x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

## Separación de la base de datos en un set de entrenamiento y prueba
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
X_train.shape, x_test.shape, Y_train.shape, y_test.shape

# Implementación del random forest classifier con los hiperparámetros default
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train) ## Entrenamiento del modelo

## Se realizan las predicciones con el set de entrenamiento
train_results = rfc.predict(X_train)

## Se obtienen algunas medidas de evaluación como el mse, mae, la exactitud y la matriz de confusión
mse = mean_squared_error(train_results, Y_train)
mae = mean_absolute_error(train_results, Y_train)

print('\nResultados de las predicciones (RFC) con el set de entrenamiento:')
print('MSE con el conjunto de entrenamiento:', mse)
print('MAE con el conjunto de entrenamiento:', mae)
print("Accuracy con el conjunto de prueba:", accuracy_score(Y_train, train_results)) ## Indica que tanto el modelo clasificó correctamente
print(confusion_matrix(train_results, Y_train)) # Se muestra la matriz de confusión
print('-----------------------------------------------------------')

'''
Como se puede observar en los resultados de la matriz de confusión y de algunas medidas como el accuracy, el mse y el
mae, el modelo es muy bueno para predecirse a si mismo, indicando de esta forma un posible overfitting.

Ahora se pasará a realizar las predicciones con el set de prueba.
'''

## Con los datos de prueba
test_results = rfc.predict(x_test)
## Se obtienen algunas medidas de evaluación como el mse, mae, la exactitud y la matriz de confusión
mse = mean_squared_error(test_results, y_test)
mae = mean_absolute_error(test_results, y_test)

print('\nResultados de las predicciones (RFC) con el set de prueba:')
print('MSE con el conjunto de prueba:', mse)
print('MAE con el conjunto de prueba:', mae)
print("Accuracy con el conjunto de prueba:", accuracy_score(y_test, test_results)) ## Indica que tanto el modelo clasificó correctamente
print(confusion_matrix(test_results, y_test)) # Se muestra la matriz de confusión
print('-----------------------------------------------------------')

'''
Se observa que la exactitud del modelo ya no es tan buena, pues al tener un overfitting, el modelo es malo para
predecir con nuevos registros. Sin embargo, si se habla del MSE y MAE se puede observar que estos no son tan grandes.

Ahora se pasará a implementar el mismo modelo de random forest, pero configurando los hiperparámetros con el objetivo
de mejorar al modelo y que éste realice mejores predicciones. 
'''

## Modificación de los hiperparámetros del modelo con el objetivo de mejorar el modelo
rfc = RandomForestClassifier(n_estimators = 1000, max_depth = 5, criterion = 'entropy', random_state = 42)
rfc.fit(X_train, Y_train) ## Entrenamiento del modelo

## Se realizan las predicciones en el conjunto de entrenamiento
train_results = rfc.predict(X_train)

## Se obtienen algunas de las medidas de evaluación
mse = mean_squared_error(train_results, Y_train)
mae = mean_absolute_error(train_results, Y_train)

print('\nResultados de las predicciones (RFC mejorado) con el set de entrenamiento:')
print('MSE con el conjunto de entrenamiento:', mse)
print('MAE con el conjunto de entrenamiento:', mae)
print("Accuracy con el conjunto de prueba:", accuracy_score(Y_train, train_results)) ## Indica que tanto el modelo clasificó correctamente
print(confusion_matrix(train_results, Y_train)) # Se muestra la matriz de confusión
print('-----------------------------------------------------------')

'''
Desde aquí se observa que este modelo mejoró algo en consideración al anterior, en el sentido de que ya no se 
tiene un sobre ajuste de los datos, pues como se observa el modelo tiende fallar, poco, pero falla. Ahora probaremos
su rendimiento con el conjunto de prueba.
'''

## Con los datos de prueba
test_results = rfc.predict(x_test)

mse = mean_squared_error(test_results, y_test)
mae = mean_absolute_error(test_results, y_test)

print('\nResultados de las predicciones (RFC mejorado) con el set de prueba:')
print('MSE con el conjunto de prueba:', mse)
print('MAE con el conjunto de prueba:', mae)
print("Accuracy con el conjunto de prueba:", accuracy_score(y_test, test_results)) ## Indica que tanto el modelo clasificó correctamente
print(confusion_matrix(test_results, y_test)) # Se muestra la matriz de confusión
print('-----------------------------------------------------------')

'''
Finalmente, se observa que en realidad el modelo tuvo una mejora, sin embargo esta no fue del todo buena, ya que
solamente aumentó en 2% y de igual forma tanto el mae como el mse están más cercanos a 0. En la siguiente actividad
se realizará un análisis mucho más profundo de la evaluación del modelo con el objetivo de mejorar este mucho más.

Ahora, se pasa a implementar el modelo de regresión, por lo que se cambia la variable objetivo y las variables 
predictoras.
'''

print('\n-------------------------------')
print('    RANDOM FOREST REGRESSOR      ')
print('-------------------------------\n')

## Selección de las variables predictoras y objetivo
x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]
y = df['BMI']

## Separación de la base de datos en un set de entrenamiento y prueba
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
X_train.shape, x_test.shape, Y_train.shape, y_test.shape

# Implementación del random forest regressor con los hiperparámetros default
rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train) ## Entrenamiento del modelo

## Se realizan las predicciones con el set de entrenamiento
train_results = rfr.predict(X_train)

## Se obtienen algunas medidas de evaluación como el mse, mae, la exactitud y la matriz de confusión
mse = mean_squared_error(train_results, Y_train)
mae = mean_absolute_error(train_results, Y_train)
r2 = r2_score(train_results, Y_train)

print('\nResultados de las predicciones (RFR) con el set de entrenamiento:')
print('MSE con el conjunto de entrenamiento:', mse)
print('MAE con el conjunto de entrenamiento:', mae)
print('El coeficiente R^2 con el conjunto de entrenamiento:', r2)
print('-----------------------------------------------------------')

'''
Al tratarse de un modelo de regresión, no es posible obtener una matriz de confusión, sin embargo si se pueden
obtener métricas como el mse, msa y el r^2. Como se observa, las predicciones cuentan con un error muy pequeñito,
mientras que el valor de r^2 es elevado, sugiriendo así un buen modelo. Sin embargo, se pasará a evaluar el modelo
con el conjunto de entrenamiento
'''

## Con los datos de prueba
test_results = rfr.predict(x_test)

mse = mean_squared_error(test_results, y_test)
mae = mean_absolute_error(test_results, y_test)
r2 = r2_score(test_results, y_test)

print('\nResultados de las predicciones (RFR) con el set de prueba:')
print('MSE con el conjunto de prueba:', mse)
print('MAE con el conjunto de prueba:', mae)
print('El coeficiente R^2 con el conjunto de prueba:', r2)
print('-----------------------------------------------------------')

'''
En esta última evaluación del código, se obtiene observa que aumentó bastante tanto el MSE como el MAE,
además de disminuir el valor de R^2. Esto nos habla que el modelo de regresión cuenta con un sobreajuste, pues
cuando se le dieron datos de entrenamiento realizó un muy buen trabajo, pero cuando se le da información nunca
antes vista, el modelo emperora mucha. Claro que esto se puede mejorar modificando los hiperparámetros. 
'''

## Modificación de los hiperparámetros del modelo con el objetivo de mejorar el modelo
rfr = RandomForestRegressor(n_estimators = 1000, max_depth = 5, random_state = 42)
rfr.fit(X_train, Y_train) ## Entrenamiento del modelo

## Se realizan las predicciones en el conjunto de entrenamiento
train_results = rfr.predict(X_train)

## Se obtienen algunas de las medidas de evaluación
mse = mean_squared_error(train_results, Y_train)
mae = mean_absolute_error(train_results, Y_train)
r2 = r2_score(train_results, Y_train)

print('\nResultados de las predicciones (RFR mejorado) con el set de entrenamiento:')
print('MSE con el conjunto de entrenamiento:', mse)
print('MAE con el conjunto de entrenamiento:', mae)
print('El coeficiente R^2 con el conjunto de entrenamiento:', r2)
print('-----------------------------------------------------------')

## Con los datos de prueba
test_results = rfr.predict(x_test)

mse = mean_squared_error(test_results, y_test)
mae = mean_absolute_error(test_results, y_test)
r2 = r2_score(test_results, y_test)

print('\nResultados de las predicciones (RFC mejorado) con el set de prueba:')
print('MSE con el conjunto de prueba:', mse)
print('MAE con el conjunto de prueba:', mae)
print('El coeficiente R^2 con el conjunto de entrenamiento:', r2)
print('-----------------------------------------------------------')

'''
Finalmente se observa que el modelo mejoró muy poco con respecto al anterior (antes del ajuste de hiperparámetros),
concluyendo así que se tiene un modelo malo para predecir el BMI de los pacientes.
'''