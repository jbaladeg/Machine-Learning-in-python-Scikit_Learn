# # Apuntes Scikit learn

# ###  Jennifer Balade González

# ## Introducción

#### Este ejercicio sirve de apunte para introducir el paquete scikitlearn en python.
#### Usamos la base digits() y diferentes algoritmos de clasificación binaria:

#### KNN
#### regresion logistica (+ validación cruzada, + GridSearchCV)

# ###  1. Preparar los datos

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

digits.feature_names

digits.target_names

#####: Explicación de la base: 
#### digits.data: es un array NumPy de forma (1797, 64). 
#### Cada fila representa una imagen individual, y cada una de las 64 columnas corresponde a un valor de píxel específico. 
#### Las imágenes originales de 8x8 píxeles se han aplanado en un vector de 64 características (de ahí las columnas), 
### con valores que oscilan entre 0 (blanco) y 16 (negro), representando la intensidad de la escala de grises.


#### digits.target: es un array de forma (1797,) contiene la etiqueta de clase verdadera para cada imagen correspondiente en digits.data. 
#### Los valores en este target son números enteros del 0 al 9, 
#### indicando cuál es el dígito manuscrito correcto representado en los datos de píxeles.

#### Vamos a guardar los valores de los valores y el target, así como sus nombres, en dos objetos.

x_digits, y_digits = digits.data, digits.target
x_names, y_names = digits.feature_names, digits.target_names

#### Visualizo un poco los datos

from matplotlib import pyplot as plt

plt.figure(figsize=(10, 5)) #coloco los 10 valores,  por cada fila
for i in range(10): #recorro todos los target
    ax = plt.subplot(2, 5, i + 1) #2 filas, 5 columnas, y la posición donde irá
    ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')  # Muestra la matriz 8x8 como imagen en escala de grises
    ax.set_title(f'Target: {digits.target[i]}') #el titulo de cada "numero" será el número de su target
    ax.axis('off') # Oculta los márgenes para que se vea mejor

plt.suptitle("Ejemplos de dígitos manuscritos (8x8 píxeles)") #titulo
plt.show() #mostrar

#### Para realizar el siguiente punto (Clasificador Knn) es necesario utilizar una muestra de entrenamiento y test.
#### Vamos a usar la función train_test_solit(), y separamos en entrenamiento y prueba con una partición 75%-25%

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x_digits,y_digits,test_size = 0.25)

#### Normalizamos los datos para evitar inferencias en la precisión de los modelos aprendidos.
#### para el StandardScaler() lo que haces pasar la media a 0 y la desviación típica a 1
### el primer paso es llamar a esta función, y calcular los parámetros

from sklearn.preprocessing import StandardScaler

normalizador=StandardScaler()

normalizador.fit(x_train)

print(normalizador.mean_)
print(normalizador.scale_)

#### utilizo el método transform para normalizar los datos de x_train
#### lo guardo en un nuevo objeto llamado xn_train
#### compruebo que salió bien

####apunte: axis = 0: Un valor por columna; axis = 1: Un valor por fila

xn_train = normalizador.transform(x_train)

import numpy as np
print(np.mean(xn_train,axis=0))
print(np.std(xn_train,axis=0))

#### Ahora lo mismo pero para x_test

normalizador.fit(x_test)
xn_test = normalizador.transform(x_test)


# ###  2. Clasificador Knn

#### KNN es un algoritmo de clasificación que asigna a un ejemplo la clase más frecuente entre sus k vecinos más cercanos 
#### en el espacio de características. 

#### El parámetro k determina cuántos vecinos se consideran: 
#### un valor pequeño hace que el clasificador sea más sensible al ruido, 
#### mientras que un valor grande suaviza las decisiones pero puede perder detalles locales. 

#### No tiene fase de entrenamiento como tal, ya que la “memoria” del modelo son los datos de entrenamiento; 
#### la predicción se realiza comparando distancias (euclídeas) entre el ejemplo nuevo y los datos existentes.

#### Para usar el Knn podemos darle de entrada ya el número de vecinos k, por ejemplo:

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(xn_train,y_train)
knn.score(xn_train,y_train)

knn.predict(xn_test)
knn.score(xn_test,y_test)

#### Los resultados muestran que El 96.22 % de los ejemplos del conjunto de test 
#### han sido clasificados correctamente por el modelo KNN.

#### No obstante, podemos averiguar que "k" es mejor para nuestros datos de la siguiete manera:

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

training_accuracy = [] #lista vacia para los datos de entrenamiento 
test_accuracy = []  #lista vacía para los datos de test

neighbors_settings = range(1, 15) #pruebo con los "k" del 1 al 15

for k in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(xn_train, y_train)
    
    training_accuracy.append(knn.score(xn_train, y_train))
    test_accuracy.append(knn.score(xn_test, y_test))


#visualizo todo
plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="Test accuracy")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#### Parece ser que mi mejor "k" es el número 8.
#### Prosigo con este clasificador.

knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(xn_train,y_train)
knn.score(xn_train,y_train)

knn.predict(xn_test)
knn.score(xn_test,y_test)

#### Los resultados muestran que El 97.33 % de los ejemplos del conjunto de test 
#### han sido clasificados correctamente por el modelo KNN.

#### un poco mejor que el clasificador primero que pusimos 
#### k = 7; score = 96.22 %
#### k = 8; score = 97.33 %

# ###  3. Clasificador Regresión logística

from sklearn.linear_model import LogisticRegression

#### En la regresión logística podemos realizarlo de dos maneras: 
#### 1) probando diferentes "C" o "regulaciones" :
    #### C = 1 [por defecto]
    #### C = 100 [más regularización, más flexible]
    #### C = 0.01 [menos regularización, más simple]
#### 2) Clasificación multiclase (One vs Rest)
    #### Se entrenan 10 clasificadores binarios
    #### Cada uno separa un dígito frente a los demás
    #### La predicción final es la clase con mayor puntuación

1) #### Veamos la clasificación de regresion logística en base a las regulaciones.

#### Primero vamos a ver el comportamiento de la regresión losgística con el C=1 (por defecto)

logreg = LogisticRegression().fit(xn_train, y_train)
print("Rendimiento sobre entranamiento: {:.3f}".format(logreg.score(xn_train, y_train)))
print("Rendimiento sobre el conjunto de prueba: {:.3f}".format(logreg.score(xn_test, y_test)))

#### C=100, más regularización, más flexible

logreg = LogisticRegression(C = 100).fit(xn_train, y_train)
print("Rendimiento sobre entranamiento: {:.3f}".format(logreg.score(xn_train, y_train)))
print("Rendimiento sobre el conjunto de prueba: {:.3f}".format(logreg.score(xn_test, y_test)))

#### C=0.01, menos regularización, más simple

logreg = LogisticRegression(C = 0.01).fit(xn_train, y_train)
print("Rendimiento sobre entranamiento: {:.3f}".format(logreg.score(xn_train, y_train)))
print("Rendimiento sobre el conjunto de prueba: {:.3f}".format(logreg.score(xn_test, y_test)))

#### si comparamos las diferentes puntuaciones: 
    #### C = 1; rend. entrenamiento = 0.99; rend. prueba = 0.96
    #### C = 100; rend. entrenamiento = 1; rend. prueba = 0.96
    #### C = 0.01; rend. entrenamiento = 0.95; rend. prueba = 0.94

#### Al comparar distintos valores del parámetro de regularización C, se observa que valores muy grandes conducen a 
#### modelos excesivamente complejos que no mejoran el rendimiento en el conjunto de prueba, mientras que valores 
#### pequeños producen infraajuste. El valor intermedio C = 1 proporciona el mejor equilibrio entre capacidad de 
#### ajuste y generalización, alcanzando una precisión elevada sin sobreajuste.

#### 2) Ahora vamos a probar el clasificador multiclase (one vs rest)

lr = LogisticRegression(multi_class='ovr')
lr.fit(xn_train, y_train)

# Predicciones
y_pred = lr.predict(xn_test)
y_pred

# Mostramos una cuadrícula de imágenes con predicción y etiqueta
fig, axes = plt.subplots(4, 10, figsize=(15, 6))
axes = axes.ravel()

for i in range(40):  # mostramos las primeras 40 imágenes
    axes[i].imshow(xn_test[i].reshape(8, 8), cmap='gray')
    axes[i].set_title(f'{y_pred[i]} / {y_test[i]}', color='green' if y_pred[i]==y_test[i] else 'red')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#### Para ver la precision
y_pred = lr.predict(xn_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del clasificador: {accuracy:.4f}")

#### La precisión del clasificador es del 97.33 %

#### Para ver el reporte de clasificación
print(classification_report(y_test, y_pred))

#### El reporte de clasificación muestra que mi modelo de regresión logística One-vs-Rest para los dígitos funciona muy bien. 
#### La exactitud global es del 97%, así que la mayoría de las predicciones son correctas. 
#### Las métricas por clase, como precisión, recall y F1-score, también son muy altas, lo que significa que el modelo acierta 
#### casi siempre y rara vez confunde los dígitos, aunque las clases 8 y 9 tienen un poco más de errores (ver columna de recall). 
#### Tanto el promedio macro como el ponderado confirman que el modelo está equilibrado, 
#### así que en general la clasificación multiclase es buena.

# ###  4. Validación cruzada

#### La validación cruzada es otro método para evaluar la precisión de los modelos.
#### anteriormente hemos estado viendo este tipo de metodología: entramiento vs test
#### Esta es más precisa. Lo que hace es dividir en 5 partes iguales la data, 5 "split"
#### APUNTE: 5 partes o saltos es por defecto, pero podemos modificar la cantidad de saltos con el argumento cv = ...
#### En cada "split" 1/5 parte lo usa de test, y el 4/5 restante lo usa de entrenamiento. 
#### De cada "split" sacamos un valor de precisión.

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs', max_iter=1000).fit(xn_train, y_train)
#le he tenido que añadir un número mayor de interacciones porque no me ajustaba bien el modelo

scores = cross_val_score(logreg, digits.data, digits.target) #puntuaciones de la validación cruzada
print("Resultados de la evaluación cruzada: {}".format(scores))

#### Resultados de la evaluación cruzada: [0.92222222 0.86944444 0.94150418 0.94150418 0.89693593]
#### Estos resultados son la media de los entrenamiento dentro de los 5 saltos que ha hecho la validación cruzada.

#### Le pedimos la media de las 5 validaciones:

scores = cross_val_score(logreg, digits.data, digits.target)
print("Resultados de la evaluación cruzada: {}".format(scores))
print("Evaluación media: {:.2f}".format(scores.mean()))

#### En ocasiones puede ser que la extracción por splits no esté estractificada.
#### Es decir, que los saltos no seleccionen un número de datos equiparables entre los diferentes targets.
#### Especialemnte cuando los targets están ordenados, como pasa en los datos iris

#### Una manera de corregirlo es utilizando el argumento de estractificar, que se utiliza así:

from sklearn.model_selection import cross_val_score, StratifiedKFold

logreg = LogisticRegression(solver='lbfgs', max_iter=1000).fit(xn_train, y_train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(logreg, xn_test, y_test, cv=cv)

print("Resultados de la validación cruzada estratificada:", scores)
print("Media del rendimiento: {:.3f}".format(np.mean(scores)))

#### Podemos observar que el valor de precisión ha mejorado cuando hemos estractificado las clases (o targets) de la base
#### De 91 % hemos pasado a 95 % de precisión, mejorando también la media de precisión de los 5 splits

# ###  6. GridSearchCV

#### Un grid search es un procedimiento para seleccionar los mejores hiperparámetros de un modelo. 
#### Consiste en definir una rejilla con distintas combinaciones posibles de esos hiperparámetros y, para cada una de ellas, 
#### evaluar el modelo mediante validación cruzada. 

#### El rendimiento medio obtenido en los distintos folds se compara entre combinaciones y se elige aquella que mejor generaliza. 
#### Por tanto, el grid search no sustituye a la validación cruzada, sino que la utiliza como herramienta para decidir qué 
#### valores de los hiperparámetros son los más adecuados.

#### APUNTE:Un hiperparámetro es un valor que se fija antes de entrenar un modelo y que controla cómo aprende.
#### como por ejemplo la C en la regresión logística, k en KNN o el número de folds (o cv) en la validación cruzada.

#### El grid search busca la mejor combinación de hiperparámetros (no solo uno, si hay varios) 
#### evaluando cada combinación mediante validación cruzada y seleccionando la que obtiene mejor rendimiento medio. 
#### Es decir, sirve para decidir qué valores de los hiperparámetros hacen que el modelo generalice mejor.

#repito la normalización para que no interfiera con los anterior
scaler = StandardScaler()
xn_train = scaler.fit_transform(x_train)
xn_test = scaler.transform(x_test)

#GRID SEARCH CASERO (Como vimos en clase)
best_score = 0
for penal in ["l1", "l2"]:   #diferentes tipos de regularización
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:  #diferentes grados de regularización
        logreg = LogisticRegression(penalty=penal, C=C, solver='saga', max_iter=1000)
        logreg.fit(xn_train, y_train)
        score = logreg.score(xn_test, y_test)

        if score > best_score:
            best_score = score
            best_parameters = {'penalty': penal, 'C': C}

print("Mejor resultado:", best_score)
print("Mejor combinación:", best_parameters)

#### Lo que hemos hecho aqui es comparar l1 vs l2, con diferentes niveles de regularización (C)

#### APUNTES:
####L1 y L2 (penalties o regularizaciones)
####Son formas de penalizar los coeficientes del modelo para evitar sobreajuste:
    ####L1 (Lasso):
        #### Hace que algunos coeficientes queden exactamente en 0, eliminando variables irrelevantes
        #### Útil si queremos selección automática de características.
        #### L1 → “apaga” algunas variables.
    ####L2 (Ridge):
        ####Reduce todos los coeficientes, pero ninguno llega a 0.
        ####Favorece soluciones más estables y distribuye la penalización entre todas las variables.
        #### L2 → “achica” todas las variables pero mantiene todas activas.

#### el argumento "solver" son algoritmos que usa scikit-learn para entrenar el modelo de regresión logística
#### Algunos soportan solo L1, otros solo L2, otros ambos; algunos funcionan con multiclase, otros no:
    #### liblinear = Sporta L1 y L2. Multiclase: One-vs-Rest.	Bueno para pocas clases y datasets pequeños.
    #### lbfgs = Solo L2. Multiclase: Multinomial.	Estable, rápido, recomendado en multiclase.
    #### saga = Soporta L1 y L2. Multiclase: Multinomial.	Escalable, funciona bien con datasets grandes.
    #### newton-cg = Solo L2. Multiclase: Multinomial.	Similar a lbfgs, estable.

#### En los resultados anteriores nos ha salido un 97.11 % cuando usamos un penalty L2, y un C pequeño de 0.1:
    #### Mejor resultado: 0.9711111111111111
    #### Mejor combinación: {'penalty': 'l2', 'C': 0.1}

#### No obstante, hacer un grid search casero usando el conjunto de test es arriesgado, 
#### porque se filtra información del test al modelo, dando resultados demasiado optimistas. 
#### Para medir el rendimiento real, el test debe ser independiente, sin haberse usado ni para entrenar 
#### ni para ajustar hiperparámetros.

### Por eso se suele partir el conjunto de datos original en tres partes: 
    ### - entrenamiento (para entrenar el modelo)
    ### - validación (para ajustar los parámetros del modelo)
    ### - test (para dar una estimación del rendimiento del modelo finalmente escogido)

#GRID SEARCH CON ENTRENAMIENTO, VALIDACIÓN Y TEST

# partición: entrenamiento+validación y test
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=0)

# Partimos entrenamiento+validación en entrenamiento y validación
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, random_state=1)
print("Tamaño del conjunto de entrenamiento: {}\nTamaño del conjunto de validación: {}\nTamaño del conjunto de test:"
      " {}\n".format(x_train.shape[0], x_valid.shape[0], x_test.shape[0]))

best_score = 0

for penal in ["l1","l2"]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # por cada combinación de valores de los parámetros, entrenamos un log reg
        logreg = LogisticRegression(penalty=penal,solver="saga", C=C, max_iter = 1000)
        logreg.fit(x_train, y_train)
        # Y lo evaluamos sobre el conjunto de validacion
        score = logreg.score(x_valid, y_valid)
        # Nos vamos quedando con el de mejor resultado
        if score > best_score:
            best_score = score
            best_parameters = {'penalty': penal, 'C': C}


# Volvemos a entrenar el modelo con la mejor combinación encontrada, sobre entrenamieno+validación  
# y evaluamos el rendimiento sobre el conjunto de prueba 
logreg = LogisticRegression(**best_parameters,solver="saga", max_iter = 1000)
logreg.fit(x_train, y_train)
test_score = logreg.score(x_test, y_test)
print("Mejor resultado sobre validación: {:.2f}".format(best_score))
print("Mejor combinación de valores: ", best_parameters)
print("Evaluación sobre el conjunto de test: {:.2f}".format(test_score))

#### El grid search hecho de forma manual es correcto, pero depende mucho de cómo se haga la partición de los datos
#### Además, diferentes particiones pueden dar distinto valores óptimos de los hiperparámetros

#### Una forma más fiable es evaluar la mejor combinación de parámetros a los diferentes folds de la validación crzada
#### Esto reduce la variabilidad debida a la elección de un único conjunto de validación

#### Por ello, podemos usar la clase GridSearchCV integrado en scikit-learn:

normalizador=StandardScaler()
normalizador.fit(x_train)
xn_train = normalizador.transform(x_train)

normalizador.fit(x_test)
xn_test = normalizador.transform(x_test)


#GRID SEARCG CON VALIDACIÓN CRUZADA (integrada en sciklearn)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ["l1","l2"]}  #especifico los grids que voy a probar

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(LogisticRegression(solver="saga", max_iter = 1000), param_grid, cv=5)
                        #  return_train_score=True) # Se necesita par evitar warnings a partir de la 0.21
#por defecto, cv ya es igual a 5

grid_search.fit(xn_train, y_train)

print("Mejor resultado (media) en validación cruzada: {:.2f}".format(grid_search.best_score_))
print("Mejor clasificador encontrado: {}", grid_search.best_estimator_)
print("Evaluación sobre el conjunto de test: {:.2f}".format(grid_search.score(xn_test, y_test)))

####El modelo de regresión logística optimizado mediante GridSearchCV con validación cruzada
#### alcanza una precisión media de 0.96 en los conjuntos de validación, 
#### lo que indica que la combinación de hiperparámetros elegida (C=1, penalty='l1', solver='saga') generaliza bien 
#### durante el entrenamiento. 

#### Al evaluar este mismo modelo sobre el conjunto de test independiente, obtenemos una precisión de 0.97, 
#### confirmando que el modelo mantiene un rendimiento alto y consistente sobre datos no vistos, lo que sugiere una buena 
#### generalización.

#### Ahora podemos ver, a traves de cv_results_ los resultados de todas esas combinaciones.

import pandas as pd
# convertimos a Dataframe los resultados del gridsearchcv anterior
results = pd.DataFrame(grid_search.cv_results_)

results_sorted = results.sort_values("rank_test_score") #para que me salgan los  primeros mejores ordenados

# Mostramos las cinco primeras filas
display(results_sorted.head()) 

#### INFORMACIÓN DE ESTAS TABLAS
    #### mean_fit_time -> tiempo medio que tarda en entrenarse el modelo en cada split de la validación cruzada
    #### std_fit_time -> desviación estándar del tiempo de entrenamiento entre split
    #### mean_score_time -> tiempo medio que tarda en evaluar el modelo en cada split
    #### std_score_time -> desviación estándar del tiempo de evaluación entre split
    
    #### param_C -> valor del hiperparámetro C (controla la fuerza de la regularización)
    #### param_penalty -> tipo de penalización usada (`l1` o `l2`)
    #### params -> diccionario con todos los hiperparámetros de la combinación
    
    #### split0_test_score … split4_test_score -> accuracy del modelo en cada pliegue de la validación cruzada
    #### mean_test_score -> media de las puntuaciones de todos los pliegues; indica capacidad de generalización estimada
    #### std_test_score -> desviación estándar de las puntuaciones; indica estabilidad del modelo entre pliegues
    
    #### rank_test_score -> posición relativa de la combinación según `mean_test_score`, 1 = mejor, 2 = segunda mejor, etc.


#### También podemos ver las matrices de confusión del mejor modelo de la siguiente manera:
#### mejor modelo era l1, C=1

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logreg = LogisticRegression(solver="saga", penalty="l1", C=1, max_iter=5000).fit(xn_train, y_train)

# Hacemos predicciones
y_pred = logreg.predict(xn_test)

# Calculamos la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# La mostramos visualmente
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión - Regresión Logística")
plt.show()

#### como podemos observar, obtenemos ciertos errores fuera de la diagonal
#### Por ej., cuando se predice 0, pero el valor real es 1

#### Pero en general los aciertos entre la predicción y la clasificación real se concentra en la diagonal

# ###  7. Mostrar la predicción sobre ejemplos concretos

from matplotlib.pyplot import imshow

imshow(xn_train[0].reshape(8,8))

imshow(xn_train[6].reshape(8,8))

#### También podemos hacerlos a la vez usando el for: 

indices_ejemplos = [0, 6]

for i in indices_ejemplos:
    X_example = xn_train[i].reshape(1, -1)  # mantenemos la forma
    y_true = y_train[i]   # etiqueta real
    y_pred = lr.predict(X_example)[0]  # predicción del modelo

    # Mostrar la imagen
    plt.imshow(xn_train[i].reshape(8, 8))
    plt.title(f"Etiqueta real: {y_true}, Predicción: {y_pred}")
    plt.axis('off')
    plt.show()

