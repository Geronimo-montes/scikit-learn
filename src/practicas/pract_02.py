"""
° Elegir un algoritmo de regresión lineal.
° Obtener el promedio de calificaciones en cada semestre (dataset). El atributo es el semestre y la predicción es la calificación.
° Trazar una gráfica con Matplotlib  con las calificaciones y los semestres
° Generar el modelo de la forma y=ax+b con el algoritmo seleccionado
° Predecir el 9no semestre con el modelo y graficarlo junto con la tendencia de datos
° Describir la predicción (acertada. no acertada, regular)
"""
import numpy as np

from utils.path import PATH_DATA
from utils.load_data import load_dataset

from typing import List
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression


def pract_02_run():
    file_data_x = "calificacion.csv"
    columns = ["SEMESTRE", "CALIFICACION"]
    # CARGAMOS LOS DATOS A MEMORIA INDICADO QUE RETORNO LOS DATOS DE LA FORMA X,y
    y, x = load_dataset(
        dir_data=PATH_DATA(),
        columns=columns,
        file_name_data=file_data_x,
        return_X_y=True,
    )

    # AGREGAMOS UN INDICE AL ARRAY DE X
    X = x[:, np.newaxis]
    regresion_lineal(values=[X, y], predecir=[9])


def graficar(
    values: List[List[int]] = None,
    prediccion: List[List[int]] = None,
    reg_lin: List[List[float]] = None,
    title="Promedio por Semestre",
    ecuacion=None,
):
    """Grafica los valores en un plot

    Parameters
    ----------
    values : `List[List[int]`]
        Lista de pares de coordenadas `(x,y)` que representan los valores de entrenamiento.
    prediccion : `List[List[int]]`, default None
        Lista de pares de coordenadas `(x,y)` que representan los valores de predicción del modelo.
    reg_lin : `List[List[float]]`, default None
        Lista de pares de coordenadas `(x,y)` que representan la tendencia lineal de los datos.
    title : `str`, default "Promedio por Semestre".
        Titulo del grefico
    ecuacion : `str`, default `None`
        Ecuación de la grafica de regresión lineal.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if not values == None:
        ax.scatter(values[0], values[1], label="Calificación", color="red")

        for i in range(len(values[0])):
            ax.annotate(
                str(values[1][i][0]),
                xy=(values[0][i], values[1][i]),
                xytext=(values[0][i][0] + 0.1, values[1][i][0] + 0.01),
            )

    if not prediccion == None:
        ax.scatter(prediccion[0], prediccion[1], label="Predicción", color="yellow")

        for i in range(len(prediccion[0])):
            ax.annotate(
                str(prediccion[1][i][0]),
                xy=(prediccion[0][i], prediccion[1][i]),
                xytext=(prediccion[0][i][0] + 0.1, prediccion[1][i][0] + 0.01),
            )

    if not reg_lin == None:
        if len(reg_lin) == 2:
            ax.plot(reg_lin[0], reg_lin[1], label="Regresión Lineal")
        else:
            ax.plot(reg_lin, reg_lin, label="Ridge_CV")

    plt.title(ecuacion)
    plt.suptitle(title)
    plt.xlabel("Semestre")
    plt.ylabel("Calificacion")
    plt.legend()
    plt.show()


def regresion_lineal(values: List[List[float]], predecir: List[float]):
    """Recibe como parametre un conjunto de coordenadas cartesianas. Obtiene la tendencia lineal de los datos y predice los valores indicados.

    Parameters
    ----------
    values : List[List[float]]
        Par de coordenada X,y del plano cartesiano. Valores provinientes del dataset cargado.
    predecir : List[float]
        Conjunto de datos que se desea predecir con el modelo de regresion lineal
    """
    x, y = values[0], values[1]
    # Presentamos los datos al metodo
    reg_lin = LinearRegression(n_jobs=2, positive=True)
    reg_lin.fit(x, y)

    # GENERAMOS EL STRING DE LA ECUACION
    ecuacion = f"Y = {reg_lin.coef_[0]} *  X  + {reg_lin.intercept_}"

    # PARA GRAFICAR LA REGRESION LINEAL CALCLAMOS LOS PARES DE CORDDENADOS MIN-MAX
    min_pt = x.min() * reg_lin.coef_[0] + reg_lin.intercept_
    max_pt = x.max() * reg_lin.coef_[0] + reg_lin.intercept_

    # PREDECIMOS LOS VALORES MEDIANTE EL MODELO CREADO
    x_pred = np.array(predecir)[:, np.newaxis]
    y_pred = reg_lin.predict(x_pred)

    # GRAFICAMOS TODOS LOS CONUNTOS DE VALORES
    graficar(
        values=[x, y],
        prediccion=[x_pred, y_pred],
        reg_lin=[[x.min(), x.max()], [min_pt, max_pt]],
        ecuacion=ecuacion,
    )
