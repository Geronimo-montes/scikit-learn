"""
ALGORITMOS A UTILIZAR
---------------------
° K-nearest neighbors
° Naive bayes
° Neuronal networks

INSTRUCCIONES
-------------
° Determina el accuracy de los algoritmos empleando una validación cruzada con 5 pliegues 
(5 fold validation).
° Despliega en una tabla el accuracy de cada iteración de los 3 clasificadores.
° Explica el algoritmo con mejor rendimiento
° Con el algoritmo de clasificación con mejor rendimiento despliega la matriz de confusión y explícala.
° Obtén las métricas precisión, recall (exhaustividad), f1 score y accuracy del algoritmo con mejor accuracy.

"""
import os
import numpy as np
from math import trunc
from typing import Dict
import matplotlib.pyplot as plt

from decimal import Decimal, getcontext

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold


from utils.load_data import load_dataset
from utils.matrix_conf import get_confusion_matrix, plot_CM


def run():
    """Metodo estatico de estrada. En este metodo se instancian los algoritmos a
    evaluar ademas de cargar el conjunto de datos con el que se va a evaluar a
    los algoritmos de clasificacion."""
    # ISNTANCIAMOS LOS CALSIFICADORES PARA LA TESTEO DE RENDIMIENTO
    clasificadores = {
        "Neuronal networks": MLPClassifier(random_state=1, max_iter=50),
        "K-nearest neighbors": KNeighborsClassifier(),
        "Naive bayes": GaussianNB(),
    }

    columns = [
        "andhair",
        "feathers",
        "eggs",
        "milk",
        "airborne",
        "aquatic",
        "predator",
        "toothed",
        "backbone",
        "breathes",
        "venomous",
        "fins",
        "legs",
        "tail",
        "domestic",
        "catsize",
        "type",
    ]
    # INICIAMPOS EL TEST, PASAMOS EL DATASET Y EL DICCIONARIO DE LOS CLASIFICADORES A LA CLASE ENCARGADA DE EFECTUAR EL TEST
    performance_test = PerformanceTest(
        dataset=load_dataset(
            dir_data=f"{os.path.dirname(__file__)}\\data",
            file_name_data="zoo.csv",
            file_name_desc="zoo.rst",
            columns=columns,
        ),
        clasificadores=clasificadores,
    )
    # OBTENEMOS UN DICCIONARIO CON EL NOMBRE DEL CLASIFICADOR COMO KEY Y EL CLASIFICADOR ENTRENADO COMO VALOR
    performance_test.run_test()


class PerformanceTest:
    """Test de rendimiento que evalua un conjunto de algoritmos de clasificaicon con
    un mismo conjunto de datos, haciendo uso de validacion cruzada de 5 pliegues."""

    def __init__(
        self, dataset, clasificadores: Dict[str, any], n_iteracion_cv: int = 5
    ) -> None:
        getcontext().prec = 2
        self.__DATASET = dataset
        self.__CLASIFICADORES = clasificadores
        self.__CV_METOD = StratifiedKFold(n_splits=n_iteracion_cv)

    def run_test(self):
        """Realiza la segmentacion de los grupos para la validacion cruzada e itera el
         total de pliegues indicado, entrenando y evaluando a cada uno de los algoritmos
          de clasificacion. Obtiene el de mayor rendimiento y despliega los resultados
           de manera grafica.

        Returns:
            Clasificador: Algoritmo con el mayor rendimiento registrado en la prueba.
            El clasificador retornado ya se encuentra entrenado
        """
        # DEL DATASET TOMAMOS LOS VALORES DE LA DATA Y LA COLUMNA QUE REPRESENTA LA CLASE
        data, target, target_names = (
            self.__DATASET.data,
            self.__DATASET.target,
            self.__DATASET.target_names,
        )

        # CORREMOS LA VALIDACION CRUZADA POR CADA CLASIFICADOR DE LA LISTA
        VALUES = {}
        VALUES_TBL = []
        ROWS_NAME = []
        mayor = 0
        for name, clasificador in self.__CLASIFICADORES.items():
            resultado = cross_validate(
                estimator=clasificador,
                X=data,
                y=target,
                cv=self.__CV_METOD,
                return_estimator=True,
            )
            scores = resultado["test_score"]
            if np.mean(scores) > mayor:
                mayor = np.mean(scores)
                title = f"Matriz de confusión ({name})"
                estimador = resultado["estimator"][0]

            VALUES[name] = {
                "format": list(Decimal(elem) for elem in scores),
                "scores": scores,
            }

            VALUES_TBL.append(list(str(trunc(elem * 100)) + "%" for elem in scores))
            ROWS_NAME.append(name)

        # LABELS GRFICO
        LABELS = list(
            str(f"ITER#{iter}") for iter in range(self.__CV_METOD.get_n_splits())
        )
        x = np.arange(len(LABELS))
        width = 1

        # DETALLE DE CADA BARRA A COLOCAR
        n = 0
        for name, values in VALUES.items():
            plt.bar(
                x=x + (width / len(LABELS)) * n,
                height=values.get("format"),
                width=(width / len(LABELS)),
                label="{:30s}Acuracy: {:.2f}".format(
                    name, np.mean(values.get("scores"))
                ),
                align="center",
            )
            n += 1

        plt.subplots_adjust(left=0.3, bottom=0.2)

        plt.title("Acuracy por iteración")
        plt.ylabel("Presisión")
        plt.xticks([])
        plt.legend()

        table = plt.table(
            cellText=VALUES_TBL,
            rowLabels=ROWS_NAME,
            colLabels=LABELS,
            colLoc="center",
            rowLoc="center",
            loc="bottom",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1)

        plt.show()

        cm, report = get_confusion_matrix(estimador=estimador, X=data, y=target)
        plot_CM(cm=cm, target_names=target_names, title=title, report=report)
