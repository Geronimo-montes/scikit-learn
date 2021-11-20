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
import itertools
import os
import numpy as np
from math import trunc
from typing import Dict
import matplotlib.pyplot as plt

from decimal import Decimal, getcontext

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils.load_data import load_dataset


def run():
    """Metodo estatico de estrada. En este metodo se instancian los algoritmos a
    evaluar ademas de cargar el conjunto de datos con el que se va a evaluar a
    los algoritmos de clasificacion."""
    # ISNTANCIAMOS LOS CALSIFICADORES PARA LA TESTEO DE RENDIMIENTO
    clasificadores = {
        "Neuronal networks": MLPClassifier(random_state=1, max_iter=50),
        "K-nearest neighbors": KNeighborsClassifier(),
        # "Naive bayes": GaussianNB(),
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
    winner = performance_test.run_test()
    # GENERAMOS LA MATRIZ DE CONFUCION Y LAS METRICAS CORRESPONDIENTES
    performance_test.run_matriz_conf(winner)


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
        data, target = self.__DATASET.data, self.__DATASET.target

        # CORREMOS LA VALIDACION CRUZADA POR CADA CLASIFICADOR DE LA LISTA
        VALUES = {}
        VALUES_TBL = []
        ROWS_NAME = []
        ACURACY = []
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
                winner = {
                    name: resultado["estimator"][self.__CV_METOD.get_n_splits() - 1]
                }

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
                    name,
                    np.mean(
                        values.get("scores"),
                    ),
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

        return winner

    def run_matriz_conf(self, dic_clasificador: Dict[str, any]):
        """Construye la matriz de confunsión para el conjunto de datos usado en la
        prueba de rendimiento haciendo uso del clasificador ganador.

        Args:
            dic_clasificador (Dict[str, any]): Diccionario que contiene el
            clasificador ganador como valor y nombre de este como clave de acceso.
        """
        for key, value in dic_clasificador.items():
            clasificador_name = key
            clasificador = value

        data, trgt, class_names = (
            self.__DATASET.data,
            self.__DATASET.target,
            self.__DATASET.target_names,
        )

        test_pred = clasificador.predict(data)
        cm = confusion_matrix(trgt, test_pred)
        report = classification_report(y_true=trgt, y_pred=test_pred, output_dict=True)

        report_list = [[], [], [], []]
        for key, value in report.items():
            if key != "macro avg" and key != "weighted avg" and key != "accuracy":
                row = report.get(key)

                report_list[0].append(str("{:.2f}".format(row.get("precision"))))
                report_list[1].append(str("{:.2f}".format(row.get("recall"))))
                report_list[2].append(str("{:.2f}".format(row.get("f1-score"))))
                report_list[3].append(str("{:.2f}".format(report.get("accuracy"))))

        self.plot_confusion_matrix(
            cm=cm,
            target_names=class_names,
            title=f"Matriz de confusión ({clasificador_name})",
            cmap=plt.get_cmap("Set2"),
            normalize=False,
            report_data=report_list,
        )

    def plot_confusion_matrix(
        self,
        cm,
        target_names,
        report_data,
        title="Confusion matrix",
        cmap=None,
        normalize=True,
    ):
        """Despliega la matriz de confusión de manera grafica.

        Args:
            cm (any): Matriz de confusion en forma de matriz
            target_names (List[str]): Lista de nombre correspondientes a las clases del
             conjunto de datos
            report_data (List[str]]): Lista de metricas obtenidas a partir de la matriz
             de confusión (presición, recall, f1, acurracy)
            title (str, optional): Titulo del grafico. Defaults to "Confusion matrix".
            cmap ([type], optional): Mapa de colores a utilizar. Defaults to None.
            normalize (bool, optional): Modo de representacion de los datos.
            True: False . Defaults to True.
        """
        if cmap is None:
            cmap = plt.get_cmap("Blues")

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks([])
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(
                    j,
                    i,
                    "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            else:
                plt.text(
                    j,
                    i,
                    "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        table = plt.table(
            cellText=report_data,
            colLabels=target_names,
            rowLabels=["precision", "recall", "f1-score", "Accuracy"],
            colLoc="center",
            rowLoc="center",
            loc="bottom",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1)

        plt.subplots_adjust(bottom=0.2)
        plt.ylabel("Valor esperado")
        plt.show()
