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
import numpy as np
from scipy.sparse import data

from utils.path import PATH_DATA
from utils.load_data import load_dataset
from utils.matrix_conf import plot_CM
from utils.matrix_conf import get_confusion_matrix

from math import trunc
from typing import Dict
from decimal import Decimal
from decimal import getcontext
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


def pract_01_run():
    """Metodo estatico de estrada. En este metodo se instancian los algoritmos a
    evaluar ademas de cargar el conjunto de datos con el que se va a evaluar a
    los algoritmos de clasificacion."""

    clmns = [
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
    data = load_dataset(PATH_DATA(), clmns, "zoo.csv", "zoo.rst")

    # ISNTANCIAMOS LOS CALSIFICADORES PARA LA TESTEO DE RENDIMIENTO
    clasificadores = {
        "Neuronal networks": MLPClassifier(random_state=1, max_iter=50),
        "K-nearest neighbors": KNeighborsClassifier(),
        "Naive bayes": GaussianNB(),
    }

    performanceTest(
        dataset=data,
        clasificadores=clasificadores,
        cv=StratifiedKFold(n_splits=5),
    )


def performanceTest(dataset, clasificadores: Dict[str, any], cv):
    """Test de rendimiento que evalua un conjunto de algoritmos de clasificaicon con
    un mismo conjunto de datos, haciendo uso de validacion cruzada de 5 pliegues.

    Returns:
        Clasificador: Algoritmo con el mayor rendimiento registrado en la prueba.
        El clasificador retornado ya se encuentra entrenado
    """
    # DEL DATASET TOMAMOS LOS VALORES DE LA DATA Y LA COLUMNA QUE REPRESENTA LA CLASE
    X, y, target_names = (dataset.data, dataset.target, dataset.target_names)

    # CORREMOS LA VALIDACION CRUZADA POR CADA CLASIFICADOR DE LA LISTA
    VLS = []
    VALUES_TBL = []
    ROWS_NAME = []
    mayor = 0

    for nm, estimador in clasificadores.items():
        r = cross_validate(estimator=estimador, X=X, y=y, cv=cv, return_estimator=True)
        scrs = r["test_score"]

        VLS.append({"nm": nm, "sfrmt": list(Decimal(e) for e in scrs), "scrs": scrs})
        VALUES_TBL.append(list(str(trunc(e * 100)) + "%" for e in scrs))
        ROWS_NAME.append(nm)

        if np.mean(scrs) > mayor:
            mayor = np.mean(scrs)
            title = f"Matriz de confusión ({nm})"
            estimador = r["estimator"][0]

    # LABELS GRFICO
    LABELS = list(str(f"ITER#{iter}") for iter in range(cv.get_n_splits()))
    x = np.arange(len(LABELS))
    width = 1

    # DETALLE DE CADA BARRA A COLOCAR
    for i in range(len(VLS)):
        plt.bar(
            x=x + (width / len(LABELS)) * i,
            height=VLS[i].get("sfrmt"),
            width=(width / len(LABELS)),
            label=f"{VLS[i]['nm']:30s}Acuracy: {np.mean(VLS[i]['scrs']):.2f}",
        )

    plt.subplots_adjust(left=0.3, bottom=0.2)

    table = plt.table(cellText=VALUES_TBL, rowLabels=ROWS_NAME, colLabels=LABELS)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1)

    plt.title("Acuracy por iteración")
    plt.ylabel("Presisión")
    plt.xticks([])
    plt.legend()
    plt.show()

    cm, report = get_confusion_matrix(estimador=estimador, X=X, y=y)
    plot_CM(cm=cm, target_names=target_names, title=title, report=report)
