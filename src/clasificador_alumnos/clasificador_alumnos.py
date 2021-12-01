"""
Estudiantes felices y no felices
--------------------------------
    Clasificar a los estudiantes por felices y no felices. Utilizando `Dataset.Alumnos`. [3: `feliz`, 2: `no feliz`|`Neutral` ].

Instrucciones:
--------------
    `1`-Seleccionar un algoritmo: `Algoritmo.`

    `2`-El dataset solo incluirá los 8 atributos tipo Likert y la clase.

    `3`-Emplear `Sklearn.SGridSearchCV` para detectar la mejor configuración del algoritmo.

    `4`-Con la mejor configuración, realizar una validación cruzada con 5 pliegues, Mostrar resultados.
"""
import os
from re import L
from matplotlib import pyplot as plot
import numpy as np

from utils import load_dataset
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.naive_bayes import GaussianNB

from utils.matrix_conf import get_confusion_matrix, plot_CM


def run_clf_alumnos():
    path = f"{os.path.dirname(__file__)}\\data"
    columns = [
        "Vinculos_afectivos",
        "Superación_logros",
        "Celebraciones",
        "Actividades_lúdicas",
        "Salud",
        "Bienes_materiales",
        "Desarrollo_personal",
        "Independencia",
        "Estado_emocional",
    ]

    DATASET = load_dataset(
        dir_data=path,
        columns=columns,
        file_name_data="felicidad_estudiantes.csv",
        file_name_desc="felicidad_estudiantes.rst",
    )

    n_samples = len(DATASET)
    data, target, target_names = DATASET.data, DATASET.target, DATASET.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.5, random_state=0
    )

    params_NB = {"var_smoothing": np.logspace(0, -9, num=100)}

    clf = GridSearchCV(
        estimator=GaussianNB(),
        param_grid=params_NB,
        cv=StratifiedKFold(n_splits=5),  # use any cross validation technique
        verbose=1,
        scoring="accuracy",
    )

    clf.fit(X_train, y_train)
    params = clf.cv_results_["params"]
    for index in range(len(params)):
        print(f"#{index} var_smoothing: {params[index]['var_smoothing']}")

    clf_cv = cross_validate(
        estimator=GaussianNB(var_smoothing=clf.best_params_["var_smoothing"]),
        X=data,
        y=target,
        cv=StratifiedKFold(5),
        return_estimator=True,
    )

    estimador = clf_cv["estimator"][0]
    cm, report = get_confusion_matrix(estimador=estimador, X=data, y=target)
    plot_CM(cm=cm, target_names=target_names, report=report)
