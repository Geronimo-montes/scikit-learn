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
import numpy as np

from utils.path import PATH_DATA
from utils.load_data import load_dataset
from utils.matrix_conf import plot_CM
from utils.matrix_conf import get_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB


def pract_03_run():
    """Practica #3 Estudiantes Felices y no Felices"""

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
    # CARGA DE DATASET A MEMORIA CON LA LIBRERIA UTILS
    DATASET = load_dataset(
        dir_data=PATH_DATA(),
        columns=columns,
        file_name_data="felicidad_estudiantes.csv",
        file_name_desc="felicidad_estudiantes.rst",
    )
    # OBTENEMOS LOS DATOS DE DATASET
    X, y, target_names = DATASET.data, DATASET.target, DATASET.target_names
    # SEGMENTAMOS EL GRUPO DE DATOS PARA LA PRUEBA
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.5, random_state=0)
    # GENERAMOS EL CONJUNTO DE CONFIGURACIONES A PROBAR
    params_NB = {"var_smoothing": np.logspace(0, -9, num=20)}
    clf = GridSearchCV(
        estimator=GaussianNB(),
        param_grid=params_NB,
        cv=StratifiedKFold(n_splits=5),  # use any cross validation technique
        verbose=1,
        scoring="accuracy",
    )

    # PROBAMOS CADA UNA DE LAS CONFIGURACIONES GENERADAS
    clf.fit(X_train, y_train)

    # ALMACENAMOS LOS RESULTADOS DE LAS PRUEBAS PARA MOSTRARLAS
    params = clf.cv_results_
    headers = f"{clf.best_params_}\n| ## | var_smoothing | MEAN FIT TIME |MEAN TIME SCORE| I #1 | I #2 | I #3 | I #4 | I #5 | MEAN | RANK  |"
    print(headers)

    for index in range(len(params["split0_test_score"])):
        print(
            "|{:2d}| {:1.11f} |  {:1.10f} |  {:1.10f} | {:1.2f} | {:1.2f} | {:1.2f} | {:1.2f} | {:1.2f} | {:1.2f} | {:5d} |".format(
                index,
                params["params"][index]["var_smoothing"],
                params["mean_fit_time"][index],
                params["mean_score_time"][index],
                params["split0_test_score"][index],
                params["split1_test_score"][index],
                params["split2_test_score"][index],
                params["split3_test_score"][index],
                params["split4_test_score"][index],
                params["mean_test_score"][index],
                params["rank_test_score"][index],
            )
        )

    # CON LA MEJOR FONFICURACION GENERAMOS LA MATRIZ DE CONFUSION
    clf_cv = cross_validate(
        estimator=GaussianNB(var_smoothing=clf.best_params_["var_smoothing"]),
        X=X,
        y=y,
        cv=StratifiedKFold(5),
        return_estimator=True,
    )

    estimador = clf_cv["estimator"][0]
    cm, report = get_confusion_matrix(estimador=estimador, X=X, y=y)
    plot_CM(cm=cm, target_names=target_names, report=report)
