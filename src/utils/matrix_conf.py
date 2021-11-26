import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def get_confusion_matrix(estimador, X, y):
    """Genera la matriz de confusion del algoritmo indicado, con los valores X,y proporcionados

    Parameters
    ----------
    `estimador` : estimator object
        Objeto que implementa la interfaz scikit-learn estimator.

    `X` : array-like of shape (n_samples, n_features)
        Puede ser `List` or `Array`.

    `y` : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Variable objetivo a intentar predecir en el caso del aprendizaje supervisado.

    Returns
    -------
    `cm` : ndarray of shape (n_classes, n_classes)
        Matriz de confusion del modelo y datos.

    `report` : str or dict
        Text summary of the precision, recall, F1 score for each class. Dictionary returned if output_dict is True. Dictionary has the following structure:
    """
    y_pred = estimador.predict(X)

    cm = confusion_matrix(
        y_true=y,
        y_pred=y_pred,
    )

    clf_report = classification_report(
        y_true=y,
        y_pred=y_pred,
        output_dict=True,
    )

    report = [[], [], [], []]
    for key, value in clf_report.items():
        is_mavg = key != "macro avg"
        is_wavg = key != "weighted avg"
        is_accr = key != "accuracy"

        if is_mavg and is_wavg and is_accr:
            r = clf_report.get(key)

            report[0].append(str("{:.2f}".format(r.get("precision"))))
            report[1].append(str("{:.2f}".format(r.get("recall"))))
            report[2].append(str("{:.2f}".format(r.get("f1-score"))))
            report[3].append(str("{:.2f}".format(clf_report.get("accuracy"))))

    return cm, report


def plot_CM(
    cm,
    target_names,
    report,
    title="Confusion matrix",
    cmap=None,
    normalize=True,
):
    """Despliega la matriz de confusión de manera grafica.

    Args:
        `cm` (any) :
            Matriz de confusion en forma de matriz
        `target_names` (List[str]) :
            Lista de nombre correspondientes a las clases del conjunto de datos
        `report` (List[str]]) :
            Lista de metricas obtenidas a partir de la matriz de confusión (presición, recall, f1, acurracy)
        `title` (str, optional) :
            Titulo del grafico. Defaults to "Confusion matrix".
        `cmap` ([type], optional) :
            Mapa de colores a utilizar. Defaults to None.
        `normalize` (bool, optional) :
            Modo de representacion de los datos.
        True: False . Defaults to True.
    """
    if cmap is None:
        cmap = plt.get_cmap("Set2")

    plt.figure(figsize=(6, 4))
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
        # COLOR DEL TEXTO
        color = "white" if cm[i, j] > thresh else "black"
        # FORMATO DEACUERDO CON DECIMALES A SIN DECIMALES
        sformat = "{:0.4f}".format(cm[i, j]) if normalize else "{:,}".format(cm[i, j])
        # PLOT
        plt.text(x=j, y=i, s=sformat, horizontalalignment="center", color=color)

    table = plt.table(
        cellText=report,
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
