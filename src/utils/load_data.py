from typing import List
from sklearn import datasets
from sklearn.utils import Bunch
from sklearn.datasets._base import load_csv_data

from .path import PATH_DATA


def _load_file(file_data, columns, file_descr=None, *, return_X_y=False):
    """Carga el dataset y su descripción a memoria. Realiza llamados a metodos del modulo :mod:`sklearn.datasets` de la libreria :lib:`sklearn`, para cargar los datos de la misma manera que los metodos de carga incluidos en la libreria :lib:`sklearn`.

    Parameters
    ----------
    file_data (srt):
        Nombre y extensión de archivo que contiene la data del dataset
    file_descr (srt):
        Nombre y extensión de archivo que contiene la descripción del dataset
    columns (List[str]):
        Lista de los nombres de las columnas del dataset
    return_X_y (bool), default=False:
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.
        .. versionadded:: 0.18
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : {ndarray, dataframe} of shape (150, 4)
        The data matrix. If `as_frame=True`, `data` will be a pandas
        DataFrame.
    target: {ndarray, Series} of shape (150,)
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    frame: DataFrame of shape (150, 5)
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.

        .. versionadded:: 0.23
    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.

        .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.18
    """
    if return_X_y:
        data, target, target_names = load_csv_data(data_file_name=file_data)
        return data, target
    else:
        data, target, target_names, fdescr = load_csv_data(
            data_file_name=file_data, descr_file_name=file_descr
        )

        feature_names = columns
        frame = None

        return Bunch(
            data=data,
            target=target,
            frame=frame,
            target_names=target_names,
            DESCR=fdescr,
            feature_names=feature_names,
        )


def load_dataset(
    dir_data: str,
    columns: List[str],
    file_name_data: str,
    file_name_desc: str = None,
    return_X_y: bool = False,
):
    """Carga a memoria la data y definición del dataset indicado.

    Parameters
    ----------
    dir_data : str
        Ruta de la ubicacion de los archivos del dataset (data, descripcion).
    columns : List[str]
        Lista de nombres de las columnas del dataset.
    file_name_data : str
        Nombre y extension del archivo que contiene la data del dataset.
    file_name_desc : str
        Nombre y extension del archivo que contiene la descripción del dataset.
    return_X_y : bool, default=False
        Si es True retorna los unicamente los valores de `data` y `target`.

    Returns
    -------
    Dict[str, Any]
        Diccionario que contiene la desfinición y la data del dataset.
    """
    import os
    import shutil

    # RUTA DEL MODULO DONDE SE ALMACENAN LOS DATASET
    path_sklearn = os.path.dirname(datasets.__file__)

    if not file_name_data == None:
        path_sklearn_data = f"{path_sklearn}\\data\\{file_name_data}"
        # VERIFICAMOS SI EL ARCHIVO EXISTE Y LO ELIMINAMOS
        if _checkFileExistance(path_sklearn_data):
            os.remove(path_sklearn_data)
        # COPIAMOS EL ARCHIVO A LA RUTA
        shutil.copyfile(f"{dir_data}\\{file_name_data}", path_sklearn_data)

    if not file_name_desc == None:
        path_sklearn_desc = f"{path_sklearn}\\descr\\{file_name_desc}"
        # VERIFICAMOS SI EL ARCHIVO EXISTE Y LO ELIMINAMOS
        if _checkFileExistance(path_sklearn_desc):
            os.remove(path_sklearn_desc)
        # COPIAMOS EL ARCHIVO A LA RUTA
        shutil.copyfile(f"{dir_data}\\{file_name_desc}", path_sklearn_desc)

    return _load_file(
        file_data=file_name_data,
        file_descr=file_name_desc,
        columns=columns,
        return_X_y=return_X_y,
    )


def _checkFileExistance(filePath):
    """Valida si el archivo existe en la ruta especificada. `Estoy dudado si quitarla...`

    Parameters
    ----------
    filePath : str
        Ruta del archivo que se desea verificar

    Returns
    -------
    bool
        True si el archivo existe en la ruta.
    """
    try:
        with open(filePath, "r") as f:
            return True
    except FileNotFoundError as e:
        return False
    except IOError as e:
        return False


if __name__ == "__main__":
    """Metodo de entrada para probar la carga de dataset a memoria"""

    # DATASET DEL TIPO X,y
    clmn = ["SEMESTRE", "CALIFICACION"]
    X, y = load_dataset(
        dir_data=PATH_DATA(),
        columns=clmn,
        file_name_data="calificacion.csv",
        return_X_y=True,
    )
    print(X, y)

    # DEL TIPO DATASET DE CLASIFICACION
    clmn = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280/od315_of_diluted_wines",
        "proline",
    ]
    data = load_dataset(
        dir_data=PATH_DATA(),
        file_name_data="zoo.csv",
        file_name_desc="zoo.rst",
        columns=clmn,
    )
    print(data)
