"""
:mod:`scikit_learn.utils`: Modulo de utilidades para los proyectos que usan :mod:`skilearn`
"""
from .load_data import load_dataset
from .matrix_conf import get_confusion_matrix, plot_CM
from .path import PATH_DATA, PATH_ROOT

__all__ = [
    "load_dataset",
    "get_confusion_matrix",
    "plot_CM",
    "PATH_DATA",
    "PATH_ROOT",
]
