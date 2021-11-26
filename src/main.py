import sys
from typing import List

from performance_test.performance_test import run as pt_run
from calificacion_reg_lin.calificacion_reg_linel import run as crl_run
from clasificador_alumnos.clasificador_alumnos import run_clf_alumnos as clf_alum_run


class Main:
    @staticmethod
    def run(num_prac: List[str] = ["1", "2", "3"]):
        print(num_prac)
        # PRACTICA #1
        if "1" in num_prac:
            pt_run()
        # PRACTICA #2
        if "2" in num_prac:
            crl_run()
        # PRACTICA #3
        if "3" in num_prac:
            clf_alum_run()


if __name__ == "__main__":
    Main.run(num_prac=sys.argv) if len(sys.argv) > 1 else Main.run()
