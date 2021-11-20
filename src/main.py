import sys
from typing import List

from performance_test.performance_test import run as pt_run
from calificacion_reg_lin.calificacion_reg_linel import run as crl_run


class Main:
    @staticmethod
    def run(num_prac: List[str] = ["1"]):
        # PRACTICA #1
        if "1" in num_prac:
            pt_run()
        # PRACTICA #2
        if "2" in num_prac:
            crl_run()


if __name__ == "__main__":
    num_prac = sys.argv
    Main.run(num_prac=num_prac)
