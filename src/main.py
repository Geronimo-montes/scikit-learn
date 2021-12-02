import sys
from typing import List

from practicas.pract_01 import pract_01_run
from practicas.pract_02 import pract_02_run
from practicas.pract_03 import pract_03_run


class Main:
    @staticmethod
    def run(num_prac: List[str] = ["1", "2", "3"]):
        print(num_prac)
        # PRACTICA #1
        if "1" in num_prac:
            pract_01_run()
        # PRACTICA #2
        if "2" in num_prac:
            pract_02_run()
        # PRACTICA #3
        if "3" in num_prac:
            pract_03_run()


if __name__ == "__main__":
    Main.run(num_prac=sys.argv) if len(sys.argv) > 1 else Main.run()
