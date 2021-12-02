def PATH_ROOT() -> str:
    """`PATH.ROOT`: Ruta relativa del proyecto."""
    import os

    return os.path.realpath(f"{os.path.dirname(__file__)}\\..\\..")


def PATH_DATA() -> str:
    """`PATH.ROOT.DATA`: Ruta relativa del directorio `DATA`."""
    import os

    return os.path.realpath(f"{os.path.dirname(__file__)}\\..\\..\\data")
