"""Las células del turno: cada módulo fabrica una `Cell` y la mayoría arranca "La célula que...".

`handle` es la excepción a medias: no es una célula, es el turno de sistema que el
`worker` manda, y va acá porque es la mitad del worker que se lee. Lo que estas
células toman del REPL es poco y nombrable: el Protocol `Environment` (executor),
el default de `HERRAMIENTAS` (handle), el colector `Verbos` (planner) y
`sin_acento` (citas). Ninguna sabe si el código corre en proceso o en un hijo.
"""

from .citas import citada, citadas, inventada
from .executor import executor, extract_code, render
from .grounding import grounded
from .handle import Handle, render_handle
from .planner import Registro, planner, render_plan
from .worker import window, worker

__all__ = [
    "Handle",
    "Registro",
    "citada",
    "citadas",
    "executor",
    "extract_code",
    "grounded",
    "inventada",
    "planner",
    "render",
    "render_handle",
    "render_plan",
    "window",
    "worker",
]
