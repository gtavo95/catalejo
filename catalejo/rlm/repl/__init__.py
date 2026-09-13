"""El REPL: dónde corre el código que el modelo escribe, y qué tiene a mano ahí adentro.

Esta carpeta no sabe qué es una `Cell` ni un `Log`. Lo único que toma de `core` es
`PlanOp`, un tipo de dato. Es la mitad que se puede mirar sin el álgebra: el
Protocol, sus dos implementaciones (en proceso y en proceso hijo), el `grep`, y los
verbos del plan como builtins. Las células que la usan viven en `celulas/`, y la
regla es que la dependencia va en una sola dirección: de ahí hacia acá, nunca al
revés.
"""

from .contenedor import Contenedor
from .environment import Environment, Output, Stub
from .verbos import INSTRUCCIONES, Verbos
from .workspace import CABECERA, HERRAMIENTAS, Bridge, Workspace, grep, rutas, sin_acento

# Las dos formas de un REPL con contexto adentro. Lo que las distingue es dónde corre el
# código; lo que comparten es lo que el que las arma necesita: `var`, `tools`, `bridge`,
# `run` y `cerrar`. Es una unión y no un Protocol porque son dos y se llaman por nombre.
Repl = Workspace | Contenedor

__all__ = [
    "CABECERA",
    "HERRAMIENTAS",
    "INSTRUCCIONES",
    "Bridge",
    "Contenedor",
    "Environment",
    "Output",
    "Repl",
    "Stub",
    "Verbos",
    "Workspace",
    "grep",
    "rutas",
    "sin_acento",
]
