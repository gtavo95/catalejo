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

__all__ = [
    "CABECERA",
    "HERRAMIENTAS",
    "INSTRUCCIONES",
    "Bridge",
    "Contenedor",
    "Environment",
    "Output",
    "Stub",
    "Verbos",
    "Workspace",
    "grep",
    "rutas",
    "sin_acento",
]
