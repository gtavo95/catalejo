"""El agente RLM: un REPL, las células que lo manejan, y las dos formas de juntarlos.

`repl/` es dónde corre el código y no sabe qué es una célula. `celulas/` son las
células de un paso y no saben dónde corre el código. `drive` compone las células
en un turno; `recurse` arma un REPL que adentro tiene agentes. Esos dos son los
únicos que necesitan las dos mitades, y son la R de RLM.
"""

from .celulas import (
    Handle,
    Registro,
    citada,
    citadas,
    executor,
    extract_code,
    grounded,
    inventada,
    planner,
    render,
    render_handle,
    render_plan,
    window,
    worker,
)
from .drive import drive
from .recurse import Metered, answer, note, recurse
from .repl import (
    HERRAMIENTAS,
    HERRAMIENTAS_UN_PASO,
    INSTRUCCIONES,
    Bridge,
    Contenedor,
    Environment,
    Output,
    Repl,
    Stub,
    Verbos,
    Workspace,
    grep,
    read,
    rutas,
)

__all__ = [
    "HERRAMIENTAS",
    "HERRAMIENTAS_UN_PASO",
    "INSTRUCCIONES",
    "Bridge",
    "Contenedor",
    "Environment",
    "Handle",
    "Metered",
    "Output",
    "Registro",
    "Repl",
    "Stub",
    "Verbos",
    "Workspace",
    "answer",
    "citada",
    "citadas",
    "drive",
    "executor",
    "extract_code",
    "grep",
    "read",
    "grounded",
    "inventada",
    "note",
    "planner",
    "recurse",
    "render",
    "render_handle",
    "render_plan",
    "rutas",
    "window",
    "worker",
]
