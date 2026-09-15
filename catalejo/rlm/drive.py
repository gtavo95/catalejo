"""El cableado del agente de REPL, con nombre.

`then(worker, executor, grounded)` adentro de un `loop` estaba escrito a mano en
`agro.py` y en `evals.py`, o sea que el eval medía una copia del agente y no el
agente. Nombrarlo es todo lo que hace este módulo, y no le agrega nada al álgebra:
es una función de células a célula, del mismo tipo que `then` o `loop`.

Eso es la mitad del planner que no cuesta un canal. La checklist es un tipo de
dato con su intérprete y vive en `core.plan`; el cableado es un patrón y vive acá.
La línea entre las dos mitades es la respuesta a si un plan es un primitivo: la
parte que compone, no.
"""

from __future__ import annotations

from collections.abc import Sequence

from catalejo.core import Cell, loop, then
from catalejo.llm import Model

from .celulas import Handle, executor, grounded, worker
from .repl import Environment

MAX_STEPS = 12


def drive(
    model: Model,
    handle: Handle,
    env: Environment,
    *,
    keep_recent: int = 6,
    max_steps: int = MAX_STEPS,
    budget: int | None = None,
    extras: Sequence[Cell] = (),
) -> Cell:
    """Un paso es proponer, ejecutar y revisar; el turno es repetir eso hasta el tope.

    `model` es el que propone el código de cada paso. Ve el preámbulo y la
    transcripción recortada, nunca el payload, así que puede ser uno barato.

    `handle` es lo que el modelo sabe del contexto: el nombre de la variable, el
    tamaño, el esquema y las herramientas. Si el `env` viene de `recurse`, sus
    `tools` tienen que ser los del workspace, o el modelo no se entera de que
    tiene `llm`.

    `env` es donde corre el código que el modelo propone: un `Workspace` pelado o
    uno de `recurse`. Su `var` y el de `handle` tienen que coincidir, porque el
    aviso de `grounded` y el preámbulo nombran el mismo.

    `keep_recent` es cuántos mensajes de la cola ve el modelo además del primero,
    que se conserva siempre porque es la pregunta. Lo aplica `window`; cero es sin
    recorte.

    `max_steps` y `budget` son los dos topes del `loop`: pasos y tokens del turno
    entero. `budget=None` deja solo el de pasos. El `budget` de `recurse` es otro,
    el del árbol de delegaciones, y los dos conviven.

    El orden no es negociable en las tres primeras. `executor` va después del
    worker porque corre lo que el worker acaba de proponer, y `grounded` va después
    del executor porque mira `reads`, que el executor acaba de subir, y porque
    comparte con él `extract_code`.

    `extras` van al final por lo mismo: una célula que juzgue el paso necesita ver
    el paso entero, y en `then` cada célula ve lo que llegó más lo que agregaron
    las anteriores. Ahí entra el `planner`, que es `then(avanzar, exigir)`, o el
    paso de la venta, que es un `then` más largo sobre dos canales.
    """
    paso = then(
        worker(model, handle, keep_recent=keep_recent),
        executor(env),
        grounded(handle.var),
        *extras,
    )
    return loop(paso, max_steps=max_steps, budget=budget)
