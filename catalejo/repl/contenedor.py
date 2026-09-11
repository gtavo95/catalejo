"""El Workspace en un proceso aparte: lo que corre va al hijo, lo que espera y mata se queda acá.

`Workspace.run` hace dos trabajos fundidos. Uno es correr código: el namespace,
el `exec`, capturar el `print`, convertir la excepción en `err`. El otro es dónde
corre: el hilo, el lock, medir. Acá se separan por la costura que ya existía:
`Environment` es un Protocol, así que esto es un hermano de `Workspace` y nada
arriba de `run` se entera.

# Por qué un proceso y no un hilo con timeout

Un hilo no se mata. Un `while True` del modelo, o un `(a+)+$` sobre una línea
larga, se quedan hasta que muera el proceso entero (medido: más de 90 s, hubo que
matarlo a mano). `signal.alarm` solo llega al hilo principal y `re` no se
interrumpe. Lo único que se puede matar es un proceso, y eso es esto.

El mismo proceso borra la otra causa: el intérprete deja de ser compartido, así
que `json.__builtins__` llega al `open` DEL HIJO, que no tiene el cliente MCP, ni
el modelo, ni las credenciales. Lo que protege es la memoria del padre y la
posibilidad de matar. No le quita al hijo sockets ni disco; eso es otro escalón
(`resource`, o correr el hijo bajo un sandbox de verdad) y no va acá.

# Qué cruza el pipe, y qué no

El payload y el `extra` cruzan UNA vez, como argumentos del proceso, y por eso
tienen que ser picklables: datos, no funciones. Un builtin con estado del padre
(una closure, un bound method) no puede correr en el hijo, y un bound method de
un objeto picklable es peor que un error: se picklea como copia y las llamadas se
pierden en silencio. Por eso `extra` rechaza cualquier callable, con el motivo.

Por paso cruza `code` de ida y `(Output, movidas)` de vuelta. `Output` no cambia
de forma: las movidas del plan viajan al lado y no adentro, porque `Output` es lo
que ven las células y las movidas las lee el planner por `Verbos.tomar()`. Si se
pasa `verbos=`, el hijo arma su propio `Verbos`, instala sus builtins, y después
de cada corrida manda lo que apiló; el padre lo vuelca en el `Verbos` que le
dieron, así que `planner(verbos, ...)` sigue igual.

`spawn` arranca un intérprete limpio y vuelve a importar el módulo principal, así
que el script que arma un Contenedor necesita el `if __name__ == "__main__":`.
Sin eso el hijo vuelve a correr al padre, que lanza otro hijo, y los dos se
quedan esperando. `agro.py`, `evals.py` y `demo.py` ya lo tienen.

`llm` y `rlm` NO cruzan. Son closures sobre `bridge.loop`, que es el event loop
del padre, y necesitan un ida y vuelta DURANTE el `exec`. Esta es la versión
unidireccional: el hijo no los tiene, y `recurse` sigue armando `Workspace` en
proceso. La versión que sigue es el mismo Bridge con un pipe donde había un
`run_coroutine_threadsafe`.

# El timeout es flujo normal, no un Fail

Si el hijo no contesta a tiempo se mata, se relanza, y lo que vuelve es un
`Output.err` que enseña: qué pasó, que el workspace se reinició y las variables
guardadas se perdieron, y qué evitar. Es la misma regla que `_sin_import`: un
error que no enseña se paga en turnos. Lo mismo si el hijo muere solo. Lo único
que levanta es no poder lanzar el proceso, y eso `executor` ya lo convierte en
`Fail("repl", ...)`, porque ahí el que falló es el host y no el snippet.

Lo que el proceso NO arregla, y ya está escrito en `environment.py`: la
inyección. Un corpus que le hable al modelo produce código bien formado que el
hijo corre bien, y la respuesta es mentira igual.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import pickle
from collections.abc import Mapping
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Any

from catalejo.core import PlanOp

from .environment import Output
from .verbos import Verbos
from .workspace import HERRAMIENTAS, Workspace

TIMEOUT = 30.0

# Cuánto se le espera al hijo para que cierre solo antes de matarlo.
CORTESIA = 2.0


def aviso_timeout(segundos: float, var: str) -> str:
    return (
        f"el snippet no terminó en {segundos:g} s y se mató. El workspace se reinició: "
        f"`{var}` está de nuevo, pero las variables que habías guardado se perdieron. "
        f"Evitá bucles sin condición de salida y patrones con repeticiones anidadas como "
        f"`(a+)+`; para buscar usá grep(texto, patron) con un patrón simple."
    )


def aviso_muerto(motivo: str, var: str) -> str:
    return (
        f"el proceso del REPL murió mientras corría el snippet ({motivo}). Se reinició: "
        f"`{var}` está de nuevo, pero las variables que habías guardado se perdieron."
    )


def _servir(
    conn: Connection[Any, Any],
    payload: str,
    var: str,
    extra: dict[str, object],
    con_verbos: bool,
) -> None:
    """El lado del hijo: un Workspace y un loop que atiende pedidos hasta el `None`.

    A nivel de módulo porque `spawn` arranca un intérprete limpio y lo importa
    por nombre. Llama a `correr`, no a `run`: acá no hay event loop que proteger.
    """
    verbos = Verbos() if con_verbos else None
    ws = Workspace(payload, var=var, extra={**extra, **(verbos.builtins if verbos else {})})
    try:
        while True:
            code = conn.recv()
            if code is None:
                break
            out = ws.correr(code)
            conn.send((out, verbos.tomar() if verbos else ()))
    except EOFError:
        pass
    finally:
        conn.close()


def _picklable(extra: Mapping[str, object]) -> dict[str, object]:
    """El `extra` que puede cruzar, o un TypeError que dice cuál no y qué hacer."""
    for nombre, valor in extra.items():
        if callable(valor):
            raise TypeError(
                f"`{nombre}` es una función y no puede cruzar al proceso hijo: un builtin "
                f"con estado del padre no corre ahí. Para las movidas del plan pasá "
                f"`verbos=`; para `llm` y `rlm` usá `Workspace` en proceso."
            )
        try:
            pickle.dumps(valor)
        except Exception as e:
            raise TypeError(
                f"`{nombre}` no se puede picklear y el `extra` cruza al proceso hijo una "
                f"sola vez: {type(e).__name__}: {e}"
            ) from e
    return dict(extra)


class Contenedor:
    """Un `Environment` que corre el Workspace en un proceso hijo y lo mata si tarda.

    Expone `var` y `tools` igual que `Workspace`, porque `agro.py` y `evals.py`
    los leen para armar el `Handle`. `tools` se calcula acá sin preguntarle al
    hijo: es una propiedad fija de lo que se instaló, no un hecho de la corrida.
    """

    def __init__(
        self,
        payload: str = "",
        *,
        var: str = "ctx",
        extra: Mapping[str, object] | None = None,
        note: str = "",
        verbos: Verbos | None = None,
        timeout: float = TIMEOUT,
    ) -> None:
        self.var = var
        self.note = note
        self.verbos = verbos
        self.timeout = timeout
        self._payload = payload
        self._extra = _picklable(extra or {})
        self._lock = asyncio.Lock()
        self._ctx: SpawnContext = multiprocessing.get_context("spawn")
        self._proc: SpawnProcess | None = None
        self._conn: Connection[Any, Any] | None = None
        self._lanzar()

    @property
    def tools(self) -> str:
        return f"{HERRAMIENTAS}\n\n{self.note}" if self.note else HERRAMIENTAS

    @property
    def vivo(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _lanzar(self) -> None:
        padre, hijo = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_servir,
            args=(hijo, self._payload, self.var, self._extra, self.verbos is not None),
            daemon=True,
        )
        proc.start()
        hijo.close()
        self._proc, self._conn = proc, padre

    def _matar(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join()
        if self._conn is not None:
            self._conn.close()
        self._proc = self._conn = None

    async def run(self, code: str) -> Output:
        """Manda el snippet al hijo y espera la respuesta, con tope.

        El lock serializa por lo mismo que en `Workspace`: el namespace del hijo
        es estado compartido. El hilo es para que la espera no congele el loop.
        """
        async with self._lock:
            return await asyncio.to_thread(self._pedir, code)

    def _pedir(self, code: str) -> Output:
        if self._conn is None or not self.vivo:
            self._matar()
            self._lanzar()
        assert self._conn is not None
        try:
            self._conn.send(code)
            if not self._conn.poll(self.timeout):
                self._matar()
                self._lanzar()
                return Output(err=aviso_timeout(self.timeout, self.var))
            out, movidas = self._conn.recv()
        except (EOFError, OSError) as e:
            self._matar()
            self._lanzar()
            return Output(err=aviso_muerto(f"{type(e).__name__}: {e}", self.var))
        if self.verbos is not None:
            self.verbos.propuestos.extend(_movidas(movidas))
        return _output(out)

    def cerrar(self) -> None:
        """Le pide al hijo que termine y, si no lo hace a tiempo, lo mata.

        `daemon=True` ya cubre el caso en que nadie llame a esto: el hijo muere
        con el padre. Esto es para no dejar procesos colgando entre tests.
        """
        if self._conn is not None:
            try:
                self._conn.send(None)
            except OSError:
                pass
        if self._proc is not None:
            self._proc.join(timeout=CORTESIA)
        self._matar()


def _output(x: object) -> Output:
    if not isinstance(x, Output):
        raise TypeError(f"el hijo devolvió {type(x).__name__} en vez de Output")
    return x


def _movidas(x: object) -> tuple[PlanOp, ...]:
    if not isinstance(x, tuple) or not all(isinstance(op, PlanOp) for op in x):
        raise TypeError("el hijo devolvió movidas que no son PlanOp")
    return x
