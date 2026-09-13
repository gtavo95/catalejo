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

El `extra` se reparte en dos por una sola pregunta: ¿es una función? Lo que no lo
es (el índice de productos, la tabla de la ontología) cruza UNA vez, como
argumento del proceso, y por eso tiene que ser picklable. Lo que sí lo es (`llm`,
`rlm`, `add_step`) NO cruza: se queda en el padre, y el hijo recibe un stub con
el mismo nombre que manda el pedido por el pipe, bloquea, y devuelve lo que el
padre contestó. Un builtin con estado del padre no puede correr en el hijo, y un
bound method de un objeto picklable sería peor que un error: se picklearía como
copia y las llamadas se perderían en silencio. Por eso el reparto es por
`callable` y no por lo que pickle acepte.

Ese ida y vuelta es el mismo `Bridge` con un pipe donde había un
`run_coroutine_threadsafe`. `llm` es una closure que corre la corutina en el
event loop del padre y bloquea el hilo que la llamó; acá el hilo que la llama es
el que atiende el pipe, así que `_atender` llama a la misma closure y la
corutina corre donde siempre. El hijo nunca ve el modelo, ni la red, ni las
credenciales: ve un nombre. Y `add_step` apila en el `Verbos` del padre en el
momento en que el hijo lo llama, así que el planner lo lee por `tomar()` sin
enterarse de que hubo un proceso.

Por paso cruza `code` de ida y, de vuelta, cero o más `Pedido` hasta el `Output`.
Un pedido que revienta en el padre vuelve como error con el nombre del tipo, y el
hijo lo levanta con ese mismo nombre: `Output.err` dice `ValueError: ...` igual
que en proceso, y el modelo no distingue.

`spawn` arranca un intérprete limpio y vuelve a importar el módulo principal, así
que el script que arma un Contenedor necesita el `if __name__ == "__main__":`.
Sin eso el hijo vuelve a correr al padre, que lanza otro hijo, y los dos se
quedan esperando. `agro.py`, `evals.py` y `demo.py` ya lo tienen.

# El timeout es flujo normal, no un Fail

Si el hijo no contesta a tiempo se mata, se relanza, y lo que vuelve es un
`Output.err` que enseña: qué pasó, que el workspace se reinició y las variables
guardadas se perdieron, y qué evitar. Es la misma regla que `_sin_import`: un
error que no enseña se paga en turnos. Lo mismo si el hijo muere solo. Lo único
que levanta es no poder lanzar el proceso, y eso `executor` ya lo convierte en
`Fail("repl", ...)`, porque ahí el que falló es el host y no el snippet.

El tope corre entre mensajes, no sobre el paso entero. Mientras el padre atiende
un `llm` de un minuto el hijo está bloqueado esperando, y ese minuto no es suyo;
lo que se mide es cuánto tarda el hijo en volver a hablar.

Lo que el proceso NO arregla, y ya está escrito en `environment.py`: la
inyección. Un corpus que le hable al modelo produce código bien formado que el
hijo corre bien, y la respuesta es mentira igual.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Any

from .environment import Output
from .workspace import Bridge, Workspace, herramientas

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


@dataclass(frozen=True, slots=True)
class Pedido:
    """Lo que el hijo manda cuando el snippet llama a un builtin que vive en el padre."""

    nombre: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Respuesta:
    """Lo que el padre contesta: el valor, o el error con el nombre de su tipo."""

    valor: Any = None
    error: tuple[str, str] | None = None


def _stub(conn: Connection[Any, Any], nombre: str) -> Callable[..., Any]:
    """El lado del hijo de un builtin remoto: manda el pedido, espera, devuelve o levanta.

    El error se levanta con una clase que se llama como la del padre, así que lo
    que el modelo lee en `err` es lo mismo que leería en proceso.
    """

    def llamar(*args: Any, **kwargs: Any) -> Any:
        conn.send(Pedido(nombre, args, kwargs))
        r = conn.recv()
        if not isinstance(r, Respuesta):
            raise RuntimeError(f"el padre contestó {type(r).__name__} en vez de Respuesta")
        if r.error is not None:
            tipo, motivo = r.error
            raise type(tipo, (Exception,), {})(motivo)
        return r.valor

    llamar.__name__ = nombre
    return llamar


def _servir(
    conn: Connection[Any, Any],
    payload: str,
    var: str,
    datos: dict[str, object],
    remotos: tuple[str, ...],
    dos_pasos: bool,
) -> None:
    """El lado del hijo: un Workspace y un loop que atiende snippets hasta el `None`.

    A nivel de módulo porque `spawn` arranca un intérprete limpio y lo importa
    por nombre. Llama a `correr`, no a `run`: acá no hay event loop que proteger.
    """
    stubs = {nombre: _stub(conn, nombre) for nombre in remotos}
    ws = Workspace(payload, var=var, extra={**datos, **stubs}, dos_pasos=dos_pasos)
    try:
        while True:
            code = conn.recv()
            if code is None:
                break
            conn.send(ws.correr(code))
    except EOFError:
        pass
    finally:
        conn.close()


def _repartir(
    extra: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, Callable[..., object]]]:
    """Lo que cruza una vez y lo que se queda: datos picklables, y funciones.

    Un dato que no se puede picklear es un TypeError que dice cuál, porque el
    `extra` cruza al proceso hijo una sola vez y no hay otra forma de llevarlo.
    """
    datos: dict[str, object] = {}
    remotos: dict[str, Callable[..., object]] = {}
    for nombre, valor in extra.items():
        if callable(valor):
            remotos[nombre] = valor
            continue
        try:
            pickle.dumps(valor)
        except Exception as e:
            raise TypeError(
                f"`{nombre}` no se puede picklear y el `extra` cruza al proceso hijo una "
                f"sola vez: {type(e).__name__}: {e}"
            ) from e
        datos[nombre] = valor
    return datos, remotos


class Contenedor:
    """Un `Environment` que corre el Workspace en un proceso hijo y lo mata si tarda.

    Toma lo mismo que `Workspace`, y expone lo mismo: `var`, `tools` y `bridge`,
    porque `agro.py` y `evals.py` los leen para armar el `Handle` y contar las
    lecturas delegadas. `tools` se calcula acá sin preguntarle al hijo: es una
    propiedad fija de lo que se instaló, no un hecho de la corrida.
    """

    def __init__(
        self,
        payload: str = "",
        *,
        var: str = "ctx",
        extra: Mapping[str, object] | None = None,
        note: str = "",
        bridge: Bridge | None = None,
        dos_pasos: bool = True,
        timeout: float = TIMEOUT,
    ) -> None:
        self.var = var
        self.note = note
        self.bridge = bridge
        self.dos_pasos = dos_pasos
        self.timeout = timeout
        self._payload = payload
        self._datos, self._remotos = _repartir(extra or {})
        self._lock = asyncio.Lock()
        self._ctx: SpawnContext = multiprocessing.get_context("spawn")
        self._proc: SpawnProcess | None = None
        self._conn: Connection[Any, Any] | None = None
        self._lanzar()

    @property
    def tools(self) -> str:
        base = herramientas(self.dos_pasos)
        return f"{base}\n\n{self.note}" if self.note else base

    @property
    def vivo(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _lanzar(self) -> None:
        padre, hijo = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_servir,
            args=(hijo, self._payload, self.var, self._datos, tuple(self._remotos), self.dos_pasos),
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
        es estado compartido. El hilo es para que la espera no congele el loop, y
        porque ahí adentro se atienden los pedidos del hijo con `bridge.wait`,
        que bloquea al hilo que lo llama. El `Bridge` se maneja igual que en
        `Workspace`: se le da el loop antes de cruzar y se mide lo que gastó.
        """
        async with self._lock:
            if self.bridge is None:
                return await asyncio.to_thread(self._pedir, code)
            self.bridge.loop = asyncio.get_running_loop()
            antes = self.bridge.spent
            out = await asyncio.to_thread(self._pedir, code)
            return replace(out, spent=self.bridge.spent - antes)

    def _pedir(self, code: str) -> Output:
        if self._conn is None or not self.vivo:
            self._matar()
            self._lanzar()
        assert self._conn is not None
        conn = self._conn
        try:
            conn.send(code)
            while True:
                if not conn.poll(self.timeout):
                    self._matar()
                    self._lanzar()
                    return Output(err=aviso_timeout(self.timeout, self.var))
                msg = conn.recv()
                if isinstance(msg, Output):
                    return msg
                conn.send(self._atender(_pedido(msg)))
        except (EOFError, OSError) as e:
            self._matar()
            self._lanzar()
            return Output(err=aviso_muerto(f"{type(e).__name__}: {e}", self.var))

    def _atender(self, p: Pedido) -> Respuesta:
        """Un pedido del hijo, resuelto en el padre con la función de verdad.

        Lo que la función devuelva tiene que volver por el pipe, así que se
        picklea acá y no en el `send`: si no se puede, es un error del builtin y
        el modelo lo ve como tal, en vez de morir el pipe.
        """
        fn = self._remotos.get(p.nombre)
        if fn is None:
            return Respuesta(error=("NameError", f"el padre no tiene `{p.nombre}`"))
        try:
            valor = fn(*p.args, **p.kwargs)
            pickle.dumps(valor)
        except Exception as e:
            return Respuesta(error=(type(e).__name__, str(e)))
        return Respuesta(valor=valor)

    def cerrar(self) -> None:
        """Le pide al hijo que termine y, si no lo hace a tiempo, lo mata.

        `daemon=True` ya cubre el caso en que nadie llame a esto: el hijo muere
        con el padre. Esto es para no dejar procesos colgando entre tests, ni
        entre casos de un eval.
        """
        if self._conn is not None:
            try:
                self._conn.send(None)
            except OSError:
                pass
        if self._proc is not None:
            self._proc.join(timeout=CORTESIA)
        self._matar()


def _pedido(x: object) -> Pedido:
    if not isinstance(x, Pedido):
        raise TypeError(f"el hijo mandó {type(x).__name__} en vez de Output o Pedido")
    return x
