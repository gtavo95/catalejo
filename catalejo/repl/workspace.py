"""El workspace de Python real: `exec` sobre un namespace que persiste.

Es la razón por la que el host es Python. El modelo escribe Python, así que
correrlo es `exec` en un diccionario, y los builtins recursivos que vienen
después (`llm`, `rlm`, `fanout`) van a ser funciones normales en vez de un puente
entre procesos.

# Hasta dónde llega el aislamiento

Esto NO es un sandbox. El namespace viene con los builtins recortados, sin
`import`, sin `open`, sin `eval`: alcanza para que el modelo no rompa nada por
accidente, y no alcanza contra código que se lo proponga, porque desde cualquier
objeto se llega a `__class__.__bases__` y de ahí a media biblioteca estándar.

La frontera de verdad es el contenedor, y va después. Mientras el contexto lo
cargues vos y el modelo sea el único que escribe código, esto sirve. En el
momento en que el contexto venga de afuera, el aislamiento tiene que ser un
proceso aparte.

Tampoco hay timeout: no se puede interrumpir un `exec` en curso desde otro hilo.
Un `while True` del modelo cuelga el workspace hasta que se muera el proceso. Es
la otra cosa que arregla el contenedor.
"""

from __future__ import annotations

import asyncio
import builtins
import re
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from .environment import Output

T = TypeVar("T")

# Lo que el modelo puede usar. Nada que escriba, abra, importe o evalúe: la
# ventana es de solo lectura.
_SEGUROS = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    # Para que el modelo pueda escribir un try/except sobre su propio código.
    # Un nombre que falta acá no desactiva el except: lo convierte en un
    # `NameError: name 'ZeroDivisionError' is not defined` que tapa el error de
    # verdad con uno que no tiene nada que ver.
    "AttributeError",
    "Exception",
    "IndexError",
    "KeyError",
    "NameError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)

MAX_HITS = 50

# Lo que el preámbulo le dice al modelo que tiene a mano. Vive acá, al lado de
# _SEGUROS, porque solo el Environment sabe qué ofrece: si mañana el sandbox es un
# contenedor con `re` importable, cambia esta nota junto con la lista de arriba.
# Sin ella el modelo escribe `import re` y se come un turno entero descubriendo
# que no anda.
HERRAMIENTAS = (
    "Es Python real con los builtins recortados: no hay `import`, `open` ni `eval`. "
    "Tienes `grep(texto, patron)` para expresiones regulares: devuelve las líneas que "
    "casan, numeradas, con el total en la primera línea. Muestra hasta 50; si hay más, "
    "sube el tope con `grep(texto, patron, max_hits=500)` o afina el patrón. El resto de "
    "Python funciona normal: rebanar, `len`, comprensiones, `sorted`."
)


def _sin_import(nombre: str, *_: object, **__: object) -> object:
    """El `import` que no hay, explicado.

    Sin esto Python levanta `ImportError: __import__ not found`, que no le dice al
    modelo qué hacer en su lugar. Dos corridas contra Gemini se perdieron ahí: pidió
    `re` una vez y `collections` la otra, y las dos veces el error no lo llevó a
    ninguna parte. Un error que no enseña se paga en turnos, o peor: una de las dos
    veces el modelo dejó de intentar y contestó de memoria.
    """
    raise ImportError(
        f"no hay `import` en este REPL, así que `{nombre}` no está disponible. "
        f"Para expresiones regulares usa grep(texto, patron). Para contar, un dict: "
        f"`c = {{}}; c[k] = c.get(k, 0) + 1`. El resto de Python funciona normal."
    )


def grep(texto: str, patron: str, max_hits: int = MAX_HITS) -> str:
    """Las líneas que casan con el patrón, numeradas, con el total adelante.

    Existe porque `re` no es importable. Un contexto grande se recorre con esto,
    no imprimiéndolo: el punto del RLM es que el bulto nunca entre al prompt.

    El total va SIEMPRE y va PRIMERO, y no es cosmética. Un tope que no se anuncia
    convierte "contá cuántos hay" en una respuesta falsa que parece verdadera: en
    la primera corrida contra Gemini el modelo contó las 50 líneas que le
    devolvimos y contestó 50, sobre un corpus con 1823. Había pedido el dato
    correcto y le mentimos. Un tope silencioso es peor que no tener tope, porque
    el error no se ve. Este dice cuántas hay, cuántas muestra y cómo pedir el resto.

    Cero coincidencias también se dice con todas las letras. Devolver "" es
    indistinguible de un snippet que no imprimió nada.
    """
    rx = re.compile(patron)
    hits = [f"{i}: {linea}" for i, linea in enumerate(texto.splitlines(), 1) if rx.search(linea)]
    casan = "casa" if len(hits) == 1 else "casan"
    linea = "línea" if len(hits) == 1 else "líneas"
    cabecera = f"{len(hits)} {linea} {casan} con {patron!r}."
    if len(hits) > max_hits:
        cabecera = (
            f"{len(hits)} líneas casan con {patron!r}; estas son las primeras {max_hits}. "
            f"Para el resto subí max_hits o afina el patrón."
        )
        hits = hits[:max_hits]
    return "\n".join([cabecera, *hits])


@dataclass
class Bridge:
    """El paso del hilo del `exec` al event loop, y el medidor de lo que cruza.

    El modelo escribe código síncrono. Llamar a otro modelo es una corutina. El
    `exec` corre en un hilo aparte, así que ahí adentro no se puede hacer `await`
    de nada: `run_coroutine_threadsafe` es el puente, y bloquea SOLO a ese hilo
    mientras el event loop sigue atendiendo a todos los demás.

    Cuenta lo gastado porque este es el único lugar por donde pasa. Todo lo que
    consume el árbol recursivo nace de una llamada que cruzó por acá, así que
    medirlo acá es medirlo todo, y `Output.spent` lo sube al Log del padre.

    Las mutaciones de `spent` pasan siempre en el hilo del event loop, adentro de
    la corutina, así que no hace falta lock aunque haya varios `exec` en paralelo.
    """

    budget: int = 0
    spent: int = 0
    calls: int = 0
    loop: asyncio.AbstractEventLoop | None = None

    @property
    def left(self) -> int:
        return max(0, self.budget - self.spent)

    def wait(self, coro: Coroutine[Any, Any, T]) -> T:
        """Corre la corutina en el event loop y bloquea este hilo hasta que vuelva."""
        if self.loop is None:
            coro.close()
            raise RuntimeError("no hay event loop: esto corre desde el código del modelo")
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


class Workspace:
    """Un REPL de Python con el contexto grande adentro, en `var`.

    Las variables persisten entre corridas, así que el modelo puede guardar un
    hallazgo en vez de depender de releer la transcripción, que se va recortando.
    """

    def __init__(
        self,
        payload: str = "",
        *,
        var: str = "ctx",
        extra: Mapping[str, object] | None = None,
        note: str = "",
        bridge: Bridge | None = None,
    ) -> None:
        self.var = var
        self.note = note
        self.bridge = bridge
        self._salida: list[str] = []
        self._lock = asyncio.Lock()
        self._globals: dict[str, object] = {
            "__builtins__": {
                **{nombre: getattr(builtins, nombre) for nombre in _SEGUROS},
                # `print` no escribe en sys.stdout: escribe en un buffer nuestro.
                # Redirigir sys.stdout es global al proceso y se pisa entre
                # workspaces corriendo en paralelo; esto es de cada uno.
                "print": self._print,
                # No habilita nada: reemplaza un error críptico por uno que dice
                # qué usar en su lugar.
                "__import__": _sin_import,
                # Definir una clase pide esto en los builtins y `__name__` en el
                # namespace. Sin las dos cosas, `class Nota: ...` muere con
                # `NameError: __build_class__ not found`, que es el mismo error
                # que no enseña nada del `import`. Habilitarlo no ensancha la
                # ventana: una clase nueva no alcanza nada que el modelo no
                # alcanzara ya, y la ventana la define qué builtins hay, no si se
                # puede declarar un tipo.
                "__build_class__": builtins.__build_class__,
            },
            "__name__": "repl",
            "grep": grep,
            **(extra or {}),
            # Va último para que nadie pise el payload por accidente.
            var: payload,
        }

    @property
    def tools(self) -> str:
        """La nota del preámbulo: qué ofrece ESTE workspace.

        Vive acá y no en el `Handle` porque solo el Environment sabe qué tiene
        adentro. Un builtin que se agrega sin actualizar la nota es un builtin
        que el modelo nunca va a usar, porque no sabe que existe.
        """
        return f"{HERRAMIENTAS}\n\n{self.note}" if self.note else HERRAMIENTAS

    def _print(self, *args: object, sep: str = " ", end: str = "\n") -> None:
        self._salida.append(sep.join(str(a) for a in args) + end)

    async def run(self, code: str) -> Output:
        """Corre el snippet y devuelve lo que imprimió.

        Serializa con un lock porque el namespace es estado compartido: un
        `fanout` que le pase el mismo workspace a varias células no puede
        pisarse. Y va a un hilo para que un grep sobre 4M de texto no congele el
        resto de las corutinas mientras esperan al modelo.
        """
        async with self._lock:
            if self.bridge is None:
                return await asyncio.to_thread(self._run_sync, code)
            # El hilo del exec no puede alcanzar el loop por su cuenta, así que se
            # lo dejamos acá antes de cruzar.
            self.bridge.loop = asyncio.get_running_loop()
            antes = self.bridge.spent
            out = await asyncio.to_thread(self._run_sync, code)
            return replace(out, spent=self.bridge.spent - antes)

    def _run_sync(self, code: str) -> Output:
        self._salida.clear()
        err = ""
        try:
            exec(code, self._globals)
        except Exception as e:  # el modelo tiene que VER el error para corregirlo
            err = f"{type(e).__name__}: {e}"
        return Output(stdout="".join(self._salida), err=err)
