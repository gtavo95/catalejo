"""El seam del REPL: un workspace con estado que el modelo maneja con código.

El contexto grande vive acá como una variable, y NUNCA llega al prompt. Al
prompt solo llega el handle: el nombre de la variable, su tamaño y un esquema de
una línea. Esa es toda la economía del asunto, y por eso un contexto de 4M tokens
no paga 4M tokens por turno.

# El invariante: el Environment es una ventana de solo lectura

Todo builtin que se le dé al modelo puede leer y devolver texto, y nada más. Sin
filesystem, sin escrituras de red, sin más efectos que leer.

El código que el modelo escribe procesa texto no confiable, el contexto cargado y
lo que recupere de una búsqueda, así que este invariante es la frontera de
seguridad: una inyección que secuestre al modelo, en el peor caso, lee lo que el
modelo ya podía leer. Cualquier capacidad futura con efectos va detrás de
aprobación humana explícita, nunca como un builtin más.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Output:
    """El resultado de una corrida: lo que imprimió, el error si reventó, y qué costó.

    Un snippet que falla no es una excepción del host, es flujo normal del REPL
    que el modelo tiene que ver para corregirlo. Por eso el error viaja acá
    adentro y no como una excepción.

    `spent` es lo que gastó ESTA corrida. Casi siempre es cero, porque correr
    Python no cuesta tokens. Deja de serlo cuando el código del modelo llama a
    otro modelo: sin este campo, el gasto del árbol recursivo es invisible para
    el `budget` del padre y la recursión se financia sola.
    """

    stdout: str = ""
    err: str = ""
    spent: int = 0


class Environment(Protocol):
    """Corre código contra el workspace vivo y devuelve lo que imprimió.

    El workspace persiste entre llamadas, así que las variables quedan y el
    modelo puede acumular estado en código a lo largo de varios turnos. Eso es lo
    que le permite guardar un hallazgo en vez de depender de releer la
    transcripción, que se va recortando.
    """

    async def run(self, code: str) -> Output: ...


class Stub:
    """Un Environment de mentira: corre el árbol entero sin ejecutar nada real.

    Sirve para probar el loop antes de conectar un sandbox, y para escribir tests
    del executor sin depender de qué hace Python con un snippet.
    """

    def __init__(self, guion: Callable[[str], Output] | None = None) -> None:
        self._guion = guion
        self.corrido: list[str] = []

    async def run(self, code: str) -> Output:
        self.corrido.append(code)
        if self._guion is None:
            return Output(stdout=f"eco: {code}")
        return self._guion(code)
