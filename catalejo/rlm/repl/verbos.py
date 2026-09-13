"""Los verbos del plan como builtins del REPL, y el colector que los junta.

# Por qué son builtins y no una tool

En v3 el puerto del modelo es `complete(conv) -> Reply` y no hay `tools` ni
`response_format` en ninguna parte. Así que la forma barata de que el modelo edite
el plan es la que ya usa para todo lo demás: escribir Python. Sale más barato que
la alternativa obvia, que era pedirle un JSON en una segunda cerca. Ahorra un
parser, un puerto, un canal y una llamada al modelo por paso, y encima esquiva un
bug real: `extract_code` se queda con la PRIMERA cerca cerrada y no mira el
lenguaje, así que un mensaje con el JSON arriba y el Python abajo le manda el
bloque equivocado al REPL.

El vocabulario queda cerrado sin que nadie valide nada, porque lo cierra el
namespace: un verbo que no existe es un `NameError`. Es el mismo mecanismo que
`_sin_import`.

# Estos builtins tienen efecto, y ahí hay una promesa que se corrige

`environment.py` declaraba que todo builtin lee y devuelve texto, y nada más.
Estos tres apilan. La excepción es angosta y vale la pena tenerla escrita: el
efecto aterriza en un canal del Log, PROPONE y no admite, y no sale del proceso.
Quien decide qué entra es `admitir`, que corre afuera del REPL.

# No pueden levantar excepción

Una excepción adentro del `exec` se come el resto del snippet, y el resto del
snippet es la consulta del modelo. Un verbo llamado con basura apila igual y el
que lo rechaza es `admitir`, con un motivo que el modelo lee. Es la misma división
que ya rige en el REPL: un snippet que revienta no es un `Fail`, es flujo normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from catalejo.core import PlanOp

INSTRUCCIONES = """Llevás una checklist del turno. La ves en los mensajes `[plan] estado:` y
la movés desde el mismo bloque de código con el que consultas, sin cerca aparte:

- `add_step("id", "qué hay que hacer")` agrega un paso.
- `mark("id", "active")` dice en qué estás trabajando.
- `mark("id", "done")` cierra un paso, y solo los que no tienen compuerta. Los que
  dicen `se cierra con ...` los cierro yo cuando el hecho pasa; si lo intentás, te
  lo rechazo con el motivo.
- `skip("id")` descarta un paso que no aplica, y decir por qué en la respuesta.

El turno no termina hasta que no quede ningún paso abierto, así que cerrá o
descartá lo que falte EN EL ÚLTIMO BLOQUE de código, antes de escribir la
respuesta final: esa respuesta va en prosa sin backticks, así que ahí ya no podés
llamar a nada."""


def texto(x: object) -> str:
    """Lo que el modelo pasó, como cadena. Nunca revienta.

    `str()` de un objeto cuyo `__str__` falla levanta, y esto corre adentro del
    `exec` del modelo, donde levantar cuesta el resto del snippet.
    """
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


@dataclass
class Verbos:
    """Junta lo que el modelo propuso sobre el plan, para que la célula lo juzgue.

    Es la misma forma que tiene `Bridge` para `spent`: el código del modelo apila
    adentro del REPL, y el borde del executor lo sube al Log. Ahí se termina el
    parecido, porque esto no cruza al event loop y por eso sobrevive gratis la
    mudanza a un contenedor: `llm` y `rlm` necesitan un ida y vuelta DURANTE el
    `exec`, y estos tres no necesitan ninguno.

    Sin lock, por el mismo motivo que `Bridge.spent`: `Workspace.run` serializa
    las corridas con su propio lock, así que nunca hay dos `exec` apilando acá, y
    drenar pasa en el hilo del event loop cuando ya no corre ninguno.
    """

    propuestos: list[PlanOp] = field(default_factory=list)

    def tomar(self) -> tuple[PlanOp, ...]:
        """Lo propuesto desde la última vez, y deja el colector vacío.

        Drenar y no leer, porque el canal `steps` ya acumula. Si esto devolviera
        el historial, cada paso volvería a proponer todo lo de los pasos
        anteriores y `admitir` los rechazaría uno por uno como "no cambia nada".
        """
        ops = tuple(self.propuestos)
        self.propuestos.clear()
        return ops

    def add_step(self, id: object = "", intent: object = "", *, completes_when: object = "") -> None:
        self.propuestos.append(
            PlanOp(
                "add_step",
                texto(id),
                intent=texto(intent),
                completes_when=texto(completes_when),
            )
        )

    def mark(self, id: object = "", estado: object = "") -> None:
        self.propuestos.append(PlanOp("mark", texto(id), status=texto(estado)))

    def skip(self, id: object = "") -> None:
        self.propuestos.append(PlanOp("skip", texto(id)))

    @property
    def builtins(self) -> dict[str, object]:
        """Los tres nombres, para el `extra` del Workspace.

        Van por la misma puerta que el índice de productos de `agro.py`, así que
        `Workspace` no se enteró de que existe un plan.
        """
        return {"add_step": self.add_step, "mark": self.mark, "skip": self.skip}
