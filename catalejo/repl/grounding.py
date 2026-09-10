"""La célula que no deja pasar una respuesta que no consultó nada.

# Por qué existe

La terminación de este agente es sintáctica: prosa quiere decir terminé. Alcanza
para decidir el control y no alcanza para nada más, porque un modelo que contesta
sin haber mirado el contexto termina igual de bien que uno que lo leyó entero.

No es hipotético. Apuntando el demo al código de v1, el modelo escribió
`import re`, se comió un `ImportError` que no le decía qué hacer, y en vez de
corregir contestó el índice de los 80 archivos de memoria. Inventó nombres que no
existen en el repo. Gastó 36 mil tokens, votó DONE y no reportó un solo `fail`:
el sistema dijo que había salido bien.

# Cómo veta sin poder vetar

Una célula devuelve lo que agrega y no puede borrar lo que otra dijo, así que no
hay forma de cancelar el DONE del executor. Y no hace falta: `vote` se junta por
máximo y CONTINUE es mayor que DONE. Alcanza con opinar más fuerte. Eso es
exactamente para lo que estaba el orden en `Status`, y es la primera célula que
lo usa de verdad.

# Qué mira

`seen.reads`, que el executor sube cada vez que el REPL devuelve algo. Es un piso
y no una prueba: dice que el modelo tuvo algo enfrente, no que lo que afirma
salga de ahí. Atrapa el caso que pasó de verdad, que es no haber mirado nada, y
no atrapa una cita inventada sobre un texto que sí leyó. Para eso haría falta un
juez, y un juez es otra célula.
"""

from __future__ import annotations

from catalejo.core import ZERO, Cell, Fail, Log, Message, Role, Status

from .executor import extract_code

AVISO = (
    "[grounding] Todavía no ejecutaste nada que devolviera algo, así que esa respuesta "
    "no sale de `{var}` sino de tu memoria. Consulta el contexto con un bloque de código "
    "y contesta con lo que devuelva. Si un bloque anterior falló, lee el error: dice qué "
    "usar en su lugar."
)

SIN_FUNDAMENTO = Fail("grounding", "contestó sin haber consultado el contexto")


def grounded(var: str = "ctx") -> Cell:
    """Vota CONTINUE cuando el modelo contesta sin haber consultado nada.

    Va después del executor y comparte con él la única fuente de verdad sobre qué
    cuenta como código, que es `extract_code`. La regla del diseño es un solo
    parser, no un solo llamador: dos células que parsean lo mismo con su propio
    código se desincronizan, dos que llaman a la misma función no.

    Avisa una vez. Si el modelo vuelve a contestar de memoria después del aviso,
    lo deja terminar y anota el `Fail`: la respuesta sale, pero sale etiquetada.
    Insistir hasta el tope de pasos serían doce llamadas para llegar a la misma
    conclusión, y el modelo que ignoró el aviso una vez lo va a ignorar diez.
    """
    aviso = Message(Role.USER, AVISO.format(var=var))

    async def cell(seen: Log) -> Log:
        if not seen.said:
            return ZERO
        ultimo = seen.said[-1]
        if ultimo.role is not Role.ASSISTANT or extract_code(ultimo.text):
            return ZERO
        if seen.reads > 0:
            return ZERO
        if aviso in seen.said:
            return Log(fails=(SIN_FUNDAMENTO,))
        return Log(said=(aviso,), vote=Status.CONTINUE)

    return cell
