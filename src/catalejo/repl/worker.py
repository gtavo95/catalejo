"""El worker raíz: la célula que le pide al modelo el próximo bloque de código.

Es la inversión de un worker común. Uno normal pliega el contexto recuperado
ADENTRO del prompt; este manda solo el handle, y el modelo alcanza el bulto
escribiendo código que corre el executor.

Por eso un modelo barato puede manejar la raíz: nunca ve el payload de 4M tokens,
solo transcripciones chicas. Esa es toda la economía del RLM.
"""

from __future__ import annotations

from catalejo.core import Cell, Conversation, Fail, Log, Message, Role
from catalejo.llm import Model

from .handle import Handle, render_handle


def window(said: Conversation, keep_recent: int) -> Conversation:
    """Recorta la transcripción que llega al prompt, conservando la pregunta.

    La transcripción crece cada turno, porque cada salida del REPL se apila, y
    tener el prompt chico es el punto entero del RLM. Pero quedarse solo con la
    cola pierde la pregunta original, así que a los diez hops el modelo ya no
    sabe qué le pidieron. Esto guarda el primer mensaje y los últimos N.

    No toca el Log: solo cambia lo que ve el modelo, así que la transcripción
    entera sobrevive para auditar.

    El aviso del recorte no es cosmético. Le dice al modelo que lo que guardó en
    variables sigue vivo, que es la vía de recuperación de lo que se cortó.
    """
    if keep_recent <= 0 or len(said) <= keep_recent:
        return said
    cortados = len(said) - keep_recent - 1
    if cortados <= 0:
        return said
    aviso = Message(
        Role.USER,
        f"[…{cortados} mensajes anteriores recortados del prompt. Lo que guardaste en "
        f"variables sigue en el workspace.]",
    )
    return (said[0], aviso, *said[-keep_recent:])


def worker(model: Model, handle: Handle, *, keep_recent: int = 0) -> Cell:
    """Manda el preámbulo más la transcripción recortada, y devuelve lo que contestó.

    No vota. La terminación la decide el executor mirando la forma del mensaje, y
    tenerlo en un solo lugar es lo que evita que dos células que parsean lo mismo
    se desincronicen.

    Un modelo que revienta es un `Fail` y no una excepción, igual que un
    Environment caído. El voto queda en QUIET para que el loop corte en vez de
    seguir pidiéndole texto a un proveedor que no responde.
    """

    async def cell(seen: Log) -> Log:
        conv = (Message(Role.SYSTEM, render_handle(handle)), *window(seen.said, keep_recent))
        try:
            reply = await model.complete(conv)
        except Exception as e:
            return Log(fails=(Fail("model", f"{type(e).__name__}: {e}"),))
        return Log(said=(reply.message,), spent=reply.spent)

    return cell
