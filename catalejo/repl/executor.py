"""La célula que corre el código que el modelo acaba de proponer.

Su seam es un `Environment`, igual que el del worker va a ser un modelo. No muta
nada del Log: las variables del REPL son estado externo, como una base de datos.
"""

from __future__ import annotations

from catalejo.core import ZERO, Cell, Fail, Log, Message, Role, Status

from .environment import Environment, Output

FENCE = "```"

SOBRA = (
    "[repl] tu mensaje traía más de un bloque y solo corrió el primero. Manda uno por "
    "turno: lo que venía abajo no se ejecutó."
)


def _cercado(texto: str) -> tuple[str, str]:
    """El primer bloque cercado CERRADO, y lo que quedó del mensaje después de él.

    Devuelve "" como código en dos casos, y los dos significan "esto es una
    respuesta final, no código": que no haya cerca, y que la cerca esté abierta y
    sin cerrar. El segundo importa. Un bloque a medias, sea por unos backticks
    sueltos en la prosa o por una respuesta cortada, no puede correr como código,
    porque si no una cerca sin terminar rompe el contrato de "prosa quiere decir
    terminé".

    Es el único parser de cercas del repo: `extract_code` y `otro_bloque` salen
    los dos de acá, así que no pueden estar en desacuerdo sobre dónde termina un
    bloque.
    """
    abre = texto.find(FENCE)
    if abre < 0:
        return "", ""  # sin cerca: prosa, o sea una respuesta final
    resto = texto[abre + len(FENCE) :]
    cierra = resto.find(FENCE)
    if cierra < 0:
        return "", ""  # cerca sin cerrar: nunca correr medio bloque
    # Saca la etiqueta de lenguaje de la línea de apertura (```python), pero solo
    # si ese salto de línea está ANTES del cierre. Si no, un bloque de una línea
    # seguido de prosa se parsea al revés.
    nl = resto.find("\n")
    if 0 <= nl < cierra:
        resto = resto[nl + 1 :]
        cierra -= nl + 1
    return resto[:cierra].strip(), resto[cierra + len(FENCE) :]


def extract_code(texto: str) -> str:
    """El código que el modelo propuso: el PRIMER bloque cercado y cerrado.

    Es la única fuente de verdad sobre qué cuenta como código, así que el
    executor y el voto nunca pueden estar en desacuerdo: la FORMA de la respuesta
    decide el control.
    """
    return _cercado(texto)[0]


def otro_bloque(texto: str) -> bool:
    """Si además del que corre quedó un segundo bloque, que no corre.

    Va uno por turno, y hasta acá el resto se caía en silencio: el modelo pedía
    dos cosas, veía una sola salida y no tenía cómo saber que la otra nunca pasó.
    Es el mismo defecto que el `grep` que recortaba sin decirlo.
    """
    return bool(_cercado(_cercado(texto)[1])[0])


def render(out: Output) -> str:
    """Lo que el modelo lee el turno siguiente: lo que imprimió, y el error si falló.

    Sin techo de truncado a propósito. En v1 había uno de 2000 caracteres y
    destruía hechos: el precio del único producto de un catálogo caía 68
    caracteres pasado el corte, así que el worker contestaba "no me apareció el
    precio" con el documento en la mano. El presupuesto se cuida acotando el DATO
    en la fuente, no la SALIDA por conteo de caracteres, que corta a ciegas donde
    caiga.
    """
    partes = ["[repl] salida:"]
    if out.stdout:
        partes.append(out.stdout.rstrip("\n"))
    if out.err:
        partes.append(f"error: {out.err}")
    if not out.stdout and not out.err:
        partes.append("(el snippet no imprimió nada)")
    return "\n".join(partes)


def executor(env: Environment) -> Cell:
    """Corre el código de la última propuesta y devuelve su salida como un dicho.

    El voto sale de la forma del mensaje, que es lo único que hace falta mirar.
    Hubo bloque quiere decir que el modelo sigue trabajando, así que vota
    CONTINUE. No hubo quiere decir que ya contestó, así que vota DONE y el loop
    corta. Por eso no hace falta un supervisor aparte: la célula que parsea la
    forma es la que sabe.

    El gasto del Environment sube al Log. Correr Python no cuesta nada, pero un
    snippet que llama a otro modelo sí, y ese gasto tiene que cruzar el borde
    para que el `budget` del loop lo vea. Es la única vía: lo que pasa adentro
    del REPL no lo ve el álgebra.

    Un snippet que revienta NO es un `Fail`. Es flujo normal del REPL: el modelo
    ve el error en la salida y lo corrige en el bloque siguiente, y el turno sigue
    votando CONTINUE. `fails` es para cuando se rompe la maquinaria, o sea cuando
    el Environment mismo no responde, y ahí el voto queda en QUIET para que el
    loop corte en vez de seguir pidiéndole código a un REPL muerto.

    Corre lo que propuso el MODELO, así que mira el rol antes que el texto. Si el
    último dicho no es del modelo no hay propuesta y esta célula no opina, que es
    el caso del worker caído: no agregó nada, y el último dicho pasa a ser la
    salida anterior del REPL. Esa salida es texto del contexto no confiable, y
    correrla sería ejecutar el corpus.
    """

    async def cell(seen: Log) -> Log:
        ultimo = seen.said[-1] if seen.said else None
        if ultimo is None or ultimo.role is not Role.ASSISTANT:
            return ZERO
        code = extract_code(ultimo.text)
        if not code:
            return Log(vote=Status.DONE)
        try:
            out = await env.run(code)
        except Exception as e:
            return Log(fails=(Fail("repl", f"{type(e).__name__}: {e}"),))
        salida = render(out)
        if otro_bloque(ultimo.text):
            salida = f"{salida}\n{SOBRA}"
        return Log(
            said=(Message(Role.USER, salida),),
            vote=Status.CONTINUE,
            spent=out.spent,
            # Volvió algo: el modelo tiene de dónde sacar lo que diga después. Un
            # snippet que no imprimió nada no le enseñó nada, así que no cuenta.
            reads=1 if out.stdout else 0,
        )

    return cell
