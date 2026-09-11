"""La cita contra el conjunto cerrado: las rutas que la respuesta declara tienen que existir.

# El agujero que `grounded` no tapa

`grounded` mira `reads`: si el REPL devolvió algo alguna vez. Es un piso y no una
prueba, y el docstring lo dice: atrapa "contestó sin mirar" y no atrapa una cita
inventada sobre un texto que sí leyó. Pasó de verdad: preguntando por zompopo, una
plaga que el catálogo no cubre, el modelo consultó el catálogo, no encontró nada,
y cerró con tres `FUENTE: productos/...md` que no existen (beaveria-90,
isaria-forte, metarhizium-50). `grounded` lo dejó pasar, porque había leído.

Eso NO necesita un juez. Una cita es una afirmación sobre un conjunto conocido, y
eso se verifica con código, exacto y gratis. `inventada` ya lo hacía en
`evals.py`, o sea después de la corrida y sin frenar nada. Acá hace lo mismo
durante el turno.

# De dónde sale el conjunto

De las cabeceras `=== ruta ===` del texto que se le dio al modelo, con `rutas`,
y se calcula en el padre ANTES de armar el Environment. Tiene que ser así porque
la célula corre en el padre y el corpus vive en el workspace, que con el
`Contenedor` está en otro proceso. Y sale del texto y no de un segundo recorrido
del disco: una cita a un archivo que existe en el bundle pero no en el corpus es
tan inventada como una a un archivo que no existe en ninguna parte.

# Un solo parser

`citadas`, `inventada` y `rutas` viven acá y `evals.py` las importa. Es la regla
que ya rige entre `grounded` y `executor` con `extract_code`: dos células que
parsean lo mismo con su propio código se desincronizan, dos que llaman a la
misma función no.

Compara sin acentos ni mayúsculas. Los archivos del bundle son slugs en
minúscula, pero el modelo escribe `Viventem.md` a veces, y un error de mayúscula
no es una ficha inventada. Cambia el scorer del eval en la misma dirección.

# Misma forma que `grounded`, otro hecho

Avisa una vez votando CONTINUE, con las rutas que no existen y de dónde salen las
de verdad. Si el modelo insiste, lo deja salir con un `Fail`: la respuesta sale,
pero etiquetada. El aviso lista rutas, así que no es un mensaje fijo y "ya
avisé" se detecta por el prefijo `[cita]`, no por igualdad.

Lo que NO mira: la prosa. Una respuesta que recomienda un producto para una
plaga que la ontología no tiene, sin citar nada, pasa. Eso no es un chequeo de
cadenas; lo chequeable es darle la ontología como dato, y va aparte.
"""

from __future__ import annotations

from collections.abc import Collection

from catalejo.core import ZERO, Cell, Fail, Log, Message, Role, Status

from .executor import extract_code
from .workspace import CABECERA, sin_acento

PREFIJO = "[cita]"

AVISO = (
    "{prefijo} Citaste {rutas}, y {falta} en `{var}`. Las rutas de verdad son las de las "
    "líneas `=== ruta ===`: buscá la ficha con grep({var}, patron) y citá solo los "
    "archivos de los que leíste el dato, o cerrá con `FUENTE: ninguna`."
)


def rutas(texto: str) -> frozenset[str]:
    """Las rutas que EXISTEN en el texto: las cabeceras `=== ruta ===`."""
    return frozenset(m.group(1) for m in map(CABECERA.match, texto.splitlines()) if m)


def citadas(texto: str) -> tuple[str, ...]:
    """Todas las rutas que la respuesta declara como fuente.

    Todas y no la última, porque el CONTRATO de `agro.py` pide una línea
    `FUENTE:` por cada archivo del que salió un dato. Mirando solo la última, una
    ruta fabricada en la primera pasaba sin que nadie la viera.
    """
    return tuple(
        linea.split(":", 1)[1].strip().strip("`").lstrip("/")
        for linea in texto.splitlines()
        if linea.strip().upper().startswith("FUENTE:")
    )


def _llave(ruta: str) -> str:
    return sin_acento(ruta).lower()


def inventada(texto: str, reales: Collection[str]) -> tuple[str, ...]:
    """Las rutas citadas que no existen, si las hay. `ninguna` no es una ruta."""
    llaves = {_llave(r) for r in reales}
    return tuple(
        r
        for r in citadas(texto)
        if r and r.lower() != "ninguna" and _llave(r) not in llaves
    )


def aviso(fantasmas: tuple[str, ...], var: str) -> str:
    lista = ", ".join(f"`{r}`" for r in fantasmas)
    falta = "no existe" if len(fantasmas) == 1 else "ninguna existe"
    return AVISO.format(prefijo=PREFIJO, rutas=lista, falta=falta, var=var)


def citada(reales: Collection[str], var: str = "ctx") -> Cell:
    """Vota CONTINUE cuando la respuesta cita una ruta que no está en `reales`.

    Va después del executor, como `grounded`, y solo opina sobre prosa: un
    mensaje con código no es una respuesta. `reales` son las rutas del corpus,
    de `rutas(texto)`.
    """
    conjunto = frozenset(reales)

    async def cell(seen: Log) -> Log:
        if not seen.said:
            return ZERO
        ultimo = seen.said[-1]
        if ultimo.role is not Role.ASSISTANT or extract_code(ultimo.text):
            return ZERO
        fantasmas = inventada(ultimo.text, conjunto)
        if not fantasmas:
            return ZERO
        if any(m.role is Role.USER and m.text.startswith(PREFIJO) for m in seen.said):
            return Log(fails=(Fail("cita", f"citó rutas que no existen: {', '.join(fantasmas)}"),))
        return Log(said=(Message(Role.USER, aviso(fantasmas, var)),), vote=Status.CONTINUE)

    return cell
