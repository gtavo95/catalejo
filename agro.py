"""Un asesor agronómico sobre un catálogo que el modelo nunca lee entero.

    uv run agro.py                       sesión interactiva
    uv run agro.py "¿qué uso para trips en tomate?"
    uv run agro.py --ver "..."           además, la transcripción completa
    uv run agro.py --openai "..."        el mismo agente contra OpenAI
    uv run agro.py --plan "..."          además, la checklist del turno
    uv run agro.py --sin-citas "..."     sin la célula que verifica las FUENTE:
    uv run agro.py --sin-ontologia "..." sin `objetivos` en el REPL, la plaga solo por grep
    uv run agro.py --contenedor "..."    el REPL en un proceso hijo que se mata si tarda
    uv run agro.py --un-paso "..."       el grep de antes, que muestra las líneas, sin `read`

El corpus son las 39 fichas de producto y la ontología de `successo-okf`: 360 KB,
~90k tokens. Cada ficha trae `certified_crops` en el frontmatter, con los cultivos
del registro, y un bloque `# Agronomía` en JSON con las plagas con su nombre
científico, y las dosis con su vía, su volumen de agua y sus restricciones. Eso es
lo que hace que la pregunta agronómica se pueda contestar con un hecho en vez de
con una impresión.

# Por qué esta pregunta necesita el REPL

"¿Qué me sirve para mosca blanca en tomate?" es una intersección sobre 39
documentos, y una intersección es justo lo que un retriever hace mal: los
fragmentos más parecidos a la pregunta son los que dicen "mosca blanca", no los
que además certifican tomate. Con código es un filtro de dos condiciones, exacto
y gratis, y el modelo recién lee las tres fichas que quedaron.

La otra mitad del problema es cómo se escribe lo que se busca, y medirlo corrigió
lo que yo creía. La sinonimia casi no molesta: de los 69 conceptos de
`objetivos.md`, 65 aparecen en las fichas con su nombre común, porque la ficha
escribe las dos formas, "Mosca blanca ( Bemisia tabaci )". Lo que sí rompía era
la ortografía. `grep` compilaba el patrón sin banderas, así que `mosca blanca` en
minúscula traía 6 de las 23 líneas que hay y `arana roja` sin la eñe traía cero.
Por eso ahora pliega caso y acentos, y `ontologia/objetivos.md` sigue en el corpus
por las cuatro que sí necesitan el alias.

# Los dos puntos que costaban el turno entero

`ESQUEMA` dice "donde está el dato duro. Ese bloque trae `crops`, `targets`..." y no
"...el dato duro: `crops`, `targets`...". Con los dos puntos, Gemini devolvía turno
vacío en 13 de 13 corridas, o sea 65 llamadas contando reintentos, cortando con
MALFORMED_FUNCTION_CALL sin que este repo declare herramientas en ninguna parte. Sin
ellos, 6 de 6. Es un solo cambio, medido contra el mismo corpus en la misma ronda, y
mientras tanto el eval de la wiki daba 6/6 con los dos rigs.

No sé el mecanismo. Sospecho lo mismo que con `FUENTE: <ruta>` en `evals.py`: dos
puntos seguidos de una lista de identificadores entre backticks se leen como la firma
de una función, y el modelo emite algo que su propio parser rechaza. Las dos filas
están en `bitacora.tsv`. Antes de reescribir esta línea, medila.

# El riesgo que gobierna el contrato

Una dosis inventada se aplica en una hectárea de verdad. El modo de falla que
importa acá no es "no encontró": es "no encontró y contestó igual", que es el
mismo que atrapa `grounded` y el mismo que persiguen los evals. Por eso el
contrato de abajo pide la dosis con su unidad y su base, el archivo de donde
salió, y decir con todas las letras cuándo el catálogo no lo tiene. Un `grep`
que devuelve "0 líneas casan" es un hecho duro; contestar sobre eso de memoria y
sin avisar, no.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from catalejo.core import ZERO, Cell, Log, Message, PlanOp, Role, Status, pendientes, proyectar
from catalejo.llm import Gemini, OpenAI, Provider
from catalejo.rlm import (
    INSTRUCCIONES,
    Handle,
    Registro,
    Repl,
    Verbos,
    citada,
    drive,
    planner,
    recurse,
    render_plan,
    rutas,
)
from catalejo.rlm.repl import sin_acento

BUNDLE = Path(__file__).resolve().parent.parent / "okf" / "successo-okf"

CARPETAS = ("productos", "ontologia")

ESQUEMA = (
    "el catálogo de una empresa de bioinsumos agrícolas, un archivo markdown por página, "
    "cada uno precedido por una línea `=== ruta ===`. En `productos/` hay una ficha por "
    "producto: frontmatter con `tags` (insecticida, fungicida, foliar, apto-organico) y "
    "`certified_crops`, un resumen, presentaciones con precios, ingredientes, y un bloque "
    "`# Agronomía` en JSON que es donde está el dato duro. Ese bloque trae `crop_scope`, "
    "`targets` (cada plaga con su `name`, su `kind` y sus `aliases`, que traen el nombre "
    "científico), `overrides` "
    "con los `regimens` de dosis (`dose` con `min`, `max`, `unit` y `basis`, más `route` y "
    "`water_volume`) y `safety` (pH del agua, incompatibilidades, horas de separación, "
    "fitotoxicidad, días a cosecha). Abajo de todo, `# Ficha` es el texto de la ficha del "
    "fabricante. En `ontologia/` están los vocabularios cerrados: `cultivos.md`, "
    "`objetivos.md` con los alias de cada plaga, `vias.md`, `unidades.md`, `paises.md`"
)

BUSCAR = """- Buscá en el catálogo antes de opinar. Si el término del cliente no aparece, probá el
  nombre científico: `ontologia/objetivos.md` tiene los alias de cada plaga.
"""

BUSCAR_EN_OBJETIVOS = """- Buscá en el catálogo antes de opinar. En el REPL tenés `objetivos`, el vocabulario
  cerrado de plagas, una lista de dicts con `id`, `etiqueta`, `padre`, `alias` (lista),
  `nota` y `fichas`, las rutas de los productos que cubren esa plaga, con las especies
  hijas incluidas. Se filtra con Python, no con `grep`, que es para el texto. Buscá lo
  que dice el cliente en `etiqueta` y en `alias`; si casa, `fichas` dice qué leer. Si no
  casa con nada, el catálogo no cubre esa plaga, y eso se dice antes de recomendar.
"""

TIPO = """- Si el cliente pide un tipo de producto y no una plaga (un adherente, algo para el agua,
  un enraizador), el eje es `tags` en `productos`. Son pocos, así que listalos todos en
  una línea y elegí de ahí, en vez de adivinar con grep la palabra que usa la ficha.
  `ontologia/tipos-producto.md` tiene la etiqueta y los alias de cada tag.
"""
"""El tercer eje, después de plaga y cultivo: qué tipo de cosa pide el cliente.

Salió de "algo para el tratamiento del agua". Novicor 96 SP existe y es un corrector
de dureza, y el modelo lo daba por inexistente 2 de 3 veces: buscaba `corrector de
pH|buffer|acidificante` y la ficha dice "corrector de dureza", "buferiza". Con grep
hay que adivinar la palabra; con un conjunto cerrado no. El tag estaba en `productos`
desde siempre y la hoja de alias en el corpus; lo que faltaba era una línea que
dijera que el eje existe. Va acá y no en el preámbulo por la misma medición que
`nota()`, y no es un builtin porque el dato ya está. Medido (filas `eje_tipo` de la
bitácora, n=3, 10 casos): novicor 2/3 -> 3/3, cero regresiones, +3% tokens que es
ruido. Si un día no alcanza, el paso siguiente es `tipos` como dato, igual que
`objetivos`, con el join por `tags`.
"""

CONTRATO = f"""

Para contestar esto:

{BUSCAR}{TIPO}- Si hay producto, decí la dosis tal cual está: el número, la unidad y la base (l/ha),
  la vía de aplicación y el volumen de agua. No conviertas unidades ni promedies un
  rango sin decir que lo hiciste.
- Decí si el cultivo está en `certified_crops` de ese producto o no. `crop_scope` separa
  dónde está registrado de dónde funciona, y son dos hechos distintos.
- Traé lo que diga `safety`: incompatibilidades, horas de separación, pH del agua, días
  a cosecha. Es la mitad de la recomendación.
- Si el catálogo no lo tiene, decilo con todas las letras. Recién ahí, si sabés la
  respuesta por agronomía general, dala marcada como tal.
- En el REPL tenés `productos`, una lista con una entrada por ficha ya parseada.
  Imprimí `productos[0]` para ver qué trae; filtrar ahí es exacto y gratis.
- `productos` es un índice derivado del texto, no la fuente. Si el índice y la ficha
  difieren, manda la ficha, y para lo que el índice no modela (dosis por manzana, forma
  de aplicación, hora del día) andá al texto con grep.
- Cerrá con una línea `FUENTE: productos/x.md` por cada archivo del que sacaste un dato,
  o `FUENTE: ninguna` si no salió del catálogo.
"""


def contrato(*, ontologia: bool = True) -> str:
    """El CONTRATO, que nombra `objetivos`, y con `ontologia=False` el que manda a la hoja.

    Cambia una sola viñeta, la primera, porque es la que dice qué hacer cuando el
    término del cliente no aparece. Hoy la respuesta es "probá el nombre
    científico" y un grep sobre `ontologia/objetivos.md`, que devuelve la fila como
    texto. Con el vocabulario en el REPL la respuesta es recorrer una lista: si casa,
    la entrada trae las `fichas` que leer; si no casa, el catálogo no lo cubre y el
    cero es una afirmación sobre un conjunto cerrado, no una ausencia en un texto. Es
    palanca de prompt y de builtin a la vez, así que se mide con las dos juntas.
    """
    return CONTRATO.replace(BUSCAR, BUSCAR_EN_OBJETIVOS) if ontologia else CONTRATO


REGISTRO: Registro = {
    "consulto_el_catalogo": lambda log: log.reads > 0,
    "mire_una_ficha": lambda log: any(
        m.role is Role.USER and "productos/" in m.text and ".md:" in m.text for m in log.said
    ),
    "cito_la_fuente": lambda log: any(
        m.role is Role.ASSISTANT and "FUENTE:" in m.text for m in log.said
    ),
}
"""Las compuertas del plan: hechos sobre el Log, no campos que el Log no tiene.

Los tres son pisos y no pruebas, igual que `reads` para `grounded`.
`consulto_el_catalogo` dice que el REPL devolvió algo, no que sea lo que hacía
falta. `mire_una_ficha` busca el prefijo `productos/x.md:` que `grep` le pega a
cada hit, así que dice que una LÍNEA de una ficha le pasó por delante, no que la
haya leído: es el piso que separa contestar desde el índice de contestar desde la
ficha, y la dosis tiene que salir de la ficha. `cito_la_fuente` mira lo mismo que
después lee `evals.citadas`, así que el requisito y la medición no se pueden
desincronizar.

No hay un `sin_fallas`. Sería verdad en el paso 1, cuando todavía no falló nada,
así que cerraría su paso antes de que el paso empiece. Una compuerta que se cumple
sola no es una compuerta.
"""

SEMILLA = (
    PlanOp(
        "add_step",
        "buscar",
        "encontrar qué productos sirven para esa plaga en ese cultivo",
        completes_when="consulto_el_catalogo",
    ),
    PlanOp(
        "add_step",
        "dosis",
        "traer la dosis como está: número, unidad, base, vía y volumen de agua",
        completes_when="mire_una_ficha",
    ),
    PlanOp("add_step", "cultivo", "decir si el cultivo está en certified_crops de ese producto"),
    PlanOp("add_step", "seguridad", "traer safety: incompatibilidades, horas, pH, días a cosecha"),
    PlanOp(
        "add_step",
        "fuente",
        "cerrar con una línea FUENTE por cada archivo del que salió un dato",
        completes_when="cito_la_fuente",
    ),
)
"""El esqueleto que declara el host, que es el CONTRATO escrito como checklist.

Mixto a propósito: tres pasos con compuerta, que se cierran cuando el hecho pasa,
y dos en la válvula, que los cierra el modelo. Los dos de la válvula son los que
no tienen un hecho observable en el Log: que el cultivo esté certificado y que la
seguridad esté traída son afirmaciones sobre el TEXTO de la respuesta, y
verificarlas pide un juez. Mientras no haya juez, el plan dice que se hicieron
porque el modelo lo dijo, y eso es todo lo que dice.

El modelo puede agregar los suyos con `add_step`, que es lo que `flex="acotado"`
le permite. Lo que no puede es borrar ni reescribir estos, porque `revise` no
existe.
"""


def corpus() -> str:
    """Las fichas y la ontología, cada archivo con su ruta adelante.

    Sin `company_info/`, que es horarios y cuentas bancarias, sin `raw/`, que es el
    OCR crudo de lo que la ficha ya trae curado, y sin `evals/`, que tiene las
    respuestas escritas al lado de las preguntas.
    """
    partes = []
    for f in sorted(BUNDLE.rglob("*.md")):
        rel = f.relative_to(BUNDLE)
        if rel.parts[0] not in CARPETAS:
            continue
        partes.append(f"=== {rel} ===\n{f.read_text(errors='replace')}")
    return "\n\n".join(partes)


CAMPOS = ("title", "estado", "tags", "certified_crops")


def frontmatter(texto: str) -> dict[str, object]:
    """Los cuatro campos del frontmatter que se consultan, sin traer un parser de YAML.

    Lee escalares (`clave: valor`) y listas de guion, que son las dos formas que usa
    el bundle en esas cuatro claves. Lo que no entra en esas dos formas queda afuera,
    y no se pierde: el texto completo sigue en `catalogo` y el grep lo alcanza.
    """
    if not texto.startswith("---\n"):
        return {}
    cuerpo = texto[4:].split("\n---", 1)[0]
    datos: dict[str, object] = {}
    clave = ""
    for linea in cuerpo.splitlines():
        if linea.startswith("  - "):
            if clave in CAMPOS:
                lista = datos.setdefault(clave, [])
                if isinstance(lista, list):
                    lista.append(linea[4:].strip().strip('"'))
        elif not linea.startswith(" ") and ":" in linea:
            clave, valor = (x.strip() for x in linea.split(":", 1))
            if valor and clave in CAMPOS:
                datos[clave] = valor.strip('"')
    return datos


def agronomia(texto: str) -> dict[str, Any]:
    """El bloque `# Agronomía` parseado, o vacío con el motivo si no se pudo."""
    if "\n# Agronomía" not in texto:
        return {}
    tras = texto.split("\n# Agronomía", 1)[1]
    if tras.count("```") < 2:
        return {"error": "el bloque no tiene código cercado"}
    try:
        datos = json.loads(tras.split("```", 2)[1].removeprefix("json"))
    except json.JSONDecodeError as e:
        return {"error": f"JSON inválido: {e}"}
    return datos if isinstance(datos, dict) else {"error": "el bloque no es un objeto"}


def registro(ruta: Path, texto: str, hoja: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Un producto como dato consultable: lo que el frontmatter y el bloque ya dicen.

    Es un índice, no una fuente. Nada acá se calcula ni se normaliza, con una
    excepción: `cultivos` son los ids de `certificados` resueltos por la hoja de
    cultivos. Hasta el 13 de septiembre de 2026 el perfil repetía esa lista en
    `crops`, en ids, y el índice la copiaba; ese día el bundle alineó las 24 fichas
    con lista a `crop_scope: all_crops`, que rechaza `crops`, así que la lista vive
    una sola vez, en el frontmatter, con las palabras de la ficha. Resolverla acá es
    lo mismo que hacía `verificar.sh` para exigir que las dos copias coincidieran,
    y deja el índice como estaba: el id es lo que usan los `overrides` para decir en
    qué cultivo cambia la dosis, y filtrar por id no depende de la tilde.
    `certificados` sigue tal cual, y con un escalar (`all_crops`, `not_applicable`)
    `cultivos` queda vacío. Las dosis van como están, con su unidad y su base. Un
    campo vacío quiere decir que la ficha no lo trae, y las 38 fichas parsean, así
    que vacío nunca quiere decir que el parser falló.

    `dosis[].para` es el `match` del override tal cual, que puede venir por plaga
    (`target_ids`), por cultivo (`crops`) o por vía (`routes`). Aplanarlo a una sola
    de las tres inventaría una regla que la ficha no dice. Vacío es el régimen por
    defecto, el que rige cuando ningún override matchea.
    """
    fm = frontmatter(texto)
    ag = agronomia(texto)
    certificados = fm.get("certified_crops", [])
    dosis = [
        {"para": {}, "dosis": r.get("dose", {}), "via": r.get("route", ""), "agua": r.get("water_volume", {})}
        for r in ag.get("default_regimens", [])
    ] + [
        {
            "para": o.get("match", {}),
            "dosis": r.get("dose", {}),
            "via": r.get("route", ""),
            "agua": r.get("water_volume", {}),
        }
        for o in ag.get("overrides", [])
        for r in o.get("regimens", [])
    ]
    return {
        "nombre": ruta.stem,
        "titulo": fm.get("title", ruta.stem),
        "estado": fm.get("estado", ""),
        "ruta": f"productos/{ruta.name}",
        "tags": fm.get("tags", []),
        "certificados": fm.get("certified_crops", []),
        "cultivos": resolver(certificados, hoja or {}),
        "alcance": ag.get("crop_scope", ""),
        "plagas": [
            {"id": t.get("id", ""), "nombre": t.get("name", ""), "alias": t.get("aliases", [])}
            for t in ag.get("targets", [])
        ],
        "dosis": dosis,
        "seguridad": ag.get("safety", {}),
        **({"error": ag["error"]} if ag.get("error") else {}),
    }


def indice() -> list[dict[str, Any]]:
    """Un registro por ficha, de los mismos archivos que arma `corpus()`.

    El OCR del fabricante no se separa en una variable aparte, aunque era la mitad
    del plan. Nombrarlo en el preámbulo costaba caro y la razón está abajo, en
    `nota`. Sigue alcanzable donde siempre estuvo: adentro de `catalogo`, por grep.
    """
    hoja = claves(vocabulario((BUNDLE / "ontologia" / "cultivos.md").read_text(errors="replace")))
    return [
        registro(f, f.read_text(errors="replace"), hoja)
        for f in sorted((BUNDLE / "productos").glob("*.md"))
        if f.name != "index.md"
    ]


def claves(hoja: list[dict[str, Any]]) -> dict[str, str]:
    """De una hoja de ontología, el diccionario término plegado -> id.

    Entran el id, la etiqueta y cada alias, plegando caso y acentos con el mismo
    `sin_acento` de `grep`, porque la ficha escribe "Güicoy" y "Sandía" y la hoja
    los lista como alias con o sin tilde. Si dos filas reclaman el mismo término,
    gana la primera, que es lo que hace `verificar.sh` al leer la hoja de arriba
    abajo.
    """
    tabla: dict[str, str] = {}
    for fila in hoja:
        for termino in (fila.get("id", ""), fila.get("etiqueta", ""), *fila.get("alias", [])):
            clave = sin_acento(termino).lower().strip()
            if clave:
                tabla.setdefault(clave, fila["id"])
    return tabla


def resolver(terminos: object, hoja: Mapping[str, str]) -> list[str]:
    """Los ids de una lista de cultivos escrita como la escribe la ficha.

    Un término que la hoja no conoce se conserva plegado, en minúscula y sin tilde,
    porque tirarlo sería un cultivo certificado que desaparece del índice sin aviso
    e inventarle un id sería peor. Sobre el bundle no pasa: `verificar.sh` rechaza
    la página. Un escalar (`all_crops`, `not_applicable`) no es una lista y da vacío.
    """
    if not isinstance(terminos, list):
        return []
    vistos: list[str] = []
    for termino in terminos:
        clave = sin_acento(str(termino)).lower().strip()
        id_ = hoja.get(clave, clave)
        if id_ and id_ not in vistos:
            vistos.append(id_)
    return vistos


def vocabulario(texto: str) -> list[dict[str, Any]]:
    """La tabla de conceptos de una hoja de `ontologia/`, una entrada por fila.

    Lee como manda `FORMATO.md`: la primera tabla que abre con `| id |` es la de
    conceptos, y lo de arriba es prosa. Las columnas salen del encabezado y no de
    una lista fija, así que una hoja con una columna más la trae. `alias` es la
    única que se parte, por coma, porque es la única que el formato define como
    lista. `padre` vacío queda vacío, que es como la hoja escribe una raíz.
    """
    lineas = iter(texto.splitlines())
    columnas: list[str] = []
    for linea in lineas:
        if linea.startswith("| id |"):
            columnas = [c.strip() for c in linea.strip().strip("|").split("|")]
            next(lineas, None)
            break
    entradas: list[dict[str, Any]] = []
    for linea in lineas:
        if not linea.startswith("|"):
            break
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        fila: dict[str, Any] = dict(zip(columnas, celdas))
        if "alias" in fila:
            fila["alias"] = [a.strip() for a in fila["alias"].split(",") if a.strip()]
        entradas.append(fila)
    return entradas


def objetivos(regs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Las plagas como dato: la hoja parseada, y en cada concepto las `fichas` que lo cubren.

    Los `id` de la hoja son los que las fichas usan en `targets`, verificado sobre
    el bundle: los 56 que usan las fichas están todos en la hoja, y los 13 que
    sobran son los padres de agrupación (`chupadores`, `masticadores`), que ninguna
    ficha nombra y que son el árbol.

    `fichas` es el join hecho acá y no por el modelo. La primera versión le daba
    solo el `id` y la viñeta decía que era el mismo de `plagas`; en 3 de 6 corridas
    de rodenticida el modelo escribió el join mal (el dict de `plagas` contra una
    lista de ids), obtuvo cero, y en dos de esas cerró con "el catálogo contempla
    ratas pero ningún producto las cubre". Con la hoja como texto ese cero no
    existía, porque `grep('rata')` caía en la ficha directo. Un dato que abre un
    camino nuevo a un falso cero es peor que el texto, así que el camino se cierra
    acá: la ruta viene en la entrada, y va como ruta porque es lo que `doc=` acota
    y lo que `FUENTE:` cita.

    Un padre cubre lo que cubren sus hijos, que es la regla de `FORMATO.md` ("una
    consulta por pulgón alcanza las tres especies") escrita una sola vez.
    """
    hoja = vocabulario((BUNDLE / "ontologia" / "objetivos.md").read_text(errors="replace"))
    hijos: dict[str, list[str]] = {}
    for e in hoja:
        hijos.setdefault(e["padre"], []).append(e["id"])
    directas: dict[str, set[str]] = {}
    for r in regs:
        for plaga in r["plagas"]:
            directas.setdefault(plaga["id"], set()).add(r["ruta"])

    def cubre(id_: str) -> set[str]:
        rutas = set(directas.get(id_, ()))
        for hijo in hijos.get(id_, ()):
            rutas |= cubre(hijo)
        return rutas

    return [{**e, "fichas": sorted(cubre(e["id"]))} for e in hoja]


def nota(regs: list[dict[str, Any]]) -> str:
    """Lo que el preámbulo NO dice sobre el índice, y por qué.

    Devuelve vacío. El índice existe, vive en el namespace, y se nombra en el
    `CONTRATO`, que viaja en el turno del usuario. Nombrarlo en el preámbulo, que es
    lo que pide la regla del repo, rompía las corridas: `gemini-3.8-flash` con
    `thinking=low` devolvía turnos vacíos, una sola parte con la firma del
    razonamiento y texto en blanco, que la API etiqueta `MALFORMED_FUNCTION_CALL`.

    Doce llamadas idénticas por variante, misma pregunta, y la medición repetida en
    tres rondas distintas porque la tasa del proveedor se mueve sola entre rondas:

        sin índice                                  6/12   (la base de ese día)
        el índice nombrado en el turno del usuario  6/12
        el índice nombrado en el preámbulo          0/12

    Lo de la izquierda es lo mismo escrito en dos lugares. `toolConfig` en NONE y
    `tools: []` no cambian nada, y `thinking=high` levanta la tasa pero gasta 55 mil
    tokens por paso, que es más que la pregunta entera.

    La regla sigue siendo que un builtin que no se nombra es uno que el modelo no
    usa. Lo que este caso agrega es dónde nombrarlo. La función se queda, vacía y con
    la medición al lado, porque el default obvio es volver a escribir el párrafo en
    el preámbulo y perder dos horas averiguando por qué el agente no arranca.
    """
    return ""


def traza() -> Cell:
    """Imprime lo que pasó en este paso. Mira y no manda.

    Devuelve `ZERO`, así que no dice nada, no vota y no puede quedarse con el loop
    de rehén. Es la célula más barata que existe y es lo único que hace falta para
    ver el agente trabajando: va última en el `then`, y como cada célula ve lo que
    llegó más lo que agregaron las anteriores, ahí adentro ya están el código que
    propuso el modelo y lo que devolvió el REPL.
    """

    async def cell(seen: Log) -> Log:
        """Mira `said` para lo que se dijo y `steps` para el plan, que no se dice.

        La ventana arranca en el último dicho del modelo, que es exactamente donde
        arrancó este paso: el modelo habla una vez y todo lo que viene después lo
        agregaron las células de este paso. Un tope fijo de dos o tres mensajes
        parecía lo mismo y reimprimía el mensaje del paso anterior en cuanto el
        plan o el grounding agregaban uno.
        """
        desde = max(
            (i for i, m in enumerate(seen.said) if m.role is Role.ASSISTANT),
            default=len(seen.said),
        )
        movio = False
        for m in seen.said[desde:]:
            if m.role is Role.ASSISTANT and "```" in m.text:
                bloque = m.text.split("```")[1].removeprefix("python").strip()
                if bloque:
                    print(f"\033[2m  › {bloque.splitlines()[0][:100]}\033[0m")
            elif m.role is Role.USER and m.text.startswith("[repl]"):
                lineas = m.text.splitlines()
                if len(lineas) > 1:
                    print(f"\033[2m    {lineas[1][:100]}\033[0m")
            elif m.role is Role.USER and m.text.startswith("[plan]"):
                movio = True
                for linea in m.text.splitlines():
                    if linea.startswith("rechazado"):
                        print(f"\033[2m    {linea[:100]}\033[0m")
        if movio:
            plan = proyectar(seen.steps)
            abiertos = ", ".join(s.id for s in pendientes(plan))
            hechos = len(plan) - len(pendientes(plan))
            print(f"\033[2m    plan {hechos}/{len(plan)}: falta {abiertos or 'nada'}\033[0m")
        return ZERO

    return cell


def armar(
    texto: str,
    modelo: Provider,
    *,
    ver: bool,
    plan: bool = False,
    citas: bool = True,
    ontologia: bool = True,
    contenedor: bool = False,
    dos_pasos: bool = True,
) -> tuple[Cell, Repl]:
    """El agente y su workspace, que sobreviven a toda la sesión.

    El workspace se arma una sola vez a propósito. Las variables persisten entre
    corridas, así que el índice que el modelo construya en la primera pregunta
    sigue vivo en la quinta y no hay que volver a pagarlo.

    `depth=0` le da `llm` y no `rlm`: sobre 360 KB nunca hace falta un sub-agente
    con su propio REPL, y sí hace falta mandar a leer tres fichas en paralelo, que
    es lo que el código no puede hacer.

    El índice va por `extra`, así que vive en el namespace del REPL y no en el
    prompt. Lo que el prompt paga es la `nota`, que son veinte líneas describiendo
    la forma de un registro, no los 38 registros.

    Con `plan=True` los tres verbos entran por la misma puerta que el índice y la
    célula que los juzga va después de `grounded`. La bandera existe porque esto
    cambia la terminación del agente, que es lo más delicado que tiene, y hay que
    poder medir las dos ramas en la misma tarde.

    `citada` va por default y `citas=False` la saca. Compara las `FUENTE:` de la
    respuesta contra las rutas del corpus y avisa cuando una no existe. Se midió
    (filas `cita_cerrada` de la bitácora): con luna no habló en 54 corridas y no
    dio un falso positivo, así que no cambia nada de lo que el modelo ve hasta
    que cite una ficha inventada, que es justo cuando tiene que hablar. Es un
    regex sobre el último mensaje; el conjunto sale del texto acá, en el padre,
    porque el corpus vive en el Environment y la célula no.

    `traza` va después del planner y no antes: mira los últimos dichos del paso, y
    si corriera primero, el mensaje del plan todavía no existiría.

    `ontologia` mete `objetivos` por la misma puerta que `productos` y cambia la
    primera viñeta del CONTRATO (ver `contrato`). Estuvo apagada desde que se
    midió sobre nueve casos simples (filas `ontologia_como_dato` de la bitácora):
    no movía aciertos, zompopo cerraba en 3 turnos siempre en vez de 5 a 13, y
    cogollero pagaba un turno más porque `grep('cogollo')` cae en la ficha
    directo. Es default desde el 14 de septiembre de 2026 por la pregunta
    compuesta, "salivazo en la caña, un enraizador y algo para el picudo": con
    grep, el OR de los tres pedidos casa en 38 documentos y la ficha de Biomet,
    con 3 hits, queda enterrada, o el modelo la encuentra en el turno 2 y la
    pierde del prompt antes de contestar. Salivazo salía 1/3 y con `objetivos`
    3/3, en 5 a 7 mensajes en vez de 17 (filas `ontologia_default`).
    `ontologia=False` (`--sin-ontologia`) restaura el arm anterior.

    `contenedor=True` corre el código del modelo en un proceso hijo que se mata
    si tarda; `llm` y el índice siguen acá. Lo que el modelo ve no cambia. El que
    llama cierra el workspace al terminar, porque un proceso no se va solo.

    `dos_pasos` va por default: el `grep` que solo dice dónde y el `read` que trae
    la ficha entera, como hace una persona con una carpeta: buscar, elegir, abrir.
    Cambia la nota de herramientas y nada más del prompt; el CONTRATO sigue
    diciendo "andá al texto con grep", que ahora son dos consultas. Se midió
    (filas `dos_pasos` de la bitácora): 8/9 estables en los dos arms, +4% tokens
    que es ruido, 4.6 a 5.1 turnos. Es default porque la respuesta sale de la
    ficha entera, con la equivalencia por manzana y la seguridad adentro, y eso
    es lo que el que atiende necesita aunque `acierta()` no lo mida.
    `dos_pasos=False` (`--un-paso`) restaura el arm anterior.
    """
    regs = indice()
    verbos = Verbos()
    extra: dict[str, Any] = {"productos": regs}
    if ontologia:
        extra["objetivos"] = objetivos(regs)
    if plan:
        extra.update(verbos.builtins)
    ws = recurse(
        texto,
        modelo,
        var="catalogo",
        depth=0,
        budget=60_000,
        extra=extra,
        nota="",
        contenedor=contenedor,
        dos_pasos=dos_pasos,
    )
    h = Handle(
        var=ws.var,
        schema=ESQUEMA,
        size=f"{len(texto) // 1000} KB, ~{len(texto) // 4000}k tokens",
        tools=ws.tools,
    )
    extras: list[Cell] = []
    if citas:
        extras.append(citada(rutas(texto), var=ws.var))
    if plan:
        extras.append(planner(verbos, REGISTRO, semilla=SEMILLA))
    if not ver:
        extras.append(traza())
    return drive(modelo, h, ws, keep_recent=6, max_steps=12, budget=150_000, extras=extras), ws


def pedido(
    pregunta: str,
    historia: tuple[tuple[str, str], ...],
    *,
    plan: bool = False,
    ontologia: bool = True,
) -> str:
    """La pregunta de ahora, con lo ya hablado adentro del MISMO mensaje.

    Va todo junto porque `window` conserva `said[0]` y los últimos N, así que el
    primer mensaje es el único que no se recorta nunca. Con las preguntas viejas
    como mensajes sueltos, la que sobrevive al recorte es la primera de la sesión
    y la de ahora se cae a los seis pasos de REPL.

    Las respuestas van cortadas: alcanzan para resolver un "¿y en banano?" y no
    para arrastrar media sesión en cada turno.

    Con `plan=True`, acá entran los verbos y la semilla ya dibujada. Van en el
    turno del usuario y NO en el preámbulo, y eso no es gusto: `nota` tiene la
    medición de lo que pasa cuando algo así se nombra en el preámbulo, que es 0 de
    12 corridas vivas contra 6 de 12.
    """
    texto = contrato(ontologia=ontologia)
    if plan:
        texto += f"\n{INSTRUCCIONES}\n\n{render_plan(proyectar(SEMILLA))}\n"
    if not historia:
        return pregunta + texto
    previas = "\n\n".join(f"P: {p}\nR: {r[:500]}" for p, r in historia)
    return (
        f"[lo que ya hablamos en esta sesión]\n{previas}\n\n"
        f"[la pregunta de ahora]\n{pregunta}{texto}"
    )


SIN_CONSULTAR = "No pude revisar el catálogo, así que no tengo qué recomendarte. Volvé a preguntar."


def final(out: Log) -> str:
    """El último dicho que no es código, salvo que no se haya consultado nada.

    La terminación de este agente es sintáctica: prosa quiere decir terminé. El
    modelo, en cambio, tiene tres estados y solo uno se ve distinto desde afuera,
    así que un turno donde está pensando en voz alta entra por esta función como
    respuesta final. Medido contra luna con esfuerzo bajo, una corrida contestó
    esto, literalmente:

        Respond? We need code block only to execute. But final instruction says
        code block makes execution. Do it.

    Es el modelo deliberando sobre la última regla del preámbulo. La API no tiene
    canal de razonamiento en chat/completions (`message` trae `content`, `refusal`
    y nada más, y los `reasoning_tokens` se facturan sin volver), así que eso llegó
    como contenido y no hay forma de distinguirlo por la forma.

    `grounded` es lo único que lo ve, y avisa una vez: si el modelo lo ignora anota
    el `Fail` y deja salir la respuesta etiquetada. La etiqueta llega al CLI como
    una línea gris y no llega a nadie detrás de una API. Así que acá se corta. Una
    respuesta que no consultó el catálogo no es una respuesta con una advertencia,
    es un no sé, y en un catálogo agronómico la diferencia se aplica en una
    hectárea.

    El dicho crudo sigue en el `Log` para auditar. Lo que cambia es lo que se
    sirve.

    La frase es de ESTE cliente y no del motor, y por eso vive acá. El que
    pregunta es una persona en una terminal, así que lo único que puede recibir es
    texto y lo que necesita saber es que no hay recomendación. Por qué no la hay
    es asunto nuestro: nombrarle la memoria, el grounding o las fichas es
    explicarle nuestra plomería a alguien que quería saber qué echarle al cultivo.

    El servicio no usa esta frase ni la quiere. Su `final` devuelve "" y la
    respuesta viaja con `fundada: false` y `fails` aparte, porque del otro lado
    hay una app o un agente que lee un booleano, no una oración. Y `evals.final`
    no reemplaza nada, porque ahí el texto crudo es el dato. Tres lectores, tres
    políticas, un solo hecho abajo: `Fail("grounding", ...)`.
    """
    if any(f.who == "grounding" for f in out.fails):
        return SIN_CONSULTAR
    for m in reversed(out.said):
        if m.role is Role.ASSISTANT and "```" not in m.text:
            return m.text.strip()
    return "(se quedó sin pasos antes de contestar)"


def cuenta(out: Log, ws: Repl) -> str:
    b = ws.bridge
    delegado = f", {b.calls} lecturas delegadas" if b is not None and b.calls else ""
    fallas = f", fails: {[f.reason for f in out.fails]}" if out.fails else ""
    return f"{out.spent:,} tokens, {len(out.said)} turnos{delegado}{fallas}"


async def responder(
    agente: Cell,
    ws: Repl,
    pregunta: str,
    historia: tuple[tuple[str, str], ...],
    *,
    ver: bool,
    plan: bool = False,
    ontologia: bool = True,
) -> str:
    """Una pregunta, contestada. El Log arranca limpio cada vez.

    Solo viaja la prosa de las respuestas anteriores. Si arrastrara el Log entero,
    `reads` vendría en más de cero desde el primer paso y `grounded` no volvería a
    disparar en toda la sesión: la pregunta cinco podría contestarse de memoria
    amparada en el grep de la pregunta uno.
    """
    dicho = pedido(pregunta, historia, plan=plan, ontologia=ontologia)
    out = await agente(Log(said=(Message(Role.USER, dicho),)))
    if ver:
        for m in out.said:
            quien = {Role.USER: "repl", Role.ASSISTANT: "modelo"}.get(m.role, m.role.value)
            print(f"\n\033[1m[{quien}]\033[0m {m.text.strip()[:2000]}")
    respuesta = final(out)
    print(f"\n{respuesta}\n")
    print(f"\033[2m[{cuenta(out, ws)}]\033[0m")
    if out.vote is Status.HALT:
        print("\033[2m[el agente frenó por su cuenta]\033[0m")
    return respuesta


def proveedor(argv: list[str]) -> Provider:
    """El modelo de la raíz, y con qué proveedor se habla.

    Es un `if` y no una capa de abstracción porque los dos adaptadores toman los
    mismos parámetros y los dos cumplen `Provider`, que es el puerto ancho: el
    resto de `agro.py` no vuelve a nombrar a ninguno de los dos. `thinking="low"` en los dos: la raíz decide cuál es la
    próxima consulta, que es razonamiento barato, y el trabajo pesado lo hace el
    REPL gratis.

    El `retries=4` va en los dos por el mismo motivo, y no es cautela genérica.
    Los dos proveedores devuelven turnos vacíos cuando el modelo gasta todo el
    presupuesto de salida razonando y no escribe nada: sobre Gemini, doce llamadas
    idénticas daban seis turnos buenos y seis vacíos.
    """
    if "--openai" in argv:
        return OpenAI(thinking="low", retries=4)
    return Gemini(thinking="low", retries=4)


async def main(argv: list[str]) -> None:
    ver = "--ver" in argv
    plan = "--plan" in argv
    citas = "--sin-citas" not in argv
    ontologia = "--sin-ontologia" not in argv
    contenedor = "--contenedor" in argv
    dos_pasos = "--un-paso" not in argv
    pregunta = " ".join(a for a in argv if not a.startswith("-"))

    texto = corpus()
    fichas = texto.count("=== productos/")
    modelo = proveedor(argv)
    agente, ws = armar(
        texto,
        modelo,
        ver=ver,
        plan=plan,
        citas=citas,
        ontologia=ontologia,
        contenedor=contenedor,
        dos_pasos=dos_pasos,
    )
    historia: tuple[tuple[str, str], ...] = ()

    print(
        f"\033[1magro\033[0m · {fichas} fichas, {len(texto) // 1000} KB "
        f"(~{len(texto) // 4000}k tokens) que el modelo consulta con código "
        f"· {modelo.model}{' · con plan' if plan else ''}{' · sin citas' if not citas else ''}"
        f"{' · sin ontología' if not ontologia else ''}"
        f"{' · en un proceso hijo' if contenedor else ''}"
        f"{' · en un paso' if not dos_pasos else ''}"
    )

    try:
        if pregunta:
            await responder(agente, ws, pregunta, (), ver=ver, plan=plan, ontologia=ontologia)
            return
        print("\033[2mpreguntá, o Ctrl-D para salir\033[0m")
        while True:
            try:
                # input() bloquea el event loop y no importa: no hay nada más corriendo.
                pregunta = input("\n\033[1m> \033[0m").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not pregunta:
                continue
            respuesta = await responder(
                agente, ws, pregunta, historia, ver=ver, plan=plan, ontologia=ontologia
            )
            historia = (*historia, (pregunta, respuesta))[-3:]
    finally:
        ws.cerrar()
        await modelo.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
