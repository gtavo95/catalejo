"""Un asesor agronómico sobre un catálogo que el modelo nunca lee entero.

    uv run agro.py                       sesión interactiva
    uv run agro.py "¿qué uso para trips en tomate?"
    uv run agro.py --ver "..."           además, la transcripción completa
    uv run agro.py --openai "..."        el mismo agente contra OpenAI
    uv run agro.py --plan "..."          además, la checklist del turno
    uv run agro.py --sin-citas "..."     sin la célula que verifica las FUENTE:

El corpus son las 39 fichas de producto y la ontología de `successo-okf`: 360 KB,
~90k tokens. Cada ficha trae un bloque `# Agronomía` en JSON con los cultivos
certificados, las plagas con su nombre científico, y las dosis con su vía, su
volumen de agua y sus restricciones. Eso es lo que hace que la pregunta
agronómica se pueda contestar con un hecho en vez de con una impresión.

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
from pathlib import Path
from typing import Any

from catalejo.core import ZERO, Cell, Log, Message, PlanOp, Role, Status, pendientes, proyectar
from catalejo.llm import Gemini, OpenAI, Provider
from catalejo.repl import (
    INSTRUCCIONES,
    Handle,
    Registro,
    Verbos,
    Workspace,
    citada,
    drive,
    planner,
    recurse,
    render_plan,
    rutas,
)

BUNDLE = Path(__file__).resolve().parent.parent / "okf" / "successo-okf"

CARPETAS = ("productos", "ontologia")

ESQUEMA = (
    "el catálogo de una empresa de bioinsumos agrícolas, un archivo markdown por página, "
    "cada uno precedido por una línea `=== ruta ===`. En `productos/` hay una ficha por "
    "producto: frontmatter con `tags` (insecticida, fungicida, foliar, apto-organico) y "
    "`certified_crops`, un resumen, presentaciones con precios, ingredientes, y un bloque "
    "`# Agronomía` en JSON que es donde está el dato duro. Ese bloque trae `crops`, "
    "`targets` (cada plaga con su `name`, su `kind` y sus `aliases`, que traen el nombre "
    "científico), `overrides` "
    "con los `regimens` de dosis (`dose` con `min`, `max`, `unit` y `basis`, más `route` y "
    "`water_volume`) y `safety` (pH del agua, incompatibilidades, horas de separación, "
    "fitotoxicidad, días a cosecha). Abajo de todo, `# Ficha` es el texto de la ficha del "
    "fabricante. En `ontologia/` están los vocabularios cerrados: `cultivos.md`, "
    "`objetivos.md` con los alias de cada plaga, `vias.md`, `unidades.md`, `paises.md`"
)

CONTRATO = """

Para contestar esto:

- Buscá en el catálogo antes de opinar. Si el término del cliente no aparece, probá el
  nombre científico: `ontologia/objetivos.md` tiene los alias de cada plaga.
- Si hay producto, decí la dosis tal cual está: el número, la unidad y la base (l/ha),
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


def registro(ruta: Path, texto: str) -> dict[str, Any]:
    """Un producto como dato consultable: lo que el frontmatter y el bloque ya dicen.

    Es un índice, no una fuente. Nada acá se calcula ni se normaliza: los cultivos
    certificados van como los escribe la ficha y las dosis van como están, con su
    unidad y su base. Un campo vacío quiere decir que la ficha no lo trae, y las 38
    fichas parsean, así que vacío nunca quiere decir que el parser falló.

    `dosis[].para` es el `match` del override tal cual, que puede venir por plaga
    (`target_ids`), por cultivo (`crops`) o por vía (`routes`). Aplanarlo a una sola
    de las tres inventaría una regla que la ficha no dice. Vacío es el régimen por
    defecto, el que rige cuando ningún override matchea.
    """
    fm = frontmatter(texto)
    ag = agronomia(texto)
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
        "cultivos": ag.get("crops", []),
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
    return [
        registro(f, f.read_text(errors="replace"))
        for f in sorted((BUNDLE / "productos").glob("*.md"))
        if f.name != "index.md"
    ]


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
    texto: str, modelo: Provider, *, ver: bool, plan: bool = False, citas: bool = True
) -> tuple[Cell, Workspace]:
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
    """
    regs = indice()
    verbos = Verbos()
    ws = recurse(
        texto,
        modelo,
        var="catalogo",
        depth=0,
        budget=60_000,
        extra={"productos": regs, **verbos.builtins} if plan else {"productos": regs},
        nota="",
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


def pedido(pregunta: str, historia: tuple[tuple[str, str], ...], *, plan: bool = False) -> str:
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
    contrato = CONTRATO
    if plan:
        contrato += f"\n{INSTRUCCIONES}\n\n{render_plan(proyectar(SEMILLA))}\n"
    if not historia:
        return pregunta + contrato
    previas = "\n\n".join(f"P: {p}\nR: {r[:500]}" for p, r in historia)
    return (
        f"[lo que ya hablamos en esta sesión]\n{previas}\n\n"
        f"[la pregunta de ahora]\n{pregunta}{contrato}"
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


def cuenta(out: Log, ws: Workspace) -> str:
    b = ws.bridge
    delegado = f", {b.calls} lecturas delegadas" if b is not None and b.calls else ""
    fallas = f", fails: {[f.reason for f in out.fails]}" if out.fails else ""
    return f"{out.spent:,} tokens, {len(out.said)} turnos{delegado}{fallas}"


async def responder(
    agente: Cell,
    ws: Workspace,
    pregunta: str,
    historia: tuple[tuple[str, str], ...],
    *,
    ver: bool,
    plan: bool = False,
) -> str:
    """Una pregunta, contestada. El Log arranca limpio cada vez.

    Solo viaja la prosa de las respuestas anteriores. Si arrastrara el Log entero,
    `reads` vendría en más de cero desde el primer paso y `grounded` no volvería a
    disparar en toda la sesión: la pregunta cinco podría contestarse de memoria
    amparada en el grep de la pregunta uno.
    """
    out = await agente(Log(said=(Message(Role.USER, pedido(pregunta, historia, plan=plan)),)))
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
    pregunta = " ".join(a for a in argv if not a.startswith("-"))

    texto = corpus()
    fichas = texto.count("=== productos/")
    modelo = proveedor(argv)
    agente, ws = armar(texto, modelo, ver=ver, plan=plan, citas=citas)
    historia: tuple[tuple[str, str], ...] = ()

    print(
        f"\033[1magro\033[0m · {fichas} fichas, {len(texto) // 1000} KB "
        f"(~{len(texto) // 4000}k tokens) que el modelo consulta con código "
        f"· {modelo.model}{' · con plan' if plan else ''}{' · sin citas' if not citas else ''}"
    )

    try:
        if pregunta:
            await responder(agente, ws, pregunta, (), ver=ver, plan=plan)
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
            respuesta = await responder(agente, ws, pregunta, historia, ver=ver, plan=plan)
            historia = (*historia, (pregunta, respuesta))[-3:]
    finally:
        await modelo.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
