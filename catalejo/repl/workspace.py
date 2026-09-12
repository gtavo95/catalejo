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

`json` está en el namespace como módulo, así que `json.__builtins__` es un camino
derecho a `open`. No cambia la conclusión, la subraya: esto no aguanta código que
se lo proponga, y por eso lo que se le da al modelo se elige por lo que hace, no
por lo difícil que sea escaparse. `json` entra porque parsear no tiene efectos.

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
import json
import re
import threading
import unicodedata
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

from .environment import Output

T = TypeVar("T")

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
    "type",
    "zip",
    "Exception",
    "IndexError",
    "KeyError",
    "TypeError",
    "ValueError",
)

MAX_HITS = 50

CABECERA = re.compile(r"^=== (.+) ===$")

HERRAMIENTAS = (
    "Es Python real con los builtins recortados: no hay `import`, `open` ni `eval`. "
    "Tienes `grep(texto, patron)` para expresiones regulares: devuelve las líneas que "
    "casan, numeradas, y la primera línea dice el total y, si el texto trae documentos, "
    "en cuáles y cuántas en cada uno. Muestra hasta 50; si hay más, "
    "sube el tope con `grep(texto, patron, max_hits=500)` o afina el patrón. No distingue "
    "mayúsculas ni acentos, así que `pulgon` encuentra `Pulgón` y `arana` encuentra "
    "`araña`; cuando el caso importe, `grep(texto, patron, exacto=True)`. Tienes `json` "
    "sin importarlo, para `json.loads` sobre un bloque que venga del contexto. Si el texto "
    "trae documentos separados por una línea `=== ruta ===`, cada resultado sale como "
    "`ruta:linea: contenido`, así que no hace falta que busques a qué archivo pertenece. "
    "Ese prefijo se agrega al imprimir y NO es parte de lo que se busca: un patrón que lo "
    "incluya, como `productos/x\\.md:.*dosis`, no casa nunca y te devuelve cero. Para mirar un "
    "solo documento, `grep(texto, patron, doc='x')`, que compara contra la ruta. La línea "
    "`=== ruta ===` no es contenido: si el patrón casa con ella, la primera línea te nombra "
    "el documento sin sumarlo al total. "
    "El resto de Python funciona normal: rebanar, `len`, comprensiones, `sorted`."
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
        f"`json` ya está en el namespace, usalo sin importarlo. Para expresiones "
        f"regulares usa grep(texto, patron). Para contar, un dict: "
        f"`c = {{}}; c[k] = c.get(k, 0) + 1`. El resto de Python funciona normal."
    )


def sin_acento(s: str) -> str:
    """La misma cadena sin marcas diacríticas.

    Descompone en NFD y tira las combinantes, así que también pliega la eñe:
    `araña` y `arana` quedan iguales. Es agresivo a propósito, porque el modelo
    escribe el término tal como lo dice el cliente y el cliente no pone tildes.
    """
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


@dataclass(slots=True)
class _Plegado:
    """Un texto ya partido en líneas, y esas líneas ya sin acentos."""

    texto: str
    lineas: list[str] = field(default_factory=list)
    campo: list[str] | None = None


MAX_PLEGADOS = 8

_PLEGADOS: dict[int, _Plegado] = {}
_CANDADO = threading.Lock()


def _plegar(texto: str, *, plegar: bool) -> tuple[list[str], list[str]]:
    """Las líneas del texto y su copia buscable, calculadas una sola vez por texto.

    El costo de `grep` no estaba en la regex, estaba acá. Sobre el corpus de 7,2 MB
    de `demo.py`, una llamada tardaba 637 ms y 574 de esos eran plegar los acentos
    del corpus ENTERO, de nuevo, en cada llamada. Con el plegado guardado la misma
    búsqueda tarda 50 ms. El modelo no ve ninguna diferencia: mismos hits, misma
    cabecera. Por eso esto no lleva fila en la bitácora, que mide tokens y aciertos
    y no tiene columna de milisegundos.

    La clave es `id(texto)` y el texto se guarda al lado, que es lo que hace segura
    la clave: mientras el diccionario lo referencia, ese objeto no se libera y su
    `id` no se puede reusar para otro. La comparación real es `is`, no `==`, porque
    comparar dos corpus de 7 MB por contenido cuesta lo mismo que plegarlos.

    Ocho entradas porque ocho es `PARALELO`: un `fanout` entero puede tener su
    plegado sin pisarse. Cada hijo recibe un pedazo, así que lo que el caché puede
    llegar a retener está acotado por el texto que ya estás teniendo en memoria,
    duplicado. Cuando se llena sale el más viejo.

    `exacto=True` no pliega nada y por eso `campo` se calcula recién cuando alguien
    lo pide. Sobre el corpus de código Go, donde `Plan` y `plan` son cosas
    distintas, plegar sería trabajo tirado.
    """
    guardado = _PLEGADOS.get(id(texto))
    if guardado is None or guardado.texto is not texto:
        guardado = _Plegado(texto, texto.splitlines())
        with _CANDADO:
            if len(_PLEGADOS) >= MAX_PLEGADOS:
                _PLEGADOS.pop(next(iter(_PLEGADOS)))
            _PLEGADOS[id(texto)] = guardado
    if not plegar:
        return guardado.lineas, guardado.lineas
    if guardado.campo is None:
        guardado.campo = [sin_acento(linea) for linea in guardado.lineas]
    return guardado.lineas, guardado.campo


def grep(
    texto: str,
    patron: str,
    max_hits: int = MAX_HITS,
    *,
    exacto: bool = False,
    doc: str = "",
) -> str:
    r"""Las líneas que casan con el patrón, numeradas, con el total adelante.

    `texto` es donde se busca, línea por línea. `patron` es una regex de `re`, no
    un literal: los puntos y los paréntesis hay que escaparlos. `max_hits` es
    cuántas líneas se muestran como mucho; el total se cuenta igual y se dice
    siempre. `exacto` y `doc` van más abajo, cada uno con su motivo.

    Pliega mayúsculas y acentos, y ese default no es comodidad. Sobre el catálogo
    agronómico, `mosca blanca` en minúscula devolvía 6 de las 23 líneas que hay, y
    `arana roja` sin la eñe devolvía cero con seis en el texto. El modelo no tiene
    forma de enterarse: recibe un número que parece el total. Es el mismo error que
    el tope silencioso, que ya está resuelto dos párrafos más abajo, entrando por
    otra puerta.

    El patrón NO se pasa a minúsculas, se le sacan los acentos y las mayúsculas van
    por la bandera. En una regex `\S`, `\D`, `\W` y `\B` significan lo contrario
    que sus versiones minúsculas, así que bajar el patrón entero convierte "no
    espacio" en "espacio" y rompe en silencio cualquier búsqueda con clases.

    `exacto=True` para cuando el caso es el dato: sobre el corpus de código Go de
    `demo.py v1`, `Plan` y `plan` son cosas distintas.

    `doc="viventem"` acota la búsqueda a los documentos cuya ruta contenga eso, y
    existe por un cero falso medido. El resultado sale prefijado con `ruta:`, así
    que el modelo deduce lo razonable y escribe `productos/viventem\.md:.*DOSIS`
    para mirar una sola ficha. Ese prefijo se arma al imprimir y no está en el
    texto sobre el que se busca, así que el patrón no casa nunca. En la suite
    agronómica contra luna pasó cinco veces en una sola corrida, todas en el caso
    que pregunta si un dato ESTÁ, que es donde un cero falso se vuelve una
    respuesta segura y equivocada. Ese caso se llevó 223 mil de los 503 mil
    tokens de la suite.

    Un `doc` que no casa con ningún documento se dice con todas las letras, por lo
    mismo que se dice el cero: "no hay líneas en esa ficha" y "esa ficha no existe"
    son dos respuestas distintas y el modelo tiene que poder separarlas.

    Cuando el texto es una concatenación de documentos con la línea `=== ruta ===`
    adelante, cada hit sale con su documento. Es la diferencia entre encontrar y
    saber qué encontraste: sobre el catálogo agronómico, un hit cae a una mediana de
    119 líneas de su cabecera y a 230 en el peor eje, así que averiguar de qué
    producto era la línea 5157 costaba imprimir una ventana y caminar para atrás.
    Una corrida se fue a 108 mil tokens haciendo justamente eso. El dato ya estaba en
    el texto y no viajaba con el resultado; ahora viaja.

    La primera línea habla en documentos, porque en un corpus de fichas la unidad es
    la ficha y no la línea. Medido sobre el catálogo: `pulgon` son 24 líneas en 3
    documentos (segador 12, objetivos 7, novitrap 5) y `cogollero` 6 en 2. Esa lista
    es el esqueleto de la respuesta, qué productos y aparte qué dice el vocabulario,
    y antes el modelo la reconstruía leyendo 24 prefijos. Va ordenada por cantidad
    de líneas y con el mismo tope que las líneas, anunciado igual.

    La cabecera `=== ruta ===` no es contenido y no cuenta. Como contiene la ruta, un
    patrón que nombra un producto o una carpeta casaba con ella: `productos` daba
    120 líneas de las que 39 eran cabeceras, y `viventem` 13 con 1. En un agente
    cuyo argumento es contar bien, eso es un total inflado. Pero saltarla en
    silencio convierte `grep(ctx, '===')`, que es la forma natural de listar los
    documentos de un texto sin índice, en un cero falso, y `bio-bpbs` daría una
    línea suelta del índice sin decir que la ficha existe con ese nombre. Así que
    cuando el patrón casa con una cabecera, la primera línea nombra el documento y
    no lo suma. Lo que se compra es el número correcto y el resumen por documento;
    los tokens casi no cambian, porque bajo el tope las cabeceras eran líneas
    baratas y su lugar lo ocupan líneas de contenido más largas.

    Que pliega va dicho en `HERRAMIENTAS` y no en la cabecera de cada resultado,
    porque es una propiedad fija de la herramienta y no un hecho de esta corrida.
    La cabecera dice lo que cambia entre llamada y llamada, que es cuántas hay y
    cuántas se muestran.

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
    lineas, campo = _plegar(texto, plegar=not exacto)
    rx = re.compile(patron) if exacto else re.compile(sin_acento(patron), re.IGNORECASE)
    filtro = sin_acento(doc).lower()
    hits: list[str] = []
    mirados: list[str] = []
    por_doc: dict[str, int] = {}
    cabeceras: list[str] = []
    actual = ""
    dentro = not filtro
    for i, (original, buscable) in enumerate(zip(lineas, campo), 1):
        cabeza = CABECERA.match(original)
        if cabeza:
            actual = cabeza.group(1)
            dentro = not filtro or filtro in sin_acento(actual).lower()
            if dentro and filtro:
                mirados.append(actual)
            if dentro and rx.search(buscable):
                cabeceras.append(actual)
            continue
        if dentro and rx.search(buscable):
            hits.append(f"{actual}:{i}: {original}" if actual else f"{i}: {original}")
            if actual:
                por_doc[actual] = por_doc.get(actual, 0) + 1
    if filtro and not mirados:
        return (
            f"ningún documento casa con {doc!r}, así que no se buscó nada. `doc` se compara "
            f"contra la ruta de la línea `=== ruta ===`, no contra el contenido."
        )
    ranking = sorted(por_doc.items(), key=lambda par: -par[1])
    lista = _lista([f"{ruta} ({n})" for ruta, n in ranking], max_hits)
    if filtro and len(mirados) == 1:
        ambito = f" en {mirados[0]}"
    elif filtro and not ranking:
        ambito = f" en los {len(mirados)} documentos que casan con {doc!r}"
    elif filtro and len(ranking) == len(mirados):
        ambito = f" en los {len(mirados)} documentos que casan con {doc!r}: {lista}"
    elif filtro:
        ambito = f" en {len(ranking)} de los {len(mirados)} documentos que casan con {doc!r}: {lista}"
    elif len(ranking) == 1:
        ambito = f" en {ranking[0][0]}"
    elif ranking:
        ambito = f" en {len(ranking)} documentos: {lista}"
    else:
        ambito = ""
    casan = "casa" if len(hits) == 1 else "casan"
    linea = "línea" if len(hits) == 1 else "líneas"
    cabecera = f"{len(hits)} {linea} {casan} con {patron!r}{ambito}."
    if len(hits) > max_hits:
        resto = "subí max_hits, afina el patrón o acota con doc=" if por_doc else "subí max_hits o afina el patrón"
        cabecera = (
            f"{len(hits)} líneas casan con {patron!r}{ambito}; estas son las primeras "
            f"{max_hits}. Para el resto {resto}."
        )
        hits = hits[:max_hits]
    sueltas = [ruta for ruta in cabeceras if ruta not in por_doc]
    if sueltas:
        cuantos = f"{len(sueltas)} documento" + ("s" if len(sueltas) != 1 else "")
        cabecera += (
            f" La cabecera `=== ruta ===` de {cuantos}{' más' if por_doc else ''} casa con el "
            f"patrón y no cuenta como contenido: {_lista(sueltas, max_hits)}."
        )
    return "\n".join([cabecera, *hits])


def _lista(items: list[str], tope: int) -> str:
    """Una lista en una línea, y si pasa del tope dice cuántas quedaron afuera."""
    if len(items) <= tope:
        return ", ".join(items)
    return ", ".join(items[:tope]) + f" y {len(items) - tope} más"


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
                "print": self._print,
                "__import__": _sin_import,
            },
            "grep": grep,
            "json": json,
            **(extra or {}),
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
                return await asyncio.to_thread(self.correr, code)
            self.bridge.loop = asyncio.get_running_loop()
            antes = self.bridge.spent
            out = await asyncio.to_thread(self.correr, code)
            return replace(out, spent=self.bridge.spent - antes)

    def correr(self, code: str) -> Output:
        """El `exec` pelado, síncrono, sin hilo ni lock: solo correr y capturar.

        Es público porque es la mitad del Workspace que se lleva el contenedor.
        `run` es la otra mitad, la de dónde corre: el hilo, el lock, el Bridge.
        Adentro de un proceso hijo no hay event loop que proteger, así que el
        hijo llama esto directo.
        """
        self._salida.clear()
        err = ""
        try:
            exec(code, self._globals)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        return Output(stdout="".join(self._salida), err=err)
