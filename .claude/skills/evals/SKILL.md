---
name: evals
description: >-
  La suite de calidad y el gate de regresión. Dónde vive un caso y por qué (regla: en el bundle,
  nunca en este repo), qué columna lleva y cuál NO, cómo se arma el subset estratificado con sus
  centinelas, el gate por capas de gratis a caro, y por qué el criterio puede mentir. Úsala al
  agregar o cambiar un caso, antes de mergear algo que toca el camino del turno, al leer un ✗, o
  al decidir cuánto eval correr. Para decidir si un cambio se embarca, `ab-testing`.
---

# Evals de catalejo

Portada de `~/code/chatbots/exp`, donde el eval mide un servicio en producción contra la wiki de
cada inquilino. Acá el corpus es un bundle local y el juez es código, así que la mitad cara
desaparece y la disciplina no.

## La regla: los casos viven en el bundle

`okf/successo-okf/evals/preguntas.tsv`, nunca en este repo.

Un caso es **curaduría**, igual que una página de producto: lo escribe quien conoce el dominio y
vale por eso. El motor es genérico y el bundle no. Si el bundle cambia de manos, los casos se van
con él y catalejo sigue sirviendo para otro corpus sin arrastrar veinte preguntas sobre
bioinsumos. Es la misma regla que en `exp`, donde los casos migraron del repo a `<wiki>/evals/`
por este motivo exacto.

Corolario que ya está implementado y no hay que romper: **el corpus EXCLUYE `evals/`**. `corpus()`
lo saltea junto con `.claude/`. Meter las respuestas adentro del contexto sería hacerse trampa al
solitario.

Lo que sí vive acá es la **bitácora** (`bitacora.tsv`), porque mide el motor y no los datos del
cliente. Casos allá, mediciones acá.

## Qué lleva un caso

Cinco columnas separadas por tabulador, la primera línea empieza con `#`:

| columna | qué lleva |
| :-- | :-- |
| `id` | kebab, estable. Es con lo que se reporta un caso que falla |
| `pregunta` | como la escribiría el cliente, no como la indexa el buscador |
| `paginas` | rutas separadas por espacio, o `ninguna` cuando la respuesta correcta es que no hay |
| `seccion` | de qué parte de la página sale el dato |
| `criterio` | qué tiene que hacer la respuesta, y sobre todo qué NO |

**El criterio no lleva el valor.** Un eval que dice "contesta Q48" es una segunda copia del dato:
el día que suba la tarifa, la página se actualiza y el eval sigue exigiendo el monto viejo. Se
asegura a qué página llega y de qué sección sale; el valor lo pone la página. Es la regla del
bundle y también protege contra el modo de falla que `exp` pagó caro: un ground truth envenenado
fabrica fallos falsos. Ahí `price_min=0` era un bug del join y el eval marcó "precio inventado"
cinco respuestas que daban el precio **correcto**.

**Un caso vale si alguien lo vio fallar.** Ninguna de las veinte es "¿qué es Royano?": eso lo pasa
cualquier recuperación y no mide nada. Los que valen son los que ya fallaron atendiendo, no los
que se inventan leyendo la página.

## Los estratos

La diversidad vive en la suite, no en repetir. Los de hoy, según el bundle:

- **el nombre local**, que es otra palabra en otro país (TROYANO es Royano, 26 páginas con tabla)
- **la ambigüedad entre dos páginas legítimas** (el hierro EDDHA son dos productos y los separa la vía)
- **los descontinuados**, que no tienen precio y por eso invitan a inventarlo
- **lo que no está en el bundle** y no hay que rellenar
- **las trampas de una página** (el pegamento de NoviTrap no está declarado y el de NoviGlue sí)
- **las reglas de empresa**, donde el borde es el caso (una factura de Q1.500 exactos paga envío)

Las cuatro trampas puras son los **centinelas**: producto descontinuado, sección vacía, producto
sin página, producto que no está en el catálogo. Los cuatro invitan a lo mismo, que es el modo de
falla que importa: **no encontrar y contestar igual**.

## Cuánto eval correr

**El default es subset estratificado a n=1, no las 20 × n=3.** El subset se arma con:

- **≥1 caso por estrato.** El estrato que el cambio NO toca se incluye igual: ese es el control.
- **+ los cuatro centinelas.** Si uno regresa, el cambio se bloquea aunque el resto gane.
- **+ los casos que el cambio ataca.**

El filtro por id ya existe, son los argumentos posicionales:

```sh
uv run evals.py                    # los 20; con un id que no existe, los lista todos
uv run evals.py <id> <id> <id>     # el subset
uv run evals.py <id>               # uno solo, con la transcripción entera
uv run evals.py --repeats=3 <id>   # ese caso tres veces, para ver si flipa
uv run evals.py --agro             # las agronómicas, contra el agente de agro.py
uv run evals.py --openai           # el mismo examen contra el otro proveedor
```

Los ids no se escriben acá ni en `evals.py`: viven en el TSV porque nombran productos de un
cliente.

**La confirmación es targeted, nunca la suite entera.** Se profundiza a n≥3 **solo los casos que
flipan o quedan en la frontera**. Repetir un caso estable tres veces es gasto muerto: lo único que
n compra es varianza intra-caso. La escalera de n y por qué n=1 es suerte están en `ab-testing`.

El reporte ya separa las cuatro cosas que hay que separar. **Estable** es el caso que acertó en
todas sus corridas vivas, **flipper** el que acertó en algunas, **caído** el que no acertó en
ninguna, y **mudo** el que no dejó una sola muestra viva. La lista de flippers es la que dice qué
profundizar a n=6; lo demás ya está decidido y repetirlo es gasto muerto.

`--repeats` va con igual y no con espacio. `--repeats 3` dejaría el `3` como id de caso, porque el
filtro por id se lleva todo argumento que no empiece con guion.

Y el resumen cierra imprimiendo la fila de `bitacora.tsv` ya armada, con el sha resuelto y
`-dirty` si el árbol no estaba limpio. `palanca`, `arm` y `nota` van en `·` porque eso lo sabe el
que corrió, no el script.

## El gate, de gratis a caro

Heredado de `exp`, donde las dos primeras capas son deterministas y cazan la rotura silenciosa
antes de gastar un centavo. Acá se traduce limpio:

| capa | qué | costo |
| :-: | :-- | :-- |
| **0** | `uv run mypy`, que revisa puertos, firmas y exhaustividad | $0 |
| **1** | `uv run pytest -q`, el álgebra, el REPL y los adaptadores contra transporte falso | $0 |
| **2** | `uv run evals.py <subset>`, calidad contra el bundle, con llamadas reales | paga |

Las capas 0 y 1 corren sin red: los tests de adaptador usan `httpx.MockTransport` y los del
álgebra usan `Stub`. Si la capa 2 falla, primero descartá que sea la capa 0 o 1 disfrazada.

**El gate es cero regresiones.** Para cada ✗ la pregunta es de regresión: ¿el baseline lo pasaba?
Si sí, bloquea. Si ya fallaba, es pre-existente, no bloquea este cambio, pero se anota.

Un ✗ que viene de un turno vacío del proveedor **no es una regresión de calidad**: es una muestra
perdida, y `resumir()` ya la saca del denominador y la reporta aparte. Se reconoce por
`Fail.who == "model"`. Los fallos de `loop` (se quedó sin pasos o sin presupuesto) y de `grounding`
(contestó sin mirar) sí cuentan, porque son resultados y no accidentes de la API.

## El criterio puede mentir

`evals.py` no tiene juez LLM, y eso es una ventaja y un límite.

La ventaja: `acierta()` es substring y `inventada()` es pertenencia a un conjunto. Exactas,
gratis, sin calibración. La mitad cara del eval de `exp` acá cuesta cero, así que profundizar un
flipper a n=6 es solo el precio de las corridas del agente.

El límite, y hay que decirlo con todas las letras: **la máquina califica la CITA, no la
respuesta.** `acierta()` mira si la página esperada aparece en el texto. El `criterio`, que es
donde vive "no inventar una dosis", lo califica un humano leyendo la salida, y por eso `main()`
imprime el criterio y la respuesta de cada caso que falla. Un ✗ verde por la cita y malo por el
criterio existe y la máquina no lo ve.

`inventada()` es la excepción y vale entender por qué: el modelo SÍ leyó, así que `grounded` lo
deja pasar, y aun así la ruta que declara puede no existir. Una cita es una afirmación sobre un
conjunto conocido, y eso se verifica con código. Es el equivalente determinista del caso
adversarial de `exp`, donde metían respuestas fluidas pero falsas para ver si el juez las dejaba
pasar.

**Antes de creerle a un ✗, leé la respuesta cruda, no el puntaje.** Si el criterio contradice el
contrato que acabás de acordar, el bug vive en el criterio. Y al corregirlo, el chequeo de
honestidad es que **el baseline siga fallando** y los centinelas no se muevan: así probás que
endureciste la vara y no que la amañaste a favor de tu tratamiento.

## Las dos suites

`preguntas.tsv` son 20 preguntas comerciales que contesta el agente de la wiki. `agro.tsv` son 9
agronómicas que contesta el asesor de `agro.py`, con su corpus de dos carpetas, su contrato y su
índice de fichas en el REPL. Son dos programas, así que medirlos con la misma suite diría poco de
los dos: el ahorro de 108k a 10.4k tokens se midió sobre `agro.py`, y confirmarlo pide correr
`agro.py`.

El hecho que ordena media suite agronómica: **la dosis por manzana vive solo en el texto de
`# Ficha`.** El bloque `# Agronomía` en JSON modela únicamente `basis: "ha"`, así que el índice
`productos` que el REPL le da al modelo no la tiene y hay que ir al texto con grep. Dos casos
apuntan ahí, uno donde el valor por manzana está escrito y otro donde no.

La primera corrida ya pagó la suite. El centinela de la plaga ausente encontró que, preguntando por
una plaga que ni las fichas ni la ontología nombran, el asesor **se inventa tres fichas que no
existen** en vez de decir que no la cubre. Lo agarró `inventada()`, sin juez y sin humano.

## Lo que falta
- **El canal `cached`**, y solo si hace falta. Se midió: en este stack Gemini nunca informa
  `cachedContentTokenCount`, porque el prompt de la raíz no llega al piso de la caché implícita. El
  detalle está en `ab-testing`. Vuelve a la lista el día que el preámbulo engorde.
- **`prompt_tokens_details.cached_tokens` en OpenAI**, que es lo mismo del otro lado y sigue sin
  medirse porque el adaptador nunca habló con la API real.
