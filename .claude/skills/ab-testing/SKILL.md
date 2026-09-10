---
name: ab-testing
description: >-
  Cómo decidir si un cambio de catalejo se embarca. El A/B de una sola palanca, la escalera de n
  (n=1 es suerte, n=3 primera lectura, n=6 para confirmar un default), qué columna del Log mirar,
  los dos libros contables y los priors ya pagados. Úsala antes de cambiar el preámbulo, un default
  de `grep`, un builtin, `thinking`, el proveedor, `keep_recent`, `max_steps` o un budget; al leer
  el número de una corrida; o cuando sospeches que el A/B es la vara equivocada. Para la suite de
  calidad y el gate de regresión, `evals`.
---

# A/B en catalejo

Portada de `~/code/chatbots/exp`, donde la disciplina se pagó con dinero real sobre un servicio
en producción. Acá el repo es una biblioteca, así que el mecanismo cambia y el juicio no.

Esta skill es el **juicio**. La suite de calidad y el gate viven en la skill `evals`. La bitácora
es `bitacora.tsv`, en la raíz.

## Qué es una palanca acá

Cualquier cosa que cambie el turno sin cambiar la pregunta:

- una línea del preámbulo (`Handle`, `HERRAMIENTAS`, `note(depth)`, la `nota` de `recurse`)
- un default de `grep` (el plegado de caso y acentos, `max_hits`, el prefijo `ruta:linea:`)
- un builtin nuevo en `_SEGUROS` o en `extra`
- `thinking`, el proveedor, `temperature`, `retries`
- `keep_recent`, `max_steps`, `budget`, `depth`
- el `CONTRATO` de `agro.py` o el `CITA` de `evals.py`

Regla de oro: **mismo código, una sola palanca, mismas preguntas.** Si movés dos cosas, el Δ no es
de ninguna.

## Cuándo NO correr un A/B

Cuando podés contar, contá. El plegado de caso y acentos de `grep` no necesitó A/B: "`mosca
blanca` traía 6 de las 23 líneas que hay" es un hecho sobre el corpus, verificable con código,
y su lugar es el docstring del test que lo fija, no la bitácora. El A/B es para cuando hay que
muestrear porque el resultado varía entre corridas idénticas.

Tampoco lo corras para un refactor que no toca el camino del turno, ni para algo cuyo efecto un
test ya deja fijo.

## La escalera de n

**n=1 es suerte.** No es precaución genérica, es el régimen medido de este stack. Doce pedidos
idénticos, el mismo body JSON, contra `gemini-3.8-flash` con `thinking="low"`:

```
sin índice                                  6/12 turnos buenos
el índice nombrado en el turno del usuario  6/12
el índice nombrado en el preámbulo          0/12
```

Seis de doce vuelven con una sola parte, firma de razonamiento y `text` vacío. Es el mismo
pedido: la varianza es del proveedor. Y ese seis **no es estable entre rondas**, otra tarde la
misma línea base dio 12/12, así que el número absoluto es una foto y lo único afirmable es la
comparación adentro de una misma ronda, con las variantes corridas seguidas.

Los dos ecos que `exp` ya pagó:

- `relevance_filter`: **−84% a n=1**, se desinfló a **−3,8% a n≥3**.
- `session_recap`: **−32,6% a n=3**, quedó en **−11,6% a n=6**.

| n | para qué | qué compra |
| :-: | :-- | :-- |
| **1** | ver si el mecanismo corre | plomería, nada decidible |
| **3** | primera lectura, decide si vale seguir mirando | descarta el ruido grueso |
| **6** | confirmar un default que vas a embarcar | desinfla el espejismo de muestra chica |

n solo compra varianza intra-pregunta. Repetir un caso estable seis veces es gasto muerto; lo que
n desempata es un caso que **flipa**. Para el eval de calidad la diversidad viene de los estratos
de la suite, no de repetir: eso está en la skill `evals`.

Un piso útil: adentro de una ronda, 0/12 contra 6/12 no es ruido. Si la tasa real fuera la mitad,
doce fallos seguidos son uno en 4096.

## Polaridad: el error fácil

El baseline es el arm que **restaura el comportamiento previo**. Si la palanca es nueva, el
baseline es el arm sin ella; si ya está puesta, el baseline es el que la apaga.

Acá el error es sutil porque no hay flags: si vas a medir el prefijo `ruta:linea:` de `grep`, el
baseline es una copia de `grep` **sin** ese prefijo, no el commit anterior, que además trae el
plegado de acentos y arruina la atribución. En `exp` esto hizo que los dos arms midieran lo mismo
y el Δ saliera ≈0 espurio.

## Qué mirar

El `Log` ya trae los seams. Hay que leer más de una columna:

- **`out.spent`** es el titular, el total facturado del árbol entero.
- **`len(out.said)`** son los turnos, que es de dónde viene el gasto. 5 turnos contra 9 no se
  comparan por el total sin mirar esto.
- **`bridge.calls`** son las lecturas delegadas. Es donde el costo se esconde: una palanca que
  achica el prompt de la raíz puede empujar el trabajo a `llm()` y salir plana en el total. Antes
  de festejar un ahorro en la raíz, fijate si migró a la delegación.
- **`out.reads`** dice si consultó el corpus. Cero con respuesta en prosa es el modo de falla que
  ataca `grounded`.
- **`out.fails`** dice si se cortó. Una corrida que murió por un turno vacío es una **muestra
  perdida**, no un fallo de calidad del tratamiento. Contarla como fallo sesga el A/B.

**Confound que este repo todavía no puede ver: la caché.** `gemini.py` lee solo `totalTokenCount`.
En `exp` el −44% del worker GLM era **enteramente** su caché de prompt (70% de hit) y sin caché el
A/B salía plano. Hasta que el adaptador exponga `cachedContentTokenCount`, una baja de costo entre
corridas separadas en el tiempo no es atribuible.

## El veredicto

- **Embarcar como default.** Baja el costo **Y** cero regresiones **Y** aguanta n≥3, idealmente
  confirmado a n≥6.
- **Dejarlo disponible, no por default.** El mecanismo no es nulo pero es net-negativo o depende
  del régimen. En `exp`: `catalog_list` (+98%), `skill_analista` (+74% y regresó un caso).
- **Sacar el código.** Mecanismo nulo o foot-gun puro. En `exp`: `plan_first`, que no movió nada
  y encima infló el juez +117%.

**El gate es cero regresiones**, y una sola bloquea el default por más que el costo baje. Cómo se
mide está en `evals`.

## Dos libros contables: cuándo el A/B es la vara equivocada

El A/B contesta UNA pregunta: **«¿esto es mejor que HOY?»**. Es la correcta para una palanca de
**explotación**, que compite contra el trabajo actual. Tiene dos puntos ciegos.

**Colapsa un vector en un escalar.** "Gano algo, pierdo algo" es literal: el total esconde qué
intercambiaste. La cura no es un A/B mejor, es dejar de sumar. Reportá el cambio como un vector de
ejes con nombre (costo, calidad, turnos, robustez de cola, cobertura, mantenimiento) y **separá
medición de decisión**: el vector se mide, el "gana o no" lo decide una política de tasa de cambio
que escribís aparte. Una palanca solo sale si está **dominada**, peor en todo eje. Si está en la
frontera de Pareto, elegir es una política, no una intuición.

**Tarifa en ~cero el valor de opción.** Una palanca cuyo pago es "me deja cambiar de proveedor" o
"me cubre de un outage" SIEMPRE pierde contra el baseline sano de hoy: agrega costo y no rinde
mientras nada falle. Medirla ahí es medir un seguro en un día soleado. Para esas, declará el
estado futuro que la palanca compra, inyectalo, y medí ahí.

El ejemplo vivo de este repo es el **adaptador de OpenAI**. A/B-earlo contra Gemini sano es un
error de categoría: es un seguro contra el turno vacío y contra un cambio de precio, y su vara es
la disponibilidad con un proveedor caído, no el Δ de tokens con los dos sanos. Lo mismo `retries=4`:
no compra calidad, compra que la corrida termine.

Regla dura: **clasificá la palanca antes de elegir la vara.** Confundir explotación con inversión
hace que la decisión salga mal con números impecables.

## Priors caros, ya pagados

1. **Curá el dato y la plomería, no el prompt.** El prior con más ecos en `exp` (`fanout_nudge`
   +27%, `plan_first` retirado, `skill_analista` regresó un caso) y esta sesión lo repitió exacto.
   La pregunta cara pasó de 108.045 a 10.395 tokens, y **no** fue por el índice de productos que le
   pasamos al modelo: fue porque `grep` empezó a devolver de qué documento es cada línea. El dato
   ya estaba en el texto y no viajaba con el resultado. El índice, que es afordancia orientada al
   modelo, no movió el número y encima trajo el prior 3.
2. **El régimen es estocástico y flipa por corrida.** Dos fuentes: el turno vacío del proveedor
   (6/12) y qué patrón se le ocurre escribir al modelo. `temperature` va en 1.0 porque bajarla en
   los Gemini 3 los empeora, así que la varianza no se apaga, se muestrea.
3. **El preámbulo no es un lugar neutro donde poner texto.** El mismo párrafo da 6/12 en el turno
   del usuario y 0/12 en el preámbulo. No sé el mecanismo. Antes de agregar texto al preámbulo,
   medilo.
4. **La R suele estar dormida.** `agro.py` y `evals.py` corren con `depth=0`, así que `rlm` ni
   existe en el namespace. Si `bridge.calls` es cero, la palanca que ibas a medir sobre la
   delegación no se ejerció y el Δ que veas es de otra cosa.
5. **Un puerto que no alcanza se paga listando implementaciones.** `Model` pide solo `complete`,
   que es lo que el motor usa; `Provider` agrega `model` y `aclose`, que es lo que la aplicación
   necesita. Antes de eso `agro.py` se anotaba `Gemini | OpenAI`, o sea nombraba a los dos
   adaptadores concretos para pedirles algo que ningún puerto declaraba. `tests/test_puertos.py`
   lo fija con mypy y no con una afirmación.

## La bitácora

`bitacora.tsv` en la raíz, append-only, commiteada. Vive acá y no en el bundle porque mide el
**motor**, no los datos del cliente: la regla es que los casos viven en `okf/successo-okf/evals/`
y las mediciones acá. El esquema está en la cabecera del archivo.

Tres capas, heredadas de `exp`: **capturas** (efímeras) → **bitácora** (los números, durables) →
**memoria** (la conclusión destilada). Si no escribís la fila, el experimento se evaporó en stdout.

## Respaldo externo, en corto

**RQGM** (Iacob et al., 2026, arXiv 2606.26294): no muevas el evaluador a mitad de una medición, y
reemplazarlo exige evidencia estadística sobre un ground truth fijo con empate a favor del titular.
Acá el evaluador es `acierta()`: si lo tocás, las corridas viejas dejan de ser comparables.

**Inference scaling** (AI Security Institute, 2026, arXiv 2606.17930): medir a un solo budget
engaña en las dos direcciones, hay que reportar una **curva y no un punto**. Ellos buscan el techo
de capacidad; acá interesa el piso, o sea el `budget` y el `max_steps` más chicos que no pierden
calidad. Y su hallazgo incómodo: las ganancias vienen de alcance y **fiabilidad**, casi nunca de
eficiencia de tokens. O sea que `pass^k` (cuántas de k corridas resolvieron) es un número de
primera clase, no un "flipeó o no".
