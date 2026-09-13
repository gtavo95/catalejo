# Pendientes

Ordenados por criticidad, no por costo. Cada uno dice qué es, por qué, y qué lo bloquea.
Lo que se mide va a `bitacora.tsv`; acá solo vive lo que todavía no se hizo.

## 1. El `Contenedor`: hecho, queda como flag, y lo que le falta

Las dos fases están. El código del modelo corre en un proceso hijo con `spawn` que se mata y
relanza si tarda, y `llm`, `rlm` y `add_step` se quedan en el padre: el hijo recibe un stub con
el mismo nombre que manda el pedido por el pipe, bloquea, y devuelve lo que el padre contestó.
Es el mismo `Bridge` con un pipe donde había un `run_coroutine_threadsafe`, y por eso el reparto
del `extra` es por `callable` y no por lo que pickle acepte. Con `recurse(..., contenedor=True)`
cada REPL del árbol es un hijo, el de la raíz y el de cada sub-agente. Se corre con
`--contenedor` en `agro.py` y en `evals.py` (los dos montajes).

Medido: lo que el modelo ve es byte a byte lo mismo (las 12 llamadas de una corrida con un
proveedor de guion, en los dos arms, sobre el corpus real), `cogollero-por-categoria` con
`--agro --contenedor` cierra ok en 7 turnos con el índice cruzado como dato, y tres `llm` desde
el hijo contra luna vuelven con su gasto contado en el `Bridge`. Cero timeouts, cero procesos
colgando. Sin fila en la bitácora: la única diferencia visible sería un `[repl]` que diga "no
terminó en 30 s", y no apareció.

Lo que falta, en orden. El padre sigue teniendo un hilo bloqueado por cada `run` en vuelo,
ahora esperando el pipe en vez del `exec`; atender el pipe desde el event loop (`add_reader`)
es lo que haría que `paralelo` y `depth` se puedan subir juntos. El hijo tiene disco y sockets:
`resource` o un sandbox de verdad son otro escalón. Y si pasa a default es una decisión de A/B
(spent y turnos a n=3), no de diseño; hoy no hay motivo medido para pagar el `spawn` por caso.

Regla de diseño que no se negocia: el cliente MCP vive en el PADRE. El hijo es el lado no
confiable y darle red reabre todo lo que el proceso cierra. Y lo que el proceso NO arregla es la
inyección; para las rutas citadas ya está `citas.py`, y para la plaga en prosa es el punto 3.

Aparte, y sin dueño todavía: luna en la wiki (sin `--agro`) se niega a escribir código. En 4 de 6
corridas de `karbo-precio` y 4 de 6 de `glifosato`, con y sin `--contenedor`, contesta "no tengo
disponible el bloque de ejecución" y `[grounding]` la empuja una vez sin efecto. La wiki con luna
nunca se había corrido (el control del 09-10 era anterior al commit de luna), así que no es una
regresión: es el preámbulo pelado de la wiki contra ese proveedor.

## 2. El grep que afloja el patrón cuando devuelve cero (rebajado, y el 3 lo confirmó)

El cero es el estado donde el modelo tiene menos información y más incentivo a rellenar, y es
exactamente donde se inventó las tres fichas. Hoy `grep` dice "0 líneas casan" y lo suelta ahí.

Gemini CLI hace auto-enrichment: cuando el resultado sale flaco, ensancha sin que se lo pidan. No
se puede copiar tal cual, porque ensanchar en silencio es la misma mentira que el tope de 50. La
regla tendría que ser: ensanchar, decir con todas las letras qué se probó, y cerrar con la frase
que evita la invención. "0 con 'zompopo', 0 con 'zompopos', 0 con 'hormiga arriera'. El catálogo
no tiene esa plaga."

Rebajado porque en un corpus con ontología, que es el caso principal, `ontologia/objetivos.md`
está adentro del texto y la cabecera por documento distingue un hit ahí de uno en una ficha. El
A/B del punto 3 lo midió: con la hoja como texto el modelo baja de `masticadores` al hijo y de ahí
a la ficha en 1 a 3 consultas, sin que nadie le dé el árbol. Vuelve si un corpus sin ontología lo
pide.

## 3. La ontología como dato: medido, queda como flag

Lo que se probó (`--ontologia`, filas `ontologia_como_dato` de la bitácora): `objetivos` en el
REPL, la tabla de `ontologia/objetivos.md` parseada con `id`, `etiqueta`, `padre`, `alias`, `nota`
y `fichas`, y la primera viñeta del CONTRATO diciendo que si el cliente no casa con nada ahí, el
catálogo no lo cubre.

Lo que salió. La versión que le daba solo el `id` y le pedía el join contra `plagas` abrió un
camino nuevo a un falso cero: en rodenticida el modelo escribió el join mal (dict contra lista de
ids), obtuvo cero, y cerró con "el catálogo contempla ratas pero ningún producto las cubre". Con la
hoja como texto ese cero no existía. Por eso `fichas` viene calculado en `agro.objetivos`, con los
hijos adentro, y el modelo no escribe joins. Con eso: 0 regresiones atribuibles, zompopo cierra en
3 turnos las tres veces (mira el vocabulario, mira el texto, dice que no) contra 5/5/13 del
baseline, y cogollero paga un turno más porque `grep('cogollo')` caía en la ficha directo. No
mueve aciertos, ni en la suite ni en "insectos masticadores en el maíz" a mano, que es el padre
que ninguna ficha nombra. La ontología ya es dato para `grep` porque está en el corpus.

Lo que lo haría default: un caso donde el texto no alcance, o sea un alias que la ficha no escribe
y que la hoja sí, o un corpus donde `ontologia/` no quepa en el texto. Hoy 65 de 69 conceptos
aparecen en las fichas con su nombre común y los 4 restantes son padres de agrupación, que el
modelo resuelve leyendo las filas. `agro.vocabulario` queda para cuando aparezca.

Lo que sigue sin tocarse: la prosa. Una respuesta a "¿qué uso para el zompopo?" va a decir
"zompopo" porque la pregunta lo dice, y lo que está mal es recomendar un producto para eso, que no
es un string. Un chequeo determinístico de nombres en prosa contra `ontologia/` no está bien
definido.

## 4. El two-phase para rescatar el `doc=`

El arm `con_doc` ganó -58% en tokens y perdió calidad: tres casos de 3/3 a 2/3, tres formas
distintas, todas sin consultar el contexto. El gate lo bloquea. Desbloquearlo pide n=6 en los DOS
arms, no solo en el nuevo, y eso es plata.

La alternativa es no re-medir lo mismo sino cambiar el diseño: Codex CLI devuelve solo nombres de
archivo y obliga a un `read` aparte, así que no hay contenido que contestar sin pedirlo. Es un arm
nuevo y arranca de cero en n. Ojo con el mecanismo: la falla fue hacer DE MENOS (4.2 turnos contra
5.7), y two-phase fuerza más llamadas, lo cual es plausible como cura y también se come parte del
ahorro.

## 5. `max_hits=50`, nunca medido

Entre 1.229 y 3.079 tokens por llamada sobre el bundle, contra una corrida entera que promedia
7.788. Es el default más caro sin A/B. Bajarlo a 20 es una palanca de una palabra y la red ya está
puesta: la cabecera dice el total de verdad, cómo pedir el resto y ahora también en qué documentos
está. El riesgo es que canjee tokens por turnos, que es justo lo que la bitácora sabe leer. Ojo con
la lectura: el piso de ruido de spent a n=3 es ±30% (tres corridas del mismo código dieron 9.5k,
12.6k y 11.3k), así que un ahorro menor que eso no se ve, y el argumento tiene que ser por turnos
y aciertos.

## Cerrado, no reabrir

- **ripgrep**: pierde el plegado de acentos, que es la palanca medida (`pulgon` da 33 líneas contra
  5 de `rg -i`, `arana roja` 9 contra 0). Vuelve a la mesa si el corpus deja de entrar dos veces en
  memoria o si el bundle vive en disco de verdad.
- **El tope por ancho de línea** que tienen opencode y Cursor: medido, p99 del bundle son 418
  caracteres y ninguna línea pasa de 2.000. Ahorraría 10%. No vale el código.
- **Un motor de regex lineal en Python**: `rure` está abandonado. La única opción mantenida era el
  paquete `regex` con `timeout=`, y se descartó porque el cuelgue ya tiene dueño en el punto 1.
