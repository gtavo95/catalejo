# Pendientes

Ordenados por criticidad, no por costo. Cada uno dice qué es, por qué, y qué lo bloquea.
Lo que se mide va a `bitacora.tsv`; acá solo vive lo que todavía no se hizo.

## 1. Luna no escribe el bloque en el turno 1, y la línea que lo arregla cuesta +40%

La falla. Luna cierra el primer turno en prosa: "Voy a consultar la wiki para encontrar el
precio", `finish_reason=stop`, 4 tokens de razonamiento, ningún bloque. Se comporta como un
modelo de tool-calling que espera emitir una llamada y no tiene ninguna declarada. Medido con 8
pedidos idénticos al primer turno, JSON crudo (filas `turno_es_bloque`): en la wiki 0 de 8
escriben código, en agro 5 de 8. En la suite agronómica son 8 a 11 `[grounding]` de cada 27
corridas, y `grounded` lo rescata; en la wiki no hay rescate y hoy no anda con luna.

Lo que no lo mueve: `reasoning_effort=medium` (0/8). Un sufijo en el turno del usuario lo
empeora, ahí dice con todas las letras "no tengo la herramienta de ejecución".

Lo que lo mueve: la línea del preámbulo. "Para consultarlo, escribe UN bloque de código Python
cercado" pasa a "tu turno entero es UN bloque de código Python cercado y nada más", y el primer
turno va de 0/8 a 8/8 en la wiki y de 5/8 a 7/8 en agro. De las frases probadas es la única
que trabaja: "No hay otra herramienta: el bloque ES la consulta" 0/8, "en el siguiente turno"
2/8, "sin prosa antes: el turno de consulta es el bloque" 4/8.

El costo, en la suite a n=3 con los dos arms seguidos: `[grounding]` 11 a 0, 9/9 en los dos,
pero +40% tokens (13.4k a 18.8k), 6.3 a 6.9 turnos, 60 a 80 consultas. No es un flake: tres
casos se van de 2 a 3× en las tres vueltas y dos bajan a la mitad. La hipótesis es que "y nada
más" choca con el contrato de terminación (prosa = terminé) y el modelo explora de más antes de
animarse a la prosa. Sin transcripciones de la suite no se confirma.

Qué sigue. Primero guardar las transcripciones de una corrida de la suite (evals solo las
imprime con un caso y n=1) y mirar en cogollero o espartano-mz-trips qué hace el modelo con los
turnos extra. Después, una frase que fije la forma del turno de consulta sin pisar la
terminación, o partir la línea en dos: la forma del bloque en el preámbulo y "prosa cuando
terminaste" reforzada. Cada frase se prueba primero con los 8 pedidos crudos, que cuesta nada,
y la suite recién con la que pase 8/8 en la wiki. Hasta entonces la línea está en el árbol sin
commitear, y en la wiki es la diferencia entre andar y no andar.

## 2. El `Contenedor`: hecho, queda como flag, y lo que le falta

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
inyección; para las rutas citadas ya está `citas.py`, y para la plaga en prosa es el punto 4.

## 3. El grep que afloja el patrón cuando devuelve cero (rebajado, y el 4 lo confirmó)

El cero es el estado donde el modelo tiene menos información y más incentivo a rellenar, y es
exactamente donde se inventó las tres fichas. Hoy `grep` dice "0 líneas casan" y lo suelta ahí.

Gemini CLI hace auto-enrichment: cuando el resultado sale flaco, ensancha sin que se lo pidan. No
se puede copiar tal cual, porque ensanchar en silencio es la misma mentira que el tope de 50. La
regla tendría que ser: ensanchar, decir con todas las letras qué se probó, y cerrar con la frase
que evita la invención. "0 con 'zompopo', 0 con 'zompopos', 0 con 'hormiga arriera'. El catálogo
no tiene esa plaga."

Rebajado porque en un corpus con ontología, que es el caso principal, `ontologia/objetivos.md`
está adentro del texto y la cabecera por documento distingue un hit ahí de uno en una ficha. El
A/B del punto 4 lo midió: con la hoja como texto el modelo baja de `masticadores` al hijo y de ahí
a la ficha en 1 a 3 consultas, sin que nadie le dé el árbol. Vuelve si un corpus sin ontología lo
pide.

## 4. La ontología como dato: medido, queda como flag

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

## 5. Grep en dos pasos: default desde el 09-13, `--un-paso` es el arm de antes

Lo que hay. `grep` sin `doc=` devuelve solo la primera línea, la que dice en qué documentos y
cuántas líneas en cada uno, y `read(texto, 'parte de la ruta')` trae el documento entero sin
numerar, sin tope, con el rango de líneas adelante. Con `doc=` grep sigue mostrando las líneas.
Es lo que hace la búsqueda de Codex y lo que hace una persona con una carpeta: buscar, elegir
qué abrir, leer. Es el default del `Workspace`, baja a los sub-agentes y cruza al `Contenedor`;
`dos_pasos=False` (`--un-paso` en `agro.py` y en los dos montajes de `evals.py`) restaura el
`grep` de antes, sin `read`, para volver a medir. Lo único del prompt que cambia es la nota de
herramientas: `HERRAMIENTAS` es la de dos pasos y `HERRAMIENTAS_UN_PASO` la vieja.

Lo que salió (filas `dos_pasos` de la bitácora, n=3, los dos arms seguidos): 8/9 estables en los
dos, 26/27 vivas contra 25/27, el mismo caso flipa en los dos (viventem, una corrida de 3k tokens
sin consultar), +4% tokens que es ruido, turnos 4.6 a 5.1, consultas 42 a 48. El modelo lee de
verdad: 15 `read` en 27 corridas. Por caso no es parejo, cogollero y espartano-mz-trips pagan la
ficha entera (+20 a 40%) y espartano-agua-ph y zompopo bajan a la mitad porque una lectura
reemplaza varias consultas de 50 líneas. Y el hallazgo colateral: en el baseline el modelo casi no
usa `grep`, 36 de 42 consultas empiezan por `productos`, así que la palanca se ejerció en la
minoría de consultas que van al texto.

Por qué es default con un empate: lo que `acierta()` no mide. En el humo de cogollero la respuesta
con `read` trajo la equivalencia por manzana y el bloque de seguridad enteros, que salen de leer la
ficha y no de una línea, y esa completitud es lo que el que atiende necesita. Decisión del 09-13,
no del A/B: en los números no gana ni pierde. Y `read` no tiene tope a propósito: un tope que corta
la ficha en la línea 400 devuelve el problema que esto vino a sacar, un dato sin su contexto. En
este corpus ninguna ficha pasa de 286 líneas; un corpus de documentos de miles de líneas paga esos
tokens o usa `grep(doc=)`.

Lo que falta: la vara de completitud, para que el próximo A/B sobre esto no dependa de leer
transcripciones a mano. La explicación con el diagrama está en
https://claude.ai/code/artifact/66e9afbe-ba24-4442-a0e7-820cf591cc71.

## 6. `max_hits=50`, nunca medido

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
  paquete `regex` con `timeout=`, y se descartó porque el cuelgue ya tiene dueño en el punto 2.
