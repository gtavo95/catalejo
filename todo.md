# Pendientes

Ordenados por criticidad, no por costo. Cada uno dice qué es, por qué, y qué lo bloquea.
Lo que se mide va a `bitacora.tsv`; acá solo vive lo que todavía no se hizo.

## 1. El `Contenedor`, fase 2: `llm` y `rlm` cruzan el pipe

La fase 1 está hecha: `catalejo/repl/contenedor.py` corre el `Workspace` en un proceso hijo con
`spawn`, mata y relanza si tarda (medido: `(a+)+$` muere a los 2 s justos y el `run` siguiente
anda), y las movidas del plan vuelven por el pipe al `Verbos` del padre. Lo que falta para que
`agro.py` corra adentro: `llm` y `rlm` son closures sobre `bridge.loop` y no cruzan. La forma es la
misma del `Bridge`, con un pipe donde hoy hay un `run_coroutine_threadsafe`: el hijo manda un pedido
y bloquea, el padre lo resuelve en su loop y contesta. `agro.py` usa `recurse` con `depth=0`, que
igual instala `llm`, así que sigue en proceso hasta esto.

Antes de la fase 2, la primera prueba de campo es `evals.py` en modo no recursivo, que hoy arma un
`Workspace` pelado y puede armar un `Contenedor` con un flag. Sin fila en la bitácora: lo que el
modelo ve no cambia salvo en un timeout, y hay cero timeouts medidos.

Regla de diseño que no se negocia: el cliente MCP vive en el PADRE. El hijo es el lado no confiable
y darle red reabre todo lo que el proceso cierra. Y lo que el proceso NO arregla es la inyección;
para las rutas citadas ya está `citas.py`, y para la plaga en prosa es el punto 3.

## 2. El grep que afloja el patrón cuando devuelve cero (rebajado por el 3)

El cero es el estado donde el modelo tiene menos información y más incentivo a rellenar, y es
exactamente donde se inventó las tres fichas. Hoy `grep` dice "0 líneas casan" y lo suelta ahí.

Gemini CLI hace auto-enrichment: cuando el resultado sale flaco, ensancha sin que se lo pidan. No
se puede copiar tal cual, porque ensanchar en silencio es la misma mentira que el tope de 50. La
regla tendría que ser: ensanchar, decir con todas las letras qué se probó, y cerrar con la frase
que evita la invención. "0 con 'zompopo', 0 con 'zompopos', 0 con 'hormiga arriera'. El catálogo
no tiene esa plaga."

Rebajado porque en un corpus con ontología, que es el caso principal, `ontologia/objetivos.md`
está adentro del texto: el cero de `grep` ya es una afirmación sobre fichas y vocabulario
juntos, y con la cabecera por documento un hit en `ontologia/` se distingue de uno en una ficha
sin abrir nada. Lo que falta no es ensanchar a ciegas sino que el modelo lo sepa y pueda
recorrer padre y alias con estructura, que es el punto 3. Vuelve si un corpus sin ontología lo
pide.

## 3. La ontología como dato, no como chequeo de prosa

La cita contra el conjunto cerrado ya está (`citas.py`): las rutas que la respuesta declara
tienen que existir. Lo que no mira es la prosa, y no puede: una respuesta a "¿qué uso para el
zompopo?" va a decir "zompopo" porque la pregunta lo dice, y lo que está mal es recomendar un
producto para eso, que no es un string. Un chequeo determinístico de nombres en prosa contra
`ontologia/` no está bien definido.

Lo que sí es determinístico y barato: darle la ontología como dato. `objetivos` en `extra`,
parseado de las tablas `| id | etiqueta | padre | alias |` (71 entradas), más una línea del
`CONTRATO`: "si la plaga no está en `objetivos`, decilo antes de recomendar". El modelo ya puede
hacer `grep(catalogo, 'zompopo')` y recibir cero porque `ontologia/` está en el corpus; esto le da
el conjunto cerrado para que el cero sea una afirmación y no una ausencia. Es palanca de prompt y
builtin, y lleva A/B.

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
