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
eso es el punto 2.

## 2. La cita contra el conjunto cerrado, adentro del agente

`inventada()` vive en `evals.py`, o sea que califica después y no frena nada. El modo de falla que
persigue todo el repo (`zompopo-fuera-de-catalogo`: tres fichas inventadas, beaveria-90,
isaria-forte, metarhizium-50) lo agarra el eval, no el agente.

El docstring de `grounding.py` dice que para esto haría falta un juez. Se equivoca, y `inventada()`
es la prueba: una cita es una afirmación sobre un conjunto conocido y se verifica con código,
exacto y gratis. La célula nueva va al lado de `grounded(var)`, con el mismo mecanismo: avisa una
vez votando CONTINUE con las rutas que no existen, y si el modelo insiste lo deja salir con un
`Fail`.

Segundo escalón, más fuerte: validar contra `ontologia/`, que son vocabularios cerrados de verdad
(`objetivos.md` tiene 71 entradas con nombre común, científico y alias). "zompopo" no aparece ni una
vez en el bundle, así que la respuesta correcta era decir que el catálogo no lo cubre, y eso es
chequeable sin modelo. El path chequea rutas citadas; la ontología chequea el nombre que el modelo
usa en prosa aunque no cite nada.

## 3. La cabecera `=== ruta ===` cuenta como hit

`grep` recorre todas las líneas y la cabecera es una línea más que además contiene la ruta, así que
un patrón que nombra un producto o una carpeta casa contra el delimitador. Medido sobre el bundle:
`productos` da 279 hits de los cuales 39 son cabeceras (14%), `ontologia` 9 de 24 (37%), `viventem`
2 de 19.

En un agente cuyo argumento es contar bien, eso infla el total que anuncia la primera línea y gasta
tokens en líneas sin contenido. El arreglo es un `continue` en el loop de `grep`, después de la
bookkeeping que ya detecta la cabecera con `CABECERA`. Barato, y cambia el número que ve el modelo,
así que lleva fila en la bitácora.

**Va antes que cualquier A/B nuevo.** Si no, `max_hits` y `doc=` se miden contra totales inflados y
hay que re-baselinear dos veces.

## 4. El grep que afloja el patrón cuando devuelve cero

El cero es el estado donde el modelo tiene menos información y más incentivo a rellenar, y es
exactamente donde se inventó las tres fichas. Hoy `grep` dice "0 líneas casan" y lo suelta ahí.

Gemini CLI hace auto-enrichment: cuando el resultado sale flaco, ensancha sin que se lo pidan. No
se puede copiar tal cual, porque ensanchar en silencio es la misma mentira que el tope de 50. La
regla tiene que ser: ensanchar, decir con todas las letras qué se probó, y cerrar con la frase que
evita la invención. "0 con 'zompopo', 0 con 'zompopos', 0 con 'hormiga arriera'. El catálogo no
tiene esa plaga."

## 5. El two-phase para rescatar el `doc=`

El arm `con_doc` ganó -58% en tokens y perdió calidad: tres casos de 3/3 a 2/3, tres formas
distintas, todas sin consultar el contexto. El gate lo bloquea. Desbloquearlo pide n=6 en los DOS
arms, no solo en el nuevo, y eso es plata.

La alternativa es no re-medir lo mismo sino cambiar el diseño: Codex CLI devuelve solo nombres de
archivo y obliga a un `read` aparte, así que no hay contenido que contestar sin pedirlo. Es un arm
nuevo y arranca de cero en n. Ojo con el mecanismo: la falla fue hacer DE MENOS (4.2 turnos contra
5.7), y two-phase fuerza más llamadas, lo cual es plausible como cura y también se come parte del
ahorro.

## 6. `max_hits=50`, nunca medido

Entre 1.229 y 3.079 tokens por llamada sobre el bundle, contra una corrida entera que promedia
7.788. Es el default más caro sin A/B. Bajarlo a 20 es una palanca de una palabra y la red ya está
puesta: la cabecera dice el total de verdad y cómo pedir el resto. El riesgo es que canjee tokens
por turnos, que es justo lo que la bitácora sabe leer.

## Cerrado, no reabrir

- **ripgrep**: pierde el plegado de acentos, que es la palanca medida (`pulgon` da 33 líneas contra
  5 de `rg -i`, `arana roja` 9 contra 0). Vuelve a la mesa si el corpus deja de entrar dos veces en
  memoria o si el bundle vive en disco de verdad.
- **El tope por ancho de línea** que tienen opencode y Cursor: medido, p99 del bundle son 418
  caracteres y ninguna línea pasa de 2.000. Ahorraría 10%. No vale el código.
- **Un motor de regex lineal en Python**: `rure` está abandonado. La única opción mantenida era el
  paquete `regex` con `timeout=`, y se descartó porque el cuelgue ya tiene dueño en el punto 1.
