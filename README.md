# catalejo

Recursive Language Model. El modelo nunca lee el contexto gigante: lo consulta escribiendo código.

```
uv run pytest
uv run mypy
```

Están `core`, que es un agente hecho con un solo tipo de dato y una sola operación, y `repl`, que es
el workspace donde vive el contexto, el worker que le pide código al modelo y la célula que lo
corre. El agente entero se arma así:

```python
ws = recurse(payload, Gemini(), depth=1)          # o Workspace(payload), sin delegar
agente = loop(
    then(worker(Gemini(), Handle(schema="el manual", size="4.2M tokens", tools=ws.tools)),
         executor(ws), grounded(ws.var)),
    max_steps=12,
    budget=200_000,
)
```

`uv run demo.py` lo corre de verdad contra Gemini sobre 7 MB de tickets sintéticos, y
`uv run demo.py v1` sobre 2,3 MB de código Go real. La pregunta pide
contar y ordenar a propósito, que es lo que un retriever no sabe hacer: los fragmentos más parecidos
a "cuántos hay" no son los que hay que contar. Una corrida:

```
corpus                7,259,840 caracteres (~1,814,960 tokens)
tokens gastados       4,095
ahorro                443x
turnos                7
voto                  DONE
```

La respuesta que dio (1768 tickets con E-417, el más viejo el ticket 016591 del 2024-01-01,
licuadora con 478) coincide exacto con contarlo afuera.

## El tipo: lo dicho

Un agente, mirado de lejos, no hace más que agregarle cosas a una pila de cosas dichas. Esa pila es
un `Log` y tiene tres canales.

```python
@dataclass(frozen=True, slots=True)
class Log:
    said: Conversation = ()
    fails: tuple[Fail, ...] = ()
    vote: Status = Status.QUIET
    spent: int = 0
    reads: int = 0
```

Dos observaciones que cambian todo lo que viene.

El estado del agente y la propuesta de una célula tienen la misma forma. No son tipos distintos, es
el mismo tipo mirado con distinto alcance. Un turno del usuario tampoco es un tipo aparte: es un
`Log` con un mensaje adentro.

Los errores viven adentro del `Log`, en `fails`. Un ejecutor que reventó es un hecho sobre el
turno, igual que algo que se dijo. Por eso ninguna función de `core` levanta una excepción.

## La operación

`merge(a, b)`, que escribo `•`. Cada canal se junta distinto, porque cada uno guarda otra clase de
cosa. `said` es una lista y concatena, porque el orden es el significado. `fails` es un conjunto,
porque el mismo error reportado dos veces es un error. `vote` es el máximo de `QUIET < DONE <
CONTINUE < HALT`, así que si alguien dice que falta información, falta información, sin importar
quién votó último. `spent` y `reads` son sumas.

Agregar un canal es agregar una línea en `merge`, no un `if` en el engine. `reads` entró así, mucho
después que los otros: una línea en el tipo, una en `merge`, una en el executor.

## Las leyes

`(Log, •, ZERO)` es un monoide: neutro y asociatividad valen en los cinco canales. La
conmutatividad vale en todos menos `said`. La idempotencia vale solo en `fails` y `vote`.

Que falle en `said`, `spent` y `reads` no es un defecto, es la verdad: repetir una respuesta la
duplica, reintentar cuesta plata, y consultar dos veces son dos consultas. El álgebra lo dice sola en vez de que haya que acordarse. Y la frontera
de la conmutatividad es justo la que dice qué se puede paralelizar.

Qué compra cada una. La asociatividad deja agrupar tres células en una y ponerle nombre sin cambiar
lo que hace el agente. El neutro deja cablear siempre la misma cadena con `identity()` donde la
feature está apagada. La conmutatividad hace que `fanout` quiera decir de verdad "en paralelo", sin
elegir entre determinismo y paralelismo. La idempotencia hace que reintentar sea gratis.

Y una que no es obvia: con leyes se pueden escribir property tests. `tests/test_laws.py` genera
logs al azar con hypothesis y verifica que la ley aguante siempre, en vez de probar un caso.

## La célula

```python
async def cell(seen: Log) -> Log: ...
```

Lee todo lo dicho y devuelve lo que agrega, solo lo que agrega. No muta nada, así que quien la
compuso decide si el agregado aterriza. Eso es lo que hace posible que otra célula lo revise antes,
y que un sub-loop entero quepa adentro de una célula sin que nadie afuera se entere. Ahí vive la
recursión.

`Cell` es un `Protocol` con `__call__`, así que cualquier función async ya es una célula.

## Componer

`then(f, g)` corre en orden y le muestra a cada una lo que agregaron las anteriores:

```
(f ▷ g)(seen) = f(seen) • g(seen • f(seen))
```

Es asociativo, y la asociatividad no está programada: se hereda de que `•` lo es. El detalle que la
sostiene es que el entorno lleva `seen`, no solo lo acumulado. Cambiar `merge(seen, acc)` por `acc`
a secas se ve casi igual y rompe la ley en silencio, porque un `then` anidado le tapa el contexto a
sus propias células. El resultado coincide igual, así que un test de resultado no lo agarra: hay
que mirar qué vio cada una, y eso es lo que hace
`test_asociatividad_en_lo_que_vio_cada_celula`.

`fanout(f, g)` corre al mismo tiempo y cada una ve exactamente lo que llegó, ninguna ve a las
otras. Junta en el orden en que se las dio, no en el que terminaron.

De ahí sale la regla práctica, y sale del álgebra y no del gusto de nadie: paralelizá las células
que buscan y las que juzgan, poné en fila las que hablan.

## El loop

`loop(cell, max_steps=..., budget=...)` repite mientras el último paso pida seguir. Mira el voto de
**ese** paso y no el acumulado, y la diferencia no es un detalle: `vote` se junta por máximo, así
que una vez que alguien votó CONTINUE el acumulado se queda ahí para siempre y un loop que lo
mirara no terminaría nunca. El acumulado dice "alguien pidió seguir alguna vez", el paso dice
"todavía falta".

Un paso que vota QUIET corta. Nadie pidió otra vuelta, así que un cableado a medio hacer se detiene
en vez de quemar tokens.

`max_steps` y `budget` son los topes mecánicos que garantizan que esto termina. Todo el resto de la
política vive en las células, que la expresan votando. Quedarse sin pasos o sin presupuesto no
levanta una excepción: agrega un `Fail`, y el voto que sale es CONTINUE, que es lo honesto, porque
lo cortaron a la mitad y seguía queriendo trabajar.

El voto de salida es el del último paso, no el máximo acumulado, y eso es lo que hace que un loop
sirva de célula. El máximo vale adentro de un alcance; cruzar el borde de un sub-loop resume, no
acumula. Sin eso, un sub-loop que dio dos vueltas le reporta CONTINUE al padre para siempre y el
padre no para nunca. El "seguí pensando" de adentro no se le escapa al de afuera, que es
exactamente lo que tiene que pasar cuando un modelo manda un sub-modelo a leer por él.

El gasto sí cruza el borde: `spent` suma hacia arriba, así que el presupuesto del padre ve lo que
gastaron todos sus hijos.

Un loop es una célula, así que cabe adentro de otro. Ahí va a vivir la recursión.

## El borde

Qué cruza el borde de un alcance es una decisión, y la de por defecto no es "todo". Eso ya estaba
pasando en dos lugares antes de tener nombre. El `loop` deja pasar todo menos el voto, que reemplaza
por el del último paso. Un sub-agente deja pasar el gasto y colapsa todo lo que dijo en un solo
texto.

`border(cell, rule)` le aplica una regla a lo que la célula devuelve, una sola vez, al salir. La
regla que viene hecha es `drop`, que vacía canales, y encima de ella hay dos con nombre:

```python
mute(cell)    # corre la célula y le tira lo que dijo. Lo que gastó queda
quiet(cell)   # corre la célula y le saca el voto. Todo lo demás queda
```

`mute` es para la que trabaja y no habla, como un crítico que vota pero no ensucia la transcripción.
Callarla no la abarata, y por eso `spent` sigue cruzando. `quiet` es para la que mira y no manda:
sin eso, cualquier célula que vote CONTINUE tiene al loop de rehén, porque el voto se junta por
máximo.

Lo que hace interesante a esto no es vaciar un canal, que es evidente, sino saber dónde se puede
mover el borde. `drop` es un homomorfismo de monoide: `h(a • b) == h(a) • h(b)`, o sea que vaciar la
suma es lo mismo que sumar los vaciados. De ahí sale que `mute(fanout(f, g))` y `fanout(mute(f),
mute(g))` son la misma cosa, porque en `fanout` cada célula ve exactamente lo que llegó y mover el
borde para adentro no le cambia la entrada a nadie.

Con `then` no vale, y no es un defecto de `mute`. `then` le muestra a cada célula lo que dijeron las
anteriores, así que callar a la primera le cambia la entrada a la segunda: el resultado final
coincide, pero la segunda vio otra cosa y en general contesta distinto. El homomorfismo vale sobre
`merge`, no sobre el orden. Es la misma frontera de siempre, que en `said` las leyes se terminan.

La otra mitad de la regla es cuáles reglas son legales. Una que colapse un canal en vez de vaciarlo,
como quedarse con el último dicho, no es un homomorfismo: el último dicho de `a • b` no es el último
de `a` junto con el de `b`. Por eso `answer()` es correcta aplicada una vez en el borde y sería un
desastre adentro de `merge`. La distinción no es decorativa, es la que dice qué se puede mover de
lugar y qué no.

`drop("spent")` es la que hay que mirar dos veces. Le esconde el gasto al presupuesto del padre y ahí
el árbol se financia solo. Vale cuando ese gasto ya se contó en otro lado, como el que `Metered` mide
en la fuente, y no vale para que las cuentas den lindas.

## El voto que frena

`HALT` es el único absorbente del álgebra y entró último. Sin él CONTINUE le gana a todo, o sea que
cualquier célula puede tener al loop de rehén y ninguna puede decir "pará". Lo pide un juez que ve
algo que no se arregla con otra vuelta: una inyección, una respuesta que no puede salir.

No hizo falta tocar el engine. `loop` ya cortaba con cualquier voto que no sea CONTINUE, así que un
valor nuevo arriba del orden funciona solo. Es una línea en el enum, que es la misma economía que
tuvo `reads` en su momento.

Y no reintroduce el error como cortocircuito, que es lo que está descartado más abajo. Un HALT frena
el turno siguiente, no el que está corriendo: las células ya cableadas de ese paso corren igual, y lo
que dijeron y lo que falló aterriza igual. Los otros canales no se enteran de que alguien votó HALT.
La diferencia con abortar es exactamente esa.

El HALT sí cruza el borde de un sub-loop, porque lo que sale es el voto del último paso. Un hijo que
frenó hace frenar al padre. Es lo que corresponde: el "seguí pensando" de adentro no tiene por qué
escaparse, "esto no sigue" sí.

## Reintentar

`retry(cell, attempts=n)` repite mientras la célula traiga `fails`, y tira lo que dijo el intento
perdido. Es el primer combinador que descarta una propuesta, y puede hacerlo porque una célula
devuelve lo que agrega sin mutar nada, así que quien la compuso decide si aterriza. Ese permiso
estaba en el diseño desde el principio sin que lo usara nadie.

Reintenta por `fails` y no por el voto. `fails` significa que se rompió la maquinaria, y un snippet
que revienta no es un `Fail` sino flujo normal del REPL, así que esto no se dispara ahí, que es lo
que se quiere: el modelo lee el error y lo corrige solo.

Qué sobrevive de un intento perdido es la parte que hay que pensar, y son `spent` y `fails`. El gasto
porque la llamada caída se paga igual, y la falla porque pasó de verdad. Se van `said`, `vote` y
`reads`. Los dos primeros son obvios. `reads` es el que importa: si quedara, un intento que leyó el
contexto y después reventó le dejaría el piso servido a `grounded`, y el intento que sí aterrizó
podría contestar de memoria pareciendo fundado. Es `drop("said", "vote", "reads")`, o sea la misma
proyección de la sección anterior.

Un intento que vota HALT no se reintenta, y ahí es donde las dos piezas se cruzan. Alguien decidió
que no hay otra vuelta, y reintentar sería desobedecerlo.

El último intento vuelve entero. Fallar hasta el final no es abortar: el que llamó se lleva lo que
haya, con las fallas de todos a la vista, y decide él. Es la misma regla que hace total a toda la
composición.

Acá se ve para qué servía la frontera de la idempotencia. Reintentar es gratis en `fails` y `vote`,
así que el mismo error dos veces sigue siendo uno, y no lo es en `spent`, así que el álgebra te cobra
los tres intentos sin que nadie tenga que acordarse de sumarlos.

## El REPL

El contexto grande vive en un `Workspace` como una variable, y nunca llega al prompt. Al prompt va a
llegar solo el handle: el nombre de la variable, su tamaño y un esquema de una línea. Esa es toda la
economía del asunto, y por eso un contexto de 4M tokens no paga 4M tokens por turno.

`Workspace` es `exec` sobre un namespace que persiste entre corridas, así que el modelo puede
guardar un hallazgo en una variable en vez de depender de releer la transcripción. Rebanar y medir
salen gratis porque es Python de verdad y no un DSL propio. Lo único que hay que dar es `grep`, y
solo porque `re` no es importable.

`executor(env)` es la célula que corre el bloque que el modelo acaba de proponer y devuelve la
salida como el turno siguiente. El voto sale de la FORMA del mensaje: hubo bloque cercado quiere
decir que sigue trabajando, así que vota CONTINUE; no hubo quiere decir que ya contestó, así que
vota DONE y el loop corta. No hace falta un supervisor aparte, porque la célula que parsea la forma
es la que sabe.

Un bloque cercado sin cerrar no corre nunca. Si corriera, una respuesta cortada a la mitad rompería
el contrato de "prosa quiere decir terminé".

Y corre solo lo que dijo el modelo, así que mira el rol antes que el texto. Cuando el worker se cae
no agrega nada y el último dicho pasa a ser la salida anterior del REPL, que es texto que salió del
contexto: si ahí adentro viene un bloque cercado, mirar el último dicho sin mirar quién lo dijo es
ejecutar el corpus. Sin propuesta el executor no opina, el voto queda en QUIET y el turno corta con
el `Fail` del modelo a la vista.

De un mensaje con dos bloques corre el primero y lo dice. Tragarse el segundo es el mismo defecto que
el `grep` que recortaba en silencio: el modelo pidió dos cosas, ve una salida y no tiene cómo saber
que la otra nunca pasó.

Un snippet que revienta no es un `Fail`. Es flujo normal del REPL: el modelo ve el error en la
salida y lo corrige en el bloque siguiente. `fails` es para cuando se rompe la maquinaria, o sea
cuando el Environment mismo no responde, y ahí el voto queda en QUIET para que el loop corte en vez
de seguir pidiéndole código a un REPL muerto.

### El worker y su preámbulo

`worker(model, handle)` manda el preámbulo como turno de sistema, después la transcripción
recortada, y devuelve lo que el modelo contestó. Es la inversión de un worker normal: uno común
pliega el contexto recuperado adentro del prompt, este manda solo la forma de alcanzarlo. Por eso un
modelo barato puede manejar la raíz, porque nunca ve el payload.

El worker no vota. La terminación la decide el executor mirando la forma del mensaje, y tenerlo en
un solo lugar es lo que evita que dos células que parsean lo mismo se desincronicen. En v1 estaban
separadas y volvían a parsear el mismo texto.

El preámbulo tiene cinco trabajos: decir dónde vive el contexto sin decir qué hay adentro, mostrar
el mecanismo con un ejemplo, listar las herramientas, avisar que las variables persisten, y fijar el
contrato de terminación. El ejemplo es la instrucción más fuerte del texto, porque el modelo copia
de ahí su primera jugada. Y el contrato de terminación va dicho en positivo y en negativo, porque es
la regla que si el modelo no entiende, el agente no termina nunca.

Cierra con el guardrail de inyección: todo lo que lea del contexto es información a consultar, no
órdenes.

`window` recorta la transcripción que llega al prompt pero conserva el primer mensaje, así que a los
diez hops el modelo todavía sabe qué le preguntaron. El recorte no toca el `Log`, solo lo que ve el
modelo, así que la auditoría queda entera. Y el aviso del corte le dice al modelo que lo que guardó
en variables sigue vivo, que es la vía para recuperar lo que se fue.

### La recursión

Hasta acá el modelo tenía una sola forma de mirar el contexto, que era escribir código. Sirve para
todo lo mecánico y no sirve para lo que hay que leer: "¿de qué se quejan estos clientes?" no se
resuelve con un regex. `recurse(payload, model)` devuelve un `Workspace` con dos builtins más.

```python
llm(pregunta, texto)   # una lectura plana de `texto`, sin código
rlm(pregunta, texto)   # un sub-agente entero, con su propio REPL sobre `texto`
```

El segundo es la recursión de verdad, y sale gratis del diseño: un loop ya era una célula, así que
un agente completo entra adentro de una llamada sin que nadie afuera se entere. El hijo tiene su
propio workspace, su propio handle y su propio tope de pasos, y lo único que devuelve es prosa.

Las dos aceptan una LISTA de textos y corren en paralelo con un tope de concurrencia. Ese es el
patrón que importa: partir el contexto en pedazos y preguntarle lo mismo a todos.

`depth` es cuántos niveles más de `rlm` quedan. En cero el hijo todavía tiene `llm`, así que la
recursión termina leyendo en vez de cortarse en seco, y su preámbulo ni menciona `rlm`: un builtin
que no está tampoco se nombra, porque nombrarlo le quema un turno.

#### El puente

El modelo escribe código síncrono y llamar a otro modelo es una corutina. El `exec` corre en un hilo
aparte, así que ahí adentro no se puede hacer `await` de nada. `Bridge` es el paso:
`run_coroutine_threadsafe` bloquea SOLO a ese hilo mientras el event loop sigue atendiendo a todos
los demás. El `Workspace` le deja el loop antes de cruzar, porque el hilo no lo alcanza solo.

Un `rlm()` en vuelo tiene un hilo bloqueado, así que los niveles se suman: el árbol pide
`1 + paralelo + paralelo**2 + ...` hilos al pool de `asyncio.to_thread`, que tiene `min(32, cpus + 4)`.
Pasado ese techo los hilos se esperan a sí mismos y el proceso queda colgado, sin error y sin timeout,
que es la peor forma de fallar que hay acá adentro. Por eso `paralelo` sin decir nada es el que entra
en la máquina donde corre, y uno pedido a mano que no entre levanta `ValueError` al cablear: un
`paralelo=20` con `depth=2` pide ocho mil hilos y eso se sabe antes de arrancar, no a la hora de
colgarse. Es otra cosa que arregla el contenedor, donde cada sandbox es un proceso y no un hilo.

#### La plata

El gasto de un hijo tiene que llegar al presupuesto del padre o el árbol se financia solo. La cadena
es `Metered` avisa al `Bridge`, el `Workspace` mide la corrida y la devuelve en `Output.spent`, el
executor la sube al `Log`, y de ahí en adelante la lleva el álgebra, que ya sumaba `spent` hacia
arriba. La trampa está en que las ventanas de medición se anidan en el tiempo: el hijo mide su
corrida mientras la del padre sigue abierta. Se resuelve midiendo en un solo lugar, que es donde el
gasto nace: toda llamada a un modelo que arranque adentro del REPL pasa por `Metered` y se cuenta
exactamente una vez. `tests/test_recurse.py` lo fija con `spent == len(model.visto) * COSTO`, y
meter una suma de más rompe cinco tests.

La recursión tiene su propio `budget`, aparte del que lleva el loop. Hace falta porque una sola
corrida del REPL puede abrir cincuenta llamadas, y el loop recién mira cuando el paso terminó, o sea
cuando ya se gastaron. Agotado, delegar devuelve un aviso en vez de llamar y el modelo sigue con lo
que tiene.

#### Cuándo NO delegar

El párrafo que más trabaja del preámbulo nuevo es el que dice cuándo no usar esto. Un modelo con un
builtin nuevo lo usa para todo, y pagar tokens para contar líneas es peor que no tenerlo: cuesta
plata y encima se equivoca, porque contar leyendo es justo lo que un modelo hace mal. Contar,
filtrar y ordenar con código; leer, resumir y juzgar con un modelo.

Costó tres corridas en vivo entender dónde está el límite, y todas las perdió el corpus, no el
modelo. Le pedí agrupar por tema las quejas de 1768 tickets esperando que delegara: dedupeó con
código, se encontró con diez textos únicos y los agrupó él mismo. Le pedí un mapa de los módulos de
`v1`: encontró los `doc.go` y leyó seiscientos caracteres de cada uno. Las dos veces tenía razón.
Delegar recién paga cuando lo que quedó después de filtrar sigue sin entrar en un prompt, y un
corpus de plantillas nunca llega a eso.

La cuarta sí. `uv run demo.py v1` le pide un índice de los 80 archivos de
`v1/planner/internal/plan`, diciendo de cada uno qué hace y si tiene lógica de negocio o es
andamiaje. Eso no lo contesta un encabezado y no hay `doc.go` que lo resuelva: hay que leer 800 KB.

```
corpus                2,314,229 caracteres (~578,557 tokens)
tokens gastados       261,415
delegado              80 llamadas, 243,259 tokens
turnos                9
voto                  DONE
```

Ochenta llamadas, una por archivo, en paralelo. El 93% del gasto lo hicieron los hijos y la raíz
pagó 18 mil tokens por leerse 2,3 MB de Go a través de ellos. Eso es lo que compra la recursión, y
no es ahorro de tokens: alguien tiene que leer el código y lo lee una vez de cualquier forma. Lo que
se ahorra es el prompt de la raíz, que se queda chico mientras el trabajo pesado pasa afuera, en
llamadas que se descartan. Los resúmenes que volvieron son correctos: `billing_text.go` limpia
muletillas de los datos de facturación antes de que salgan a la factura y al ERP, que es exactamente
lo que dice el archivo.

Lo que devuelve un sub-agente es su último dicho en prosa, no el último dicho a secas: uno que se
quedó sin pasos tiene un bloque de código ahí, y devolver eso sería devolver la mitad de un
pensamiento. Y si se cortó, lo dice. Una respuesta incompleta que no se anuncia es el mismo error
que un `grep` que recorta en silencio.

#### Un error que no enseña se paga caro

La primera corrida contra el código de `v1` salió mal de la peor manera. El modelo escribió
`import re`, Python contestó `ImportError: __import__ not found`, y ahí se rindió: dejó de escribir
código y contestó el índice de los 80 archivos de memoria. Inventó todo. Listó archivos `_test.go`
que el corpus no tiene y un `action.go` que no existe en ningún lado del repo. Gastó 36 mil tokens,
votó DONE y no reportó un solo `fail`. El sistema dijo que había salido bien.

El arreglo inmediato es que el error enseñe: ahora `import` levanta un `ImportError` que dice que no
hay imports acá, que para regex está `grep`, y cómo contar con un dict. Con eso la misma pregunta
pasó a contestarse con archivos de verdad. Es la misma lección que el `grep` que recortaba en
silencio: lo que el modelo lee de vuelta del REPL es la mitad del diseño.

Lo que el arreglo no resuelve es más grande, y es lo que viene abajo.

### La célula de grounding

La terminación de este agente es sintáctica: prosa quiere decir terminé. Alcanza para decidir el
control y no alcanza para nada más, porque un modelo que contesta sin haber mirado el contexto
termina igual de bien que uno que lo leyó entero. Eso fue lo que pasó con `v1`, y el sistema lo
reportó como éxito.

`grounded()` va después del executor y mira `reads`. En cero, lo que el modelo esté por contestar no
salió del contexto, así que le devuelve un aviso y pide otra vuelta.

```python
then(worker(model, handle), executor(ws), grounded(ws.var))
```

Lo interesante es cómo veta, porque no puede. Una célula devuelve lo que agrega y no puede borrar lo
que otra dijo, así que el DONE del executor no se cancela. Y no hace falta: `vote` se junta por
máximo y CONTINUE es mayor que DONE, así que alcanza con opinar más fuerte. Para eso estaba el orden
de `Status` desde el principio, y esta es la primera célula que lo usa de verdad. Un veto sin
mecanismo de veto, que sale del álgebra sola.

El sub-agente lleva el mismo cableado. Un barrido de ochenta llamadas son ochenta lugares donde
contestar de memoria, con menos pasos y sin nadie que lea la transcripción, y lo que vuelve es prosa
que el padre no distingue de la buena.

Avisa una vez. Si el modelo vuelve a contestar de memoria después del aviso, lo deja terminar y
anota el `Fail`: la respuesta sale, pero sale etiquetada. Insistir hasta el tope de pasos serían doce
llamadas para llegar a la misma conclusión, y el que ignoró el aviso una vez lo va a ignorar diez.

Comparte con el executor la única fuente de verdad sobre qué cuenta como código, que es
`extract_code`. La regla es un solo parser, no un solo llamador: dos células que parsean lo mismo con
su propio código se desincronizan, dos que llaman a la misma función no.

`reads` es un piso, no una prueba. Dice que el REPL le devolvió algo al modelo, no que lo que el
modelo afirma después salga de ahí. Atrapa el caso que pasó de verdad, que es no haber mirado nada, y
no atrapa una cita inventada sobre un texto que sí leyó. Eso pide un juez, y un juez es otra célula.

### El invariante

El `Environment` es una ventana de solo lectura. Todo builtin que se le dé al modelo puede leer y
devolver texto, y nada más. `llm` y `rlm` salen a la red, y siguen adentro del invariante porque
mandan texto a un modelo y traen texto: no escriben nada en ningún lado. Lo que sí agregan es una
superficie nueva, porque el texto que viaja sale del contexto no confiable. Por eso el sub-agente
lleva el mismo guardrail de inyección en su preámbulo, y por eso lo que contesta vuelve al padre
como un dato más y no como una instrucción. El código que el modelo escribe procesa texto no confiable, así que este
invariante es la frontera de seguridad: una inyección que secuestre al modelo, en el peor caso, lee
lo que el modelo ya podía leer. Cualquier capacidad con efectos va detrás de aprobación humana
explícita, nunca como un builtin más.

Dicho eso, `Workspace` **no es un sandbox**. Los builtins vienen recortados, sin `import`, sin
`open`, sin `eval`, y eso alcanza para que el modelo no rompa nada por accidente. No alcanza contra
código que se lo proponga, porque desde cualquier objeto se llega a `__class__.__bases__`. Tampoco
hay timeout, porque no se puede interrumpir un `exec` en curso desde otro hilo. Las dos cosas las
arregla el contenedor, y va después.

### Sin techo de truncado

En v1 la salida del REPL se cortaba a 2000 caracteres y eso destruía hechos: el precio del único
producto de un catálogo caía 68 caracteres pasado el corte, así que el worker contestaba "no me
apareció el precio" con el documento en la mano. El presupuesto se cuida acotando el dato en la
fuente, no la salida por conteo de caracteres, que corta a ciegas donde caiga.

La primera corrida contra Gemini mostró que acotar en la fuente tampoco alcanza si el corte no se
anuncia. `grep` devolvía 50 líneas y se callaba que había 1823. El modelo preguntó cuántos tickets
tenían el error, contó las 50 que le dimos y contestó 50, con toda la confianza del mundo. Había
hecho lo correcto; le mentimos nosotros.

Un tope silencioso es peor que no tener tope, porque el error no se ve. Ahora `grep` abre siempre
con cuántas líneas casan de verdad, dice cuántas muestra y cómo pedir el resto. Cero coincidencias
también se dice con todas las letras, porque devolver "" es indistinguible de un snippet que no
imprimió nada.

### El modelo

`Model` es un método: recibe una `Conversation`, devuelve un `Reply` con el texto y lo que costó.
No conoce el `Log`, no vota, no decide nada. Si la red falla levanta excepción y el worker la
convierte en `Fail`, igual que el executor con el `Environment`.

`Gemini` es HTTP crudo contra la REST de Google, sin el SDK. El mapeo son sesenta líneas de JSON que
se leen de una sentada, y el SDK trae un árbol de dependencias y su propio contrato async para hacer
lo mismo. Cuando el mapeo crezca (imágenes, tools, streaming) la cuenta cambia y se revisa.

Tres cosas del mapeo no son obvias y las tres salieron de leer respuestas reales.

Los resúmenes de razonamiento vienen marcados con `thought` y se descartan. Si se concatenaran,
un razonamiento que menciona código trae backticks y el executor lo correría en vez de leer el turno
como respuesta final: el contrato de terminación se rompe desde adentro del modelo.

Cortar por `MAX_TOKENS` es un turno roto, no un turno corto. El bloque cercado queda sin cerrar,
`extract_code` no lo corre, el executor lo lee como prosa y vota DONE, así que el agente devolvería
media oración como si fuera la respuesta. Es un `Fail` con el motivo.

Dos turnos seguidos del mismo rol se juntan en uno con varias partes, porque Gemini quiere que
alternen. No es defensa teórica: `window` mete el aviso de recorte como USER justo después de la
pregunta, que también es USER.

`temperature` no se manda si no se pide. Los Gemini 3 vienen calibrados en 1.0 y bajarlos a 0
buscando determinismo los empeora y los hace repetirse. Los 429 y los 503 se reintentan con espera
que se duplica; un 400 no, porque es el pedido mal armado y reintentarlo es quemar tiempo.

## Lo que decidimos no tener

**El error como cortocircuito.** Si una célula pudiera abortar la cadena, la composición dejaría de
ser total y aparecería un elemento absorbente, como el cero en la multiplicación. Con tres
buscadores y uno caído se muere el turno entero aunque tuvieras dos respuestas buenas en la mano.
Acá el error es un canal, el agente degrada en vez de abortar, y decide si es fatal quien tiene el
contexto para saberlo.

**Un tipo aparte para la entrada.** Si el turno del usuario ya es un `Log`, no hace falta un tipo
suma que diga "o un mensaje o una propuesta, exactamente uno".

**El canal `feedback`.** En `v1/orch` una crítica del supervisor se limpiaba sola cuando nadie
hablaba, y eso contradice la ley del neutro: el neutro no puede cambiar nada. Una crítica es un
`Message` más y así queda auditable.

**El canal `found`.** Entra cuando haya un retriever. En un RLM el contexto vive detrás de un
handle y no entra al prompt, así que puede no hacer falta nunca.

## Comparado con `v1`

Es un puerto de `v1/algebra`, sin el canal `found` y sin el `ctx` de Go. La cancelación y el
presupuesto van a ser `contextvars` más `TaskGroup` cuando `fanout` los necesite.

De `v1/orch` sobrevive lo importante, que es proponer en vez de aplicar y que la propuesta sea data
pura. No sobrevive el `Signal` como tipo suma, ni un merge único para canales con álgebras
distintas, ni tener dos fusiones diferentes, una adentro del paso y otra entre pasos. Acá hay una
sola operación y se usa en los dos lugares.
