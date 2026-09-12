import asyncio

from catalejo.repl import Workspace, grep
from catalejo.repl.workspace import MAX_PLEGADOS, _PLEGADOS


class TestWorkspace:
    async def test_el_payload_vive_en_la_variable(self) -> None:
        ws = Workspace("hola mundo")

        out = await ws.run("print(ctx)")

        assert out.stdout == "hola mundo\n"
        assert out.err == ""

    async def test_la_variable_se_puede_renombrar(self) -> None:
        ws = Workspace("hola", var="manual")

        assert (await ws.run("print(manual)")).stdout == "hola\n"

    async def test_las_variables_persisten_entre_corridas(self) -> None:
        """Es lo que deja al modelo guardar un hallazgo en vez de depender de
        releer la transcripción, que se va recortando."""
        ws = Workspace()

        await ws.run("notas = []")
        await ws.run("notas.append('el precio')")
        out = await ws.run("print(notas)")

        assert out.stdout == "['el precio']\n"

    async def test_python_de_verdad_sobre_el_contexto(self) -> None:
        """Rebanar y medir salen gratis porque es Python, no un DSL propio."""
        ws = Workspace("abcdefghij")

        out = await ws.run("print(len(ctx)); print(ctx[2:5])")

        assert out.stdout == "10\ncde\n"

    async def test_un_error_no_explota_lo_ve_el_modelo(self) -> None:
        ws = Workspace()

        out = await ws.run("print(no_existe)")

        assert out.err == "NameError: name 'no_existe' is not defined"
        assert out.stdout == ""

    async def test_un_error_de_sintaxis_tambien(self) -> None:
        ws = Workspace()

        out = await ws.run("print(")

        assert out.err.startswith("SyntaxError:")

    async def test_lo_impreso_antes_del_error_sobrevive(self) -> None:
        ws = Workspace()

        out = await ws.run("print('voy bien')\nprint(no_existe)")

        assert out.stdout == "voy bien\n"
        assert out.err.startswith("NameError:")

    async def test_la_salida_se_limpia_entre_corridas(self) -> None:
        ws = Workspace()

        await ws.run("print('vieja')")
        out = await ws.run("print('nueva')")

        assert out.stdout == "nueva\n"

    async def test_corridas_concurrentes_no_se_pisan(self) -> None:
        """El namespace es estado compartido, así que run serializa."""
        ws = Workspace()

        outs = await asyncio.gather(ws.run("print(1)"), ws.run("print(2)"))

        assert sorted(o.stdout for o in outs) == ["1\n", "2\n"]


class TestVentanaDeSoloLectura:
    async def test_no_se_puede_importar(self) -> None:
        ws = Workspace()

        out = await ws.run("import os")

        assert out.err.startswith("ImportError:")

    async def test_el_import_dice_que_usar_en_su_lugar(self) -> None:
        """`ImportError: __import__ not found` no lleva a ninguna parte. Dos
        corridas contra Gemini se perdieron ahí, y en una el modelo dejó de
        intentar y contestó de memoria."""
        ws = Workspace()

        out = await ws.run("import re")

        assert "`re` no está disponible" in out.err
        assert "grep(texto, patron)" in out.err

    async def test_no_hay_open(self) -> None:
        ws = Workspace()

        out = await ws.run("open('/etc/passwd')")

        assert out.err == "NameError: name 'open' is not defined"

    async def test_no_hay_eval_ni_exec(self) -> None:
        ws = Workspace()

        assert (await ws.run("eval('1+1')")).err.startswith("NameError:")
        assert (await ws.run("exec('x=1')")).err.startswith("NameError:")


class TestGrep:
    def test_devuelve_las_lineas_numeradas(self) -> None:
        texto = "uno\ndos precio\ntres\ncuatro precio"

        assert grep(texto, "precio") == (
            "2 líneas casan con 'precio'.\n2: dos precio\n4: cuatro precio"
        )

    def test_sin_hits_lo_dice_con_todas_las_letras(self) -> None:
        """Devolver "" es indistinguible de un snippet que no imprimió nada."""
        assert grep("uno\ndos", "tres") == "0 líneas casan con 'tres'."

    def test_acepta_expresiones_regulares(self) -> None:
        """Existe porque `re` no es importable: esta es la vía al regex."""
        out = grep("Q475.00\nsin precio", r"Q\d+\.\d\d")

        assert out.splitlines()[-1] == "1: Q475.00"

    def test_el_total_va_siempre_aunque_no_recorte(self) -> None:
        """El modelo tiene que poder contar sin depender de si hubo recorte."""
        assert grep("hola", "hola").startswith("1 línea casa con")

    def test_dice_cuantas_hay_de_verdad_cuando_recorta(self) -> None:
        """El bug de la primera corrida real: el modelo contó las 50 que le
        devolvimos y contestó 50, sobre un corpus con 1823. Un tope que no se
        anuncia es peor que no tener tope, porque el error no se ve."""
        texto = "\n".join(f"linea {i}" for i in range(1000))

        out = grep(texto, "linea", max_hits=3)

        assert out.startswith("1000 líneas casan con 'linea'; estas son las primeras 3.")
        assert "max_hits" in out.splitlines()[0]
        assert len(out.splitlines()) == 4

    def test_no_distingue_mayusculas(self) -> None:
        """El modelo escribe el término del cliente, que va en minúscula.

        Sobre el catálogo agronómico `mosca blanca` devolvía 6 de las 23 líneas que
        hay, y el modelo no tenía cómo enterarse: el número que recibe parece el total.
        """
        texto = "Mosca Blanca ( Bemisia tabaci )\nmosca blanca en tomate"

        assert grep(texto, "mosca blanca").startswith("2 líneas casan")

    def test_no_distingue_acentos(self) -> None:
        """`arana roja` devolvía cero sobre un texto que la tiene seis veces."""
        texto = "Araña roja ( Tetranychus urticae )\nPulgón ( Aphis gossypii )"

        assert grep(texto, "arana roja").startswith("1 línea casa")
        assert grep(texto, "pulgon").startswith("1 línea casa")

    def test_devuelve_la_linea_original_no_la_plegada(self) -> None:
        """Se busca sobre la copia sin acentos y se muestra lo que dice el texto."""
        out = grep("Pulgón ceniciento", "pulgon")

        assert out.splitlines()[-1] == "1: Pulgón ceniciento"

    def test_exacto_respeta_el_caso(self) -> None:
        """En el corpus de código Go, `Plan` y `plan` son cosas distintas."""
        texto = "type Plan struct\nfunc plan() {}"

        assert grep(texto, "Plan", exacto=True).startswith("1 línea casa")
        assert grep(texto, "Plan").startswith("2 líneas casan")

    def test_no_rompe_las_clases_de_la_regex(self) -> None:
        """Bajar el patrón a minúsculas convertiría `\\S` en `\\s`, que es lo contrario.

        Se rompería en silencio: la búsqueda devuelve otra cosa, no un error.
        """
        assert grep("Q475.00\nsin precio", r"Q\d+\.\d\d").splitlines()[-1] == "1: Q475.00"
        assert grep("hola mundo\nholamundo", r"hola\S").splitlines()[-1] == "2: holamundo"

    def test_el_hit_trae_su_documento(self) -> None:
        """Encontrar la línea no sirve si no se sabe de qué archivo es.

        Sobre el catálogo agronómico un hit cae a 119 líneas de su cabecera, así que
        averiguarlo costaba imprimir una ventana y caminar para atrás. Una corrida se
        fue a 108 mil tokens haciendo eso.
        """
        texto = "=== productos/segador.md ===\n1.42 L/Ha\n=== productos/otro.md ===\nnada"

        assert grep(texto, "L/Ha").splitlines()[-1] == "productos/segador.md:2: 1.42 L/Ha"

    def test_sin_cabeceras_numera_como_siempre(self) -> None:
        """Un texto que no es una concatenación de documentos no gana un prefijo vacío."""
        assert grep("uno\ndos precio", "precio").splitlines()[-1] == "2: dos precio"

    async def test_esta_en_el_namespace_del_modelo(self) -> None:
        ws = Workspace("uno\ndos precio")

        out = await ws.run("print(grep(ctx, 'precio'))")

        assert out.stdout == "1 línea casa con 'precio'.\n2: dos precio\n"


class TestGrepPorDocumento:
    CATALOGO = (
        "=== productos/viventem.md ===\n"
        "DOSIS: 1.0 L/Ha\n"
        "pH del agua 5.5\n"
        "=== productos/segador.md ===\n"
        "DOSIS: 0.5 L/Mz\n"
    )

    def test_acota_a_una_ficha(self) -> None:
        out = grep(self.CATALOGO, "DOSIS", doc="viventem")

        assert out.splitlines()[0] == "1 línea casa con 'DOSIS' en productos/viventem.md."
        assert out.splitlines()[-1] == "productos/viventem.md:2: DOSIS: 1.0 L/Ha"

    def test_el_prefijo_impreso_no_es_buscable(self) -> None:
        """El cero falso que paga esto.

        El resultado sale como `ruta:linea: contenido`, así que el modelo deduce lo
        razonable y ancla el patrón en la ruta. Ese prefijo se arma al imprimir, no
        está en el texto, y el patrón no casa nunca. En la suite agronómica contra
        luna pasó cinco veces en una corrida, todas en el caso que pregunta si un
        dato está.
        """
        assert grep(self.CATALOGO, r"productos/viventem\.md:.*DOSIS").startswith("0 líneas")

        assert grep(self.CATALOGO, "DOSIS", doc="viventem").startswith("1 línea casa")

    def test_una_ficha_que_no_existe_no_es_cero_lineas(self) -> None:
        """"No hay líneas en esa ficha" y "esa ficha no existe" son dos respuestas."""
        out = grep(self.CATALOGO, "DOSIS", doc="royano")

        assert out.startswith("ningún documento casa con 'royano'")
        assert "`=== ruta ===`" in out

    def test_la_ficha_esta_y_el_dato_no(self) -> None:
        out = grep(self.CATALOGO, "L/Mz", doc="viventem")

        assert out == "0 líneas casan con 'L/Mz' en productos/viventem.md."

    def test_varias_fichas_casan_y_lo_dice(self) -> None:
        out = grep(self.CATALOGO, "DOSIS", doc="productos/")

        assert out.splitlines()[0] == (
            "2 líneas casan con 'DOSIS' en los 2 documentos que casan con 'productos/': "
            "productos/viventem.md (1), productos/segador.md (1)."
        )

    def test_pliega_como_el_patron(self) -> None:
        texto = "=== fichas/Pulgón.md ===\nnada\n"

        assert grep(texto, "nada", doc="pulgon").startswith("1 línea casa")

    def test_sin_doc_no_cambia_nada(self) -> None:
        """El default tiene que dar exactamente lo de antes, o las filas viejas mienten."""
        assert grep("uno\ndos precio", "precio") == "1 línea casa con 'precio'.\n2: dos precio"

    def test_el_tope_dice_el_ambito(self) -> None:
        texto = "=== a.md ===\n" + "\n".join(f"linea {i}" for i in range(100))

        out = grep(texto, "linea", max_hits=3, doc="a.md")

        assert out.startswith("100 líneas casan con 'linea' en a.md; estas son las primeras 3.")
        assert len(out.splitlines()) == 4


class TestCabeceraPorDocumento:
    """La primera línea habla en documentos, y la línea `=== ruta ===` no es una."""

    CATALOGO = (
        "=== productos/segador.md ===\n"
        "pulgón: 1.0 L/Ha\n"
        "pulgón verde también\n"
        "=== productos/index.md ===\n"
        "- [Bio BPBS](bio-bpbs.md)\n"
        "=== ontologia/objetivos.md ===\n"
        "| pulgon | Pulgón | insecto |\n"
        "=== productos/bio-bpbs.md ===\n"
        "Bio BPBS, para trips\n"
    )

    def test_dice_en_que_documentos_y_cuantas_en_cada_uno(self) -> None:
        """Sobre el catálogo, `pulgon` son 24 líneas en 3 documentos. Esa lista es
        el esqueleto de la respuesta: qué fichas, y aparte qué dice el vocabulario.
        Antes el modelo la reconstruía leyendo 24 prefijos."""
        out = grep(self.CATALOGO, "pulgon")

        assert out.splitlines()[0] == (
            "3 líneas casan con 'pulgon' en 2 documentos: productos/segador.md (2), "
            "ontologia/objetivos.md (1)."
        )

    def test_un_solo_documento_se_nombra_sin_lista(self) -> None:
        assert grep(self.CATALOGO, "verde").splitlines()[0] == (
            "1 línea casa con 'verde' en productos/segador.md."
        )

    def test_la_cabecera_no_cuenta_ni_se_imprime(self) -> None:
        """Contiene la ruta, así que un patrón que nombra la carpeta casaba con ella:
        `productos` daba 120 líneas de las que 39 eran cabeceras."""
        out = grep(self.CATALOGO, "segador")

        assert out.splitlines()[0].startswith("0 líneas casan con 'segador'.")
        assert "=== productos/segador.md ===" not in out

    def test_listar_los_documentos_no_es_un_cero_falso(self) -> None:
        """`grep(ctx, '===')` es la forma natural de listar los documentos de un
        texto sin índice. Saltar la cabecera en silencio la vuelve un cero."""
        out = grep(self.CATALOGO, "^=== ")

        assert out == (
            "0 líneas casan con '^=== '. La cabecera `=== ruta ===` de 4 documentos casa con "
            "el patrón y no cuenta como contenido: productos/segador.md, productos/index.md, "
            "ontologia/objetivos.md, productos/bio-bpbs.md."
        )

    def test_la_ficha_que_solo_casa_por_la_ruta_se_nombra_aparte(self) -> None:
        """`bio-bpbs` con guion casa con la ruta y no con el texto de la ficha, que
        dice "Bio BPBS". Sin esto el modelo ve una línea suelta del índice y
        concluye que la ficha no existe."""
        out = grep(self.CATALOGO, "bio-bpbs")

        assert out.splitlines()[0] == (
            "1 línea casa con 'bio-bpbs' en productos/index.md. La cabecera `=== ruta ===` "
            "de 1 documento más casa con el patrón y no cuenta como contenido: "
            "productos/bio-bpbs.md."
        )

    def test_un_documento_con_lineas_no_se_repite_como_ruta(self) -> None:
        out = grep(self.CATALOGO, "segador|verde").splitlines()[0]

        assert out == "1 línea casa con 'segador|verde' en productos/segador.md."

    def test_la_lista_lleva_el_mismo_tope_que_las_lineas(self) -> None:
        texto = "".join(f"=== d{i}.md ===\nx\n" for i in range(10))

        out = grep(texto, "x", max_hits=3).splitlines()[0]

        assert out.startswith("10 líneas casan con 'x' en 10 documentos: d0.md (1), d1.md (1), d2.md (1) y 7 más;")
        assert out.endswith("Para el resto subí max_hits, afina el patrón o acota con doc=.")

    def test_con_doc_dice_cuantos_de_los_mirados(self) -> None:
        out = grep(self.CATALOGO, "pulgon", doc="productos/").splitlines()[0]

        assert out == (
            "2 líneas casan con 'pulgon' en 1 de los 3 documentos que casan con 'productos/': "
            "productos/segador.md (2)."
        )

    def test_sin_documentos_no_cambia_nada(self) -> None:
        assert grep("uno\ndos precio", "precio") == "1 línea casa con 'precio'.\n2: dos precio"


class TestPlegadoGuardado:
    """El caché del plegado. Es invisible para el modelo: la única forma de que se
    note es que una respuesta cambie, y eso es exactamente lo que se testea acá."""

    def setup_method(self) -> None:
        _PLEGADOS.clear()

    def test_la_segunda_llamada_da_lo_mismo_que_la_primera(self) -> None:
        texto = "=== fichas/plagas.md ===\nel pulgon come\notra cosa\nPULGÓN de nuevo"

        primera = grep(texto, "pulgon")

        assert grep(texto, "pulgon") == primera
        assert primera.startswith("2 líneas casan")

    def test_un_texto_distinto_no_hereda_el_plegado_de_otro(self) -> None:
        """Si la clave se reusara, el segundo grep contestaría sobre el primer texto."""
        uno = "araña roja"
        otro = "mosca blanca"

        assert grep(uno, "arana").startswith("1 línea casa")
        assert grep(otro, "arana").startswith("0 líneas")

    def test_exacto_no_paga_el_plegado(self) -> None:
        """Sobre el corpus de Go plegar es trabajo tirado, así que no se hace."""
        texto = "Plan\nplan"

        assert grep(texto, "Plan", exacto=True).startswith("1 línea casa")
        assert _PLEGADOS[id(texto)].campo is None

    def test_el_mismo_texto_plegado_despues_de_un_exacto(self) -> None:
        """Primero exacto y después no: el campo se calcula recién ahí, y bien."""
        texto = "Pulgón\nplan"

        grep(texto, "Pulgón", exacto=True)

        assert grep(texto, "pulgon").startswith("1 línea casa")
        assert _PLEGADOS[id(texto)].campo == ["Pulgon", "plan"]

    def test_no_guarda_mas_que_el_tope(self) -> None:
        """Acotado por PARALELO: un fanout entero cabe y el corpus no se duplica sin fin."""
        textos = [f"linea {i}" for i in range(MAX_PLEGADOS + 5)]

        for texto in textos:
            grep(texto, "linea")

        assert len(_PLEGADOS) == MAX_PLEGADOS
        assert id(textos[-1]) in _PLEGADOS
        assert id(textos[0]) not in _PLEGADOS
