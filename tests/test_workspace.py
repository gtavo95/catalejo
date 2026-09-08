import asyncio

from catalejo.repl import Workspace, grep


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

    async def test_esta_en_el_namespace_del_modelo(self) -> None:
        ws = Workspace("uno\ndos precio")

        out = await ws.run("print(grep(ctx, 'precio'))")

        assert out.stdout == "1 línea casa con 'precio'.\n2: dos precio\n"
