import pytest

from catalejo.core import PlanOp
from catalejo.repl import HERRAMIENTAS, Contenedor, Verbos, Workspace


class TestContenedor:
    async def test_lo_impreso_cruza_y_las_variables_persisten(self) -> None:
        """Es un Workspace: lo que el modelo guarda en un paso está en el siguiente."""
        c = Contenedor("hola mundo")
        try:
            await c.run("notas = [ctx.upper()]")
            out = await c.run("print(notas)")

            assert out.stdout == "['HOLA MUNDO']\n"
            assert out.err == ""
        finally:
            c.cerrar()

    async def test_un_error_vuelve_en_err_y_no_levanta(self) -> None:
        c = Contenedor()
        try:
            out = await c.run("print(no_existe)")

            assert out.err == "NameError: name 'no_existe' is not defined"
            assert out.stdout == ""
        finally:
            c.cerrar()

    async def test_el_cuelgue_se_mata_y_el_workspace_arranca_de_nuevo(self) -> None:
        """Lo que un hilo no puede: un `while True` vuelve como error que enseña, el
        proceso se relanza, y el siguiente `run` funciona. Las variables se pierden,
        y el aviso lo dice."""
        c = Contenedor("texto", timeout=0.5)
        try:
            await c.run("guardado = 1")
            colgado = await c.run("while True: pass")
            despues = await c.run("print(ctx); print(guardado)")

            assert "no terminó en 0.5 s" in colgado.err
            assert "se perdieron" in colgado.err
            assert colgado.stdout == ""
            assert despues.stdout == "texto\n"
            assert despues.err == "NameError: name 'guardado' is not defined"
        finally:
            c.cerrar()

    async def test_las_movidas_del_plan_vuelven_al_verbos_del_padre(self) -> None:
        """`add_step` apila en el hijo y el planner lo lee acá, sin enterarse."""
        v = Verbos()
        c = Contenedor("x", verbos=v)
        try:
            await c.run("add_step('leer', 'mirar la ficha'); mark('leer', 'active')")

            assert v.tomar() == (
                PlanOp("add_step", "leer", intent="mirar la ficha"),
                PlanOp("mark", "leer", status="active"),
            )
            assert v.tomar() == ()
        finally:
            c.cerrar()

    async def test_sin_verbos_no_hay_verbos(self) -> None:
        c = Contenedor("x")
        try:
            out = await c.run("add_step('a', 'b')")

            assert out.err.startswith("NameError")
        finally:
            c.cerrar()

    async def test_el_extra_cruza_si_es_dato(self) -> None:
        c = Contenedor("x", extra={"productos": ["viventem", "royano"]})
        try:
            out = await c.run("print(sorted(productos))")

            assert out.stdout == "['royano', 'viventem']\n"
        finally:
            c.cerrar()

    async def test_el_extra_no_cruza_si_es_funcion(self) -> None:
        """Un bound method de un objeto picklable cruzaría como copia y las llamadas
        se perderían en silencio. Peor que un error, así que es un error."""
        v = Verbos()

        with pytest.raises(TypeError, match="add_step.*verbos="):
            Contenedor("x", extra=v.builtins)

    async def test_var_y_tools_como_el_workspace(self) -> None:
        c = Contenedor("x", var="wiki", note="hay un índice")
        try:
            ws = Workspace("x", var="wiki", note="hay un índice")

            assert c.var == "wiki"
            assert c.tools == ws.tools
            assert c.tools.startswith(HERRAMIENTAS)
            assert (await c.run("print(wiki)")).stdout == "x\n"
        finally:
            c.cerrar()

    async def test_cerrar_termina_el_proceso(self) -> None:
        c = Contenedor("x")
        assert c.vivo

        c.cerrar()

        assert not c.vivo

    async def test_grep_funciona_adentro_del_hijo(self) -> None:
        c = Contenedor("=== fichas/plagas.md ===\nel pulgon come\notra cosa\nPULGÓN de nuevo")
        try:
            out = await c.run("print(grep(ctx, 'pulgon'))")

            assert out.stdout.startswith("2 líneas casan con 'pulgon' en fichas/plagas.md.")
            assert "fichas/plagas.md:2: el pulgon come" in out.stdout
        finally:
            c.cerrar()

    async def test_el_hijo_no_tiene_import(self) -> None:
        """El proceso protege la máquina; el namespace sigue moldeando lo que el
        modelo escribe. Las dos cosas a la vez."""
        c = Contenedor("x")
        try:
            out = await c.run("import re")

            assert out.err.startswith("ImportError: no hay `import` en este REPL")
        finally:
            c.cerrar()
