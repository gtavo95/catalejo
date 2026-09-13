import pytest

from catalejo.core import PlanOp
from catalejo.rlm import HERRAMIENTAS, Bridge, Contenedor, Verbos, Workspace


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

    async def test_las_movidas_del_plan_apilan_en_el_verbos_del_padre(self) -> None:
        """`add_step` es un bound method, así que no cruza: el hijo lo llama por el
        pipe y apila acá, y el planner lo lee por `tomar()` sin enterarse."""
        v = Verbos()
        c = Contenedor("x", extra=v.builtins)
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

    async def test_una_funcion_del_extra_corre_en_el_padre(self) -> None:
        """La closure con estado se queda acá; el hijo tiene un stub con su nombre."""
        vistas: list[str] = []

        def anota(s: str) -> str:
            vistas.append(s)
            return s.upper()

        c = Contenedor("x", extra={"anota": anota})
        try:
            out = await c.run("print(anota('hola'), anota(s='chau'))")

            assert out.stdout == "HOLA CHAU\n"
            assert vistas == ["hola", "chau"]
        finally:
            c.cerrar()

    async def test_un_error_del_padre_vuelve_con_su_nombre(self) -> None:
        """El modelo lee `ValueError: ...`, igual que si corriera en proceso."""

        def rompe(s: str) -> str:
            raise ValueError(f"no sé qué hacer con {s!r}")

        c = Contenedor("x", extra={"rompe": rompe})
        try:
            out = await c.run("rompe('esto')")

            assert out.err == "ValueError: no sé qué hacer con 'esto'"
        finally:
            c.cerrar()

    async def test_lo_que_no_se_puede_picklear_de_vuelta_es_un_error_del_builtin(self) -> None:
        c = Contenedor("x", extra={"raro": lambda: (yield)})
        try:
            out = await c.run("raro()")

            assert out.err.startswith("TypeError: cannot pickle")
        finally:
            c.cerrar()

    async def test_el_bridge_mide_lo_que_gasta_un_remoto(self) -> None:
        """Igual que `Workspace.run`: se le da el loop y se mide la diferencia."""
        b = Bridge(budget=100)

        def cobra() -> str:
            b.spent += 7
            return "pagado"

        c = Contenedor("x", extra={"cobra": cobra}, bridge=b)
        try:
            out = await c.run("print(cobra())")

            assert out.stdout == "pagado\n"
            assert out.spent == 7
            assert (await c.run("pass")).spent == 0
        finally:
            c.cerrar()

    async def test_un_dato_que_no_se_picklea_es_un_error_al_armar(self) -> None:
        with pytest.raises(TypeError, match="`raro` no se puede picklear"):
            Contenedor("x", extra={"raro": (x for x in "ab")})

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

    async def test_dos_pasos_cruza_al_hijo(self) -> None:
        """El hijo arma su Workspace con el mismo default: el `grep` que solo dice
        dónde, y `read`. La nota se calcula acá, sin preguntarle."""
        texto = "=== a.md ===\nprecio 1\n=== b.md ===\nprecio 2"
        c = Contenedor(texto)
        try:
            donde = await c.run("print(grep(ctx, 'precio'))")
            ficha = await c.run("print(read(ctx, 'b'))")

            assert donde.stdout == "2 líneas casan con 'precio' en 2 documentos: a.md (1), b.md (1).\n"
            assert ficha.stdout == "b.md: 1 línea, de la 4 a la 4.\nprecio 2\n"
            assert c.tools == HERRAMIENTAS
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
            out = await c.run("print(grep(ctx, 'pulgon', doc='plagas'))")

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
