"""Los builtins del plan: apilar sin romper el snippet que los llamó."""

from catalejo.core import PlanOp
from catalejo.repl import Verbos, Workspace


class TestColector:
    def test_apila_los_tres_verbos(self) -> None:
        v = Verbos()

        v.add_step("buscar", "encontrar el producto", completes_when="leyo")
        v.mark("buscar", "active")
        v.skip("otro")

        assert v.tomar() == (
            PlanOp("add_step", "buscar", "encontrar el producto", completes_when="leyo"),
            PlanOp("mark", "buscar", status="active"),
            PlanOp("skip", "otro"),
        )

    def test_drenar_deja_el_colector_vacio(self) -> None:
        """Si devolviera el historial, cada paso volvería a proponer todo lo de
        los pasos anteriores y `admitir` los rechazaría uno por uno."""
        v = Verbos()
        v.mark("a", "done")

        assert len(v.tomar()) == 1
        assert v.tomar() == ()

    def test_lo_que_no_es_texto_entra_como_texto(self) -> None:
        v = Verbos()

        v.mark(3, None)

        assert v.tomar() == (PlanOp("mark", "3", status="None"),)

    def test_una_llamada_sin_argumentos_apila_un_op_que_no_aplica(self) -> None:
        """Apila igual, y lo rechaza `admitir` con un motivo que el modelo lee.
        Levantar acá se comería el resto del snippet."""
        v = Verbos()

        v.add_step()

        assert v.tomar() == (PlanOp("add_step", ""),)


class TestEnElRepl:
    async def test_el_snippet_sigue_despues_de_llamar_un_verbo(self) -> None:
        v = Verbos()
        ws = Workspace("hola", extra=v.builtins)

        out = await ws.run('mark("buscar", "active")\nprint(len(ctx))')

        assert out.err == ""
        assert out.stdout == "4\n"
        assert v.tomar() == (PlanOp("mark", "buscar", status="active"),)

    async def test_un_verbo_llamado_con_basura_no_mata_el_resto(self) -> None:
        """Es el motivo entero de que estos builtins no levanten: el resto del
        snippet es la consulta, y perderla cuesta un paso del turno."""
        v = Verbos()
        ws = Workspace("hola", extra=v.builtins)

        out = await ws.run('mark(None)\nprint(grep(ctx, "hol"))')

        assert out.err == ""
        assert "1 línea casa" in out.stdout

    async def test_un_verbo_que_no_existe_es_un_nameerror(self) -> None:
        """El vocabulario lo cierra el namespace, así que nadie lo valida."""
        v = Verbos()
        ws = Workspace("hola", extra=v.builtins)

        out = await ws.run('revise("buscar", "otra cosa")')

        assert "NameError" in out.err
        assert v.tomar() == ()
