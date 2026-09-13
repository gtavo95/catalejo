"""La aritmética del reporte de evals, que se mide sin gastar un peso.

Existe porque una suite de evals cuya cuenta no está testeada es el modo de falla
que la skill ya nombra: en `exp` un `price_min=0` del join fabricó cinco fallos
falsos sobre respuestas que daban el precio correcto. Un evaluador roto no se
nota, porque lo que reporta se parece a un resultado.
"""

from __future__ import annotations

import pytest

from catalejo.core import Fail, Log, Message, Role
from catalejo.core import Conversation
from catalejo.llm import Reply
from catalejo.rlm import HERRAMIENTAS, HERRAMIENTAS_UN_PASO, Contenedor, Workspace
from evals import (
    Caso,
    Corrida,
    acierta,
    montaje,
    repeticiones,
    resumir,
    wiki,
)

UNO = Caso("uno", "¿?", ("productos/x.md",), "# Resumen", "no inventar")
DOS = Caso("dos", "¿?", ("productos/y.md",), "# Resumen", "no inventar")

VACIO = Fail("model", "GeminiError: el modelo contestó vacío")
SIN_PASOS = Fail("loop", "tope de pasos: 12")


def corrida(
    caso: Caso = UNO,
    *,
    vuelta: int = 0,
    ok: bool = True,
    fails: tuple[Fail, ...] = (),
    spent: int = 100,
    turnos: int = 4,
    fantasmas: tuple[str, ...] = (),
) -> Corrida:
    said = tuple(Message(Role.ASSISTANT, "hola") for _ in range(turnos))
    return Corrida(
        caso=caso,
        vuelta=vuelta,
        out=Log(said=said, fails=fails, spent=spent),
        respuesta="",
        seg=1.0,
        ok=ok,
        fantasmas=fantasmas,
    )


class TestResumir:
    def test_acertar_las_tres_es_estable(self) -> None:
        r = resumir([corrida(vuelta=i) for i in range(3)], 3)

        assert r.estables == ("uno",)
        assert r.flippers == ()
        assert r.aciertos == 3

    def test_acertar_algunas_es_flipper(self) -> None:
        """Es lo único que la escalera de n compra: el caso que hay que profundizar."""
        corridas = [corrida(vuelta=0), corrida(vuelta=1, ok=False), corrida(vuelta=2)]

        r = resumir(corridas, 3)

        assert r.flippers == ("uno",)
        assert r.estables == ()
        assert r.aciertos == 2

    def test_fallar_siempre_no_es_flipear(self) -> None:
        r = resumir([corrida(vuelta=i, ok=False) for i in range(3)], 3)

        assert r.caidos == ("uno",)
        assert r.flippers == ()

    def test_un_turno_vacio_sale_del_denominador(self) -> None:
        """Una muestra perdida no es un fallo de calidad.

        Contarla como fallo sesga la comparación a favor del arm que tuvo suerte
        con la API, que es lo contrario de lo que se quiere medir.
        """
        corridas = [corrida(vuelta=0), corrida(vuelta=1, ok=False, fails=(VACIO,)), corrida(vuelta=2)]

        r = resumir(corridas, 3)

        assert r.perdidas == 1
        assert r.vivas == 2
        assert r.aciertos == 2
        assert r.estables == ("uno",)

    def test_perderlas_todas_es_mudo_y_no_caido(self) -> None:
        """Sin una muestra viva no hay nada que afirmar, ni bueno ni malo."""
        r = resumir([corrida(vuelta=i, ok=False, fails=(VACIO,)) for i in range(3)], 3)

        assert r.mudos == ("uno",)
        assert r.caidos == ()
        assert r.vivas == 0

    def test_quedarse_sin_pasos_si_cuenta(self) -> None:
        """`loop` y `grounding` fallan por resultados, no por la API.

        Quedarse sin pasos es no haber llegado, y contestar sin mirar es el modo
        de falla que persigue todo el repo. Los dos son calidad.
        """
        r = resumir([corrida(ok=False, fails=(SIN_PASOS,))], 1)

        assert r.perdidas == 0
        assert r.caidos == ("uno",)

    def test_el_gasto_medio_va_sobre_las_vivas_y_el_total_sobre_todas(self) -> None:
        """La muestra perdida se paga igual, así que el total la incluye."""
        corridas = [corrida(spent=100), corrida(vuelta=1, spent=900, fails=(VACIO,))]

        r = resumir(corridas, 2)

        assert r.spent == 100
        assert r.gastado == 1000

    def test_los_turnos_promedian_las_vivas(self) -> None:
        r = resumir([corrida(turnos=4), corrida(vuelta=1, turnos=6)], 2)

        assert r.turnos == 5.0

    def test_cuenta_los_avisos_de_cada_celula_y_no_la_salida_del_repl(self) -> None:
        """Si el arm con una célula gasta más, hay que poder ver si la célula habló."""
        said = (
            Message(Role.ASSISTANT, "de memoria"),
            Message(Role.USER, "[grounding] Todavía no ejecutaste nada"),
            Message(Role.ASSISTANT, "```python\nprint(1)\n```"),
            Message(Role.USER, "[repl] salida:\n1"),
            Message(Role.ASSISTANT, "FUENTE: productos/no.md"),
            Message(Role.USER, "[cita] Citaste `productos/no.md`"),
        )
        c = Corrida(caso=UNO, vuelta=0, out=Log(said=said), respuesta="", seg=1.0, ok=True)

        r = resumir([c, corrida(vuelta=1)], 2)

        assert r.avisos == {"grounding": 1, "cita": 1}

    def test_las_rutas_fantasma_se_juntan_entre_vueltas(self) -> None:
        """Una ruta inventada en la vuelta 2 cuenta aunque la 1 haya salido limpia."""
        corridas = [corrida(vuelta=0), corrida(vuelta=1, fantasmas=("productos/no.md",))]

        r = resumir(corridas, 2)

        assert r.fantasmas == {"uno": ("productos/no.md",)}

    def test_cuenta_casos_y_no_corridas(self) -> None:
        corridas = [corrida(UNO), corrida(UNO, vuelta=1), corrida(DOS), corrida(DOS, vuelta=1)]

        r = resumir(corridas, 2)

        assert r.casos == 2
        assert r.vivas == 4

    def test_sin_corridas_no_divide_por_cero(self) -> None:
        r = resumir([], 3)

        assert r.casos == 0
        assert r.spent == 0


class TestAcierta:
    def test_pide_la_pagina_esperada(self) -> None:
        assert acierta(UNO, "está en productos/x.md")
        assert not acierta(UNO, "está en productos/z.md")

    def test_sin_paginas_el_acierto_es_decir_que_no_hay(self) -> None:
        vacio = Caso("cero", "¿?", (), "·", "no rellenar")

        assert acierta(vacio, "no está en el catálogo.\nFUENTE: ninguna")
        assert not acierta(vacio, "usá productos/inventado.md")


class TestRepeticiones:
    def test_sin_bandera_es_una(self) -> None:
        assert repeticiones(["--agro", "un-caso"]) == 1

    def test_con_igual(self) -> None:
        assert repeticiones(["--repeats=6"]) == 6

    def test_con_espacio_no_existe_y_por_eso_va_con_igual(self) -> None:
        """`--repeats 3` dejaría `3` como id de caso, que es el bug que evita el `=`."""
        assert repeticiones(["--repeats", "3"]) == 1

    def test_una_basura_lo_dice_en_vez_de_reventar_raro(self) -> None:
        with pytest.raises(SystemExit, match="entero"):
            repeticiones(["--repeats=cero"])

    def test_cero_no_es_una_corrida(self) -> None:
        with pytest.raises(SystemExit):
            repeticiones(["--repeats=0"])


class Mudo:
    """Un `Provider` que nunca se llama: armar el montaje no habla con nadie."""

    model = "mudo"

    async def complete(self, conv: Conversation) -> Reply:
        raise AssertionError("armar el agente no debería llamar al modelo")

    async def aclose(self) -> None:
        pass


class TestMontaje:
    def test_contenedor_es_el_montaje_de_la_wiki(self) -> None:
        assert montaje(["--contenedor"]).tsv == "preguntas.tsv"

    async def test_contenedor_con_recurse_tambien_da_un_proceso_hijo(self) -> None:
        """`llm` se queda en el padre y el hijo lo llama por el pipe."""
        _, ws = wiki("x", Mudo(), recursivo=True, contenedor=True)
        try:
            assert isinstance(ws, Contenedor)
            assert "llm(" in ws.tools
            assert ws.bridge is not None
        finally:
            ws.cerrar()

    async def test_la_wiki_con_contenedor_arma_un_proceso_hijo_con_el_mismo_handle(self) -> None:
        """Lo que el modelo ve no cambia: mismo `var`, mismas herramientas en el preámbulo."""
        _, ws = wiki("=== a.md ===\nhola", Mudo(), recursivo=False, contenedor=True)
        assert isinstance(ws, Contenedor)
        try:
            assert ws.var == "wiki"
            assert ws.tools == Workspace("x", var="wiki").tools
            assert (await ws.run("print(grep(wiki, 'hola'))")).stdout.startswith("1 línea")
        finally:
            ws.cerrar()

    def test_un_paso_entra_en_los_dos_montajes(self) -> None:
        assert montaje(["--un-paso"]).tsv == "preguntas.tsv"
        assert montaje(["--agro", "--un-paso"]).tsv == "agro.tsv"

    async def test_la_wiki_busca_y_lee_por_default(self) -> None:
        """Es lo que el modelo ve: `grep` dice dónde y `read` existe."""
        _, ws = wiki("=== a.md ===\nhola", Mudo(), recursivo=False)

        assert ws.tools == HERRAMIENTAS
        assert (await ws.run("print(grep(wiki, 'hola'))")).stdout == "1 línea casa con 'hola' en a.md.\n"
        assert (await ws.run("print(read(wiki, 'a'))")).stdout.endswith("\nhola\n")

    async def test_un_paso_es_el_grep_de_antes(self) -> None:
        """El baseline de `dos_pasos` tiene que seguir existiendo tal cual, para volver a medir."""
        _, ws = wiki("=== a.md ===\nhola", Mudo(), recursivo=False, dos_pasos=False)

        assert ws.tools == HERRAMIENTAS_UN_PASO
        assert (await ws.run("print(grep(wiki, 'hola'))")).stdout == "1 línea casa con 'hola' en a.md.\na.md:2: hola\n"

    def test_sin_la_bandera_la_wiki_sigue_en_proceso(self) -> None:
        _, ws = wiki("x", Mudo(), recursivo=False)

        assert isinstance(ws, Workspace)
