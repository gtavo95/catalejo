from catalejo.core import ZERO, Cell, Fail, Log, Message, Role, Status, loop, merge, then
from catalejo.llm import Stub
from catalejo.repl import Handle, Workspace, citada, citadas, executor, inventada, rutas, worker
from catalejo.repl.citas import PREFIJO

CORPUS = "=== productos/si.md ===\ndosis: 1 L/ha\n\n=== ontologia/b.md ===\npulgón"
REALES = rutas(CORPUS)


def dicho(texto: str, role: Role = Role.ASSISTANT) -> Message:
    return Message(role, texto)


def bloque(code: str) -> str:
    return f"```python\n{code}\n```"


class TestCitadas:
    def test_saca_todas_las_fuentes(self) -> None:
        """El CONTRATO de agro pide una línea por archivo, así que son varias."""
        texto = "la dosis es 1 L/Mz\nFUENTE: productos/a.md\nFUENTE: productos/b.md"

        assert citadas(texto) == ("productos/a.md", "productos/b.md")

    def test_le_saca_los_backticks_y_la_barra(self) -> None:
        assert citadas("FUENTE: `/productos/a.md`") == ("productos/a.md",)

    def test_sin_fuente_no_hay_ninguna(self) -> None:
        assert citadas("contesté de memoria") == ()


class TestInventada:
    def test_la_ruta_que_no_existe(self) -> None:
        assert inventada("FUENTE: productos/no.md", {"productos/si.md"}) == ("productos/no.md",)

    def test_la_fabricada_en_la_primera_linea_no_se_escapa(self) -> None:
        """El bug que destapó el rig de agro: mirar solo la última fuente.

        Con dos citas y la mala adelante, quedarse con la última daba limpio.
        """
        texto = "FUENTE: productos/no.md\nFUENTE: productos/si.md"

        assert inventada(texto, {"productos/si.md"}) == ("productos/no.md",)

    def test_ninguna_no_es_una_ruta(self) -> None:
        assert inventada("FUENTE: ninguna", {"productos/si.md"}) == ()

    def test_la_que_existe_no_es_fantasma(self) -> None:
        assert inventada("FUENTE: productos/si.md", {"productos/si.md"}) == ()

    def test_una_mayuscula_o_un_acento_no_es_una_ficha_inventada(self) -> None:
        """Los archivos son slugs en minúscula y el modelo escribe `Viventem.md` a
        veces. Eso es un error de tipeo, no una fuente fabricada."""
        reales = {"productos/viventem.md", "ontologia/pulgon.md"}

        assert inventada("FUENTE: productos/Viventem.md", reales) == ()
        assert inventada("FUENTE: ontologia/pulgón.md", reales) == ()


class TestRutas:
    def test_salen_de_las_cabeceras_del_corpus(self) -> None:
        """Y no de un segundo recorrido del bundle: es lo que el modelo pudo leer.

        Con `--agro` el corpus son dos carpetas, así que citar un archivo que
        existe en el bundle y no en el corpus es inventar igual.
        """
        texto = "=== productos/a.md ===\nhola\n\n=== ontologia/b.md ===\nchau"

        assert rutas(texto) == {"productos/a.md", "ontologia/b.md"}

    def test_una_linea_que_no_es_cabecera_no_entra(self) -> None:
        assert rutas("=== a.md ===\ntexto === con === iguales") == {"a.md"}


class TestCitada:
    async def test_no_opina_mientras_escribe_codigo(self) -> None:
        seen = Log(said=(dicho(bloque("print(grep(ctx, 'dosis'))")),))

        assert await citada(REALES)(seen) == ZERO

    async def test_no_opina_si_el_ultimo_no_es_del_modelo(self) -> None:
        seen = Log(said=(dicho("[repl] salida:\n3", Role.USER),))

        assert await citada(REALES)(seen) == ZERO

    async def test_no_opina_sobre_un_log_vacio(self) -> None:
        assert await citada(REALES)(ZERO) == ZERO

    async def test_una_cita_que_existe_pasa(self) -> None:
        seen = Log(said=(dicho("1 L/ha\nFUENTE: productos/si.md"),))

        assert await citada(REALES)(seen) == ZERO

    async def test_ninguna_pasa(self) -> None:
        seen = Log(said=(dicho("el catálogo no lo cubre\nFUENTE: ninguna"),))

        assert await citada(REALES)(seen) == ZERO

    async def test_sin_fuente_no_es_asunto_de_esta_celula(self) -> None:
        """Que falte la línea lo mira la compuerta `cito_la_fuente` del plan. Esta
        célula juzga lo que se citó, no lo que no."""
        seen = Log(said=(dicho("1 L/ha, de memoria"),))

        assert await citada(REALES)(seen) == ZERO

    async def test_veta_la_cita_inventada_y_nombra_las_rutas(self) -> None:
        """El caso zompopo: consultó, no encontró, y cerró con tres fichas que no
        existen. `grounded` lo deja pasar porque leyó; esto no."""
        seen = Log(
            said=(
                dicho(
                    "usá Beaveria 90\nFUENTE: productos/beaveria-90.md\n"
                    "FUENTE: productos/isaria-forte.md"
                ),
            ),
            reads=2,
        )

        out = await citada(REALES, var="catalogo")(seen)

        assert out.vote is Status.CONTINUE
        assert len(out.said) == 1
        aviso = out.said[0]
        assert aviso.role is Role.USER
        assert aviso.text.startswith(PREFIJO)
        assert "`productos/beaveria-90.md`" in aviso.text
        assert "`productos/isaria-forte.md`" in aviso.text
        assert "ninguna existe en `catalogo`" in aviso.text
        assert "FUENTE: ninguna" in aviso.text

    async def test_con_una_sola_ruta_habla_en_singular(self) -> None:
        out = await citada(REALES)(Log(said=(dicho("FUENTE: productos/no.md"),)))

        assert "`productos/no.md`, y no existe en `ctx`" in out.said[0].text

    async def test_el_veto_le_gana_al_done_del_executor(self) -> None:
        del_executor = Log(vote=Status.DONE)
        del_citas = await citada(REALES)(Log(said=(dicho("FUENTE: productos/no.md"),)))

        assert merge(del_executor, del_citas).vote is Status.CONTINUE

    async def test_avisa_una_sola_vez_y_despues_lo_deja_salir_etiquetado(self) -> None:
        primero = await citada(REALES)(Log(said=(dicho("FUENTE: productos/no.md"),)))
        seen = Log(said=(dicho("FUENTE: productos/no.md"), *primero.said, dicho("FUENTE: productos/no.md")))

        out = await citada(REALES)(seen)

        assert out.fails == (Fail("cita", "citó rutas que no existen: productos/no.md"),)
        assert out.vote is Status.QUIET
        assert out.said == ()

    async def test_corregir_despues_del_aviso_pasa_limpio(self) -> None:
        primero = await citada(REALES)(Log(said=(dicho("FUENTE: productos/no.md"),)))
        seen = Log(said=(dicho("FUENTE: productos/no.md"), *primero.said, dicho("FUENTE: ninguna")))

        assert await citada(REALES)(seen) == ZERO


class TestElAgenteEntero:
    """Lo que `grounded` deja pasar y esto no: el modelo consultó, y citó igual una
    ficha que no existe."""

    def agente(self, model: Stub, ws: Workspace) -> Cell:
        return loop(
            then(worker(model, Handle(tools=ws.tools)), executor(ws), citada(REALES)),
            max_steps=6,
        )

    async def test_la_cita_inventada_vuelve_al_modelo_y_sale_bien(self) -> None:
        model = Stub(
            bloque("print(grep(ctx, 'zompopo'))"),
            "usá Isaria\nFUENTE: productos/isaria-forte.md",
            "el catálogo no cubre el zompopo\nFUENTE: ninguna",
        )
        ws = Workspace(CORPUS)

        out = await self.agente(model, ws)(Log(said=(dicho("¿para el zompopo?", Role.USER),)))

        assert out.vote is Status.DONE
        assert out.fails == ()
        assert out.said[-1].text.endswith("FUENTE: ninguna")
        assert any(m.text.startswith(PREFIJO) for m in out.said)

    async def test_si_insiste_sale_con_el_fail(self) -> None:
        model = Stub(
            bloque("print(grep(ctx, 'zompopo'))"),
            "usá Isaria\nFUENTE: productos/isaria-forte.md",
            "insisto\nFUENTE: productos/isaria-forte.md",
        )
        ws = Workspace(CORPUS)

        out = await self.agente(model, ws)(Log(said=(dicho("¿para el zompopo?", Role.USER),)))

        assert out.fails == (Fail("cita", "citó rutas que no existen: productos/isaria-forte.md"),)
