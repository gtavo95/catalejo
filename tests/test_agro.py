"""Lo que el asesor sirve como respuesta, que no siempre es lo que el modelo dijo."""

from catalejo.core import Fail, Log, Message, Role

from agro import SIN_CONSULTAR, final

SIN_FUNDAMENTO = Fail("grounding", "contestó sin haber consultado el contexto")
SIN_PASOS = Fail("loop", "tope de pasos: 12")


def dicho(texto: str) -> Message:
    return Message(Role.ASSISTANT, texto)


class TestFinal:
    def test_la_respuesta_normal_sale_tal_cual(self) -> None:
        out = Log(said=(Message(Role.USER, "¿dosis?"), dicho("1.0 L/Ha\nFUENTE: productos/x.md")))

        assert final(out) == "1.0 L/Ha\nFUENTE: productos/x.md"

    def test_el_ultimo_bloque_de_codigo_no_es_respuesta(self) -> None:
        out = Log(said=(dicho("ya veo"), dicho("```python\nprint(1)\n```")))

        assert final(out) == "ya veo"

    def test_sin_consultar_no_se_sirve_lo_que_dijo(self) -> None:
        """El turno que destapó esto fue el modelo deliberando en voz alta.

        `grounded` avisa una vez y, si el modelo lo ignora, deja salir la respuesta
        etiquetada. La etiqueta se ve en el CLI y no se ve detrás de una API, así
        que el corte va acá.
        """
        crudo = "Respond? We need code block only to execute. But final instruction says..."
        out = Log(said=(dicho(crudo),), fails=(SIN_FUNDAMENTO,))

        assert final(out) == SIN_CONSULTAR

    def test_el_dicho_crudo_sigue_en_el_log(self) -> None:
        """Cambia lo que se sirve, no lo que se guarda: si no, no hay qué auditar."""
        out = Log(said=(dicho("contesté de memoria"),), fails=(SIN_FUNDAMENTO,))

        final(out)

        assert out.said[-1].text == "contesté de memoria"

    def test_quedarse_sin_pasos_no_es_lo_mismo(self) -> None:
        """Esa respuesta SÍ consultó el catálogo, solo que se cortó. Sale, y sale entera."""
        out = Log(said=(dicho("Metaveria 40 EW, 1.0 L/Ha"),), fails=(SIN_PASOS,))

        assert final(out) == "Metaveria 40 EW, 1.0 L/Ha"

    def test_sin_prosa_lo_dice(self) -> None:
        out = Log(said=(dicho("```python\nprint(1)\n```"),))

        assert final(out) == "(se quedó sin pasos antes de contestar)"
