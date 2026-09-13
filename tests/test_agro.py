"""Lo que el asesor sirve como respuesta, que no siempre es lo que el modelo dijo."""

from catalejo.core import Fail, Log, Message, Role

from agro import (
    BUSCAR,
    BUSCAR_EN_OBJETIVOS,
    SIN_CONSULTAR,
    contrato,
    final,
    indice,
    objetivos,
    pedido,
    vocabulario,
)

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


HOJA = """---
type: ontologia
eje: objetivos
---

La prosa de arriba explica el eje | y puede traer barras | sin ser tabla.

| id | etiqueta | padre | alias | nota |
|---|---|---|---|---|
| chupadores | Chupadores |  | chupador | agrupación nuestra |
| pulgon | Pulgón | chupadores | pulgones, afido, afidos, aphididae |  |
| myzus | Myzus | pulgon | myzus spp. | la ficha no baja de género |

Después de la tabla vuelve la prosa.

| id | etiqueta |
|---|---|
| basura | No es vocabulario |
"""


class TestVocabulario:
    def test_una_entrada_por_fila_con_las_columnas_del_encabezado(self) -> None:
        entradas = vocabulario(HOJA)

        assert [e["id"] for e in entradas] == ["chupadores", "pulgon", "myzus"]
        assert entradas[1] == {
            "id": "pulgon",
            "etiqueta": "Pulgón",
            "padre": "chupadores",
            "alias": ["pulgones", "afido", "afidos", "aphididae"],
            "nota": "",
        }

    def test_la_raiz_tiene_padre_vacio_y_no_ausente(self) -> None:
        """Como lo escribe la hoja: vacío quiere decir raíz, y el modelo filtra por `== ""`."""
        assert vocabulario(HOJA)[0]["padre"] == ""

    def test_el_alias_es_lista_y_la_nota_texto(self) -> None:
        myzus = vocabulario(HOJA)[2]

        assert myzus["alias"] == ["myzus spp."]
        assert myzus["nota"] == "la ficha no baja de género"

    def test_solo_la_primera_tabla_es_vocabulario(self) -> None:
        """`FORMATO.md`: el lector consume la primera tabla que abre con `| id |`.

        Una segunda tabla, o una barra en la prosa, no son conceptos.
        """
        assert all(e["id"] != "basura" for e in vocabulario(HOJA))

    def test_sin_tabla_no_hay_entradas(self) -> None:
        assert vocabulario("solo prosa\n| no | es | la | tabla |") == []


class TestObjetivos:
    """Sobre el bundle de verdad, porque el join es contra las fichas de verdad."""

    def test_cada_concepto_trae_las_fichas_que_lo_cubren(self) -> None:
        por_id = {e["id"]: e for e in objetivos(indice())}

        assert por_id["rata-alcantarilla"]["fichas"] == ["productos/rodenticida-0-75-cb.md"]
        assert por_id["gusano-cogollero"]["fichas"] == ["productos/novermo-45-sc.md"]

    def test_el_padre_cubre_lo_que_cubren_sus_hijos(self) -> None:
        """La regla de FORMATO.md: una consulta por pulgón alcanza las tres especies."""
        por_id = {e["id"]: e for e in objetivos(indice())}
        hijas = {r for h in ("aphis-gossypii", "myzus", "brevicoryne-brassicae") for r in por_id[h]["fichas"]}

        assert hijas
        assert hijas <= set(por_id["pulgon"]["fichas"])
        assert set(por_id["roedores"]["fichas"]) == {"productos/rodenticida-0-75-cb.md"}

    def test_las_fichas_son_rutas_que_existen_en_el_corpus(self) -> None:
        reales = {p["ruta"] for p in indice()}
        for e in objetivos(indice()):
            assert set(e["fichas"]) <= reales


class TestContrato:
    def test_por_default_manda_a_la_hoja(self) -> None:
        assert "`ontologia/objetivos.md` tiene los alias" in contrato()
        assert "`objetivos`" not in contrato()

    def test_con_ontologia_cambia_esa_vineta_y_solo_esa(self) -> None:
        con = contrato(ontologia=True)

        assert "`ontologia/objetivos.md` tiene los alias" not in con
        assert "En el REPL tenés `objetivos`" in con
        assert con.replace(BUSCAR_EN_OBJETIVOS, BUSCAR) == contrato()

    def test_el_pedido_lleva_el_contrato_que_le_piden(self) -> None:
        assert "`objetivos`" not in pedido("¿zompopo?", ())
        assert "En el REPL tenés `objetivos`" in pedido("¿zompopo?", (), ontologia=True)
