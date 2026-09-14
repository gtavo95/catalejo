"""Lo que el asesor sirve como respuesta, que no siempre es lo que el modelo dijo."""

from catalejo.core import Fail, Log, Message, Role

from agro import (
    BUNDLE,
    BUSCAR,
    BUSCAR_EN_OBJETIVOS,
    SIN_CONSULTAR,
    claves,
    contrato,
    final,
    indice,
    objetivos,
    pedido,
    resolver,
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


CULTIVOS = """| id | etiqueta | padre | alias | nota |
|---|---|---|---|---|
| calabacita | Calabacita | cucurbitaceas | zucchini, calabacín |  |
| guicoy | Güicoy | cucurbitaceas | güicoy, ayote |  |
| cafe | Café | industriales | cafeto |  |
"""


class TestResolver:
    """Los cultivos del frontmatter, como los escribe la ficha, a su id por la hoja."""

    def test_id_etiqueta_y_alias_resuelven_plegando_caso_y_acentos(self) -> None:
        hoja = claves(vocabulario(CULTIVOS))

        assert resolver(["Zucchini", "GÜICOY", "Cafe", "calabacita"], hoja) == ["calabacita", "guicoy", "cafe"]

    def test_un_termino_desconocido_se_conserva_plegado(self) -> None:
        hoja = claves(vocabulario(CULTIVOS))

        assert resolver(["Café", "Papayo"], hoja) == ["cafe", "papayo"]

    def test_un_escalar_no_es_una_lista(self) -> None:
        hoja = claves(vocabulario(CULTIVOS))

        assert resolver("all_crops", hoja) == []
        assert resolver("not_applicable", hoja) == []


class TestIndice:
    """Sobre el bundle de verdad: desde el 13 de septiembre de 2026 el perfil ya no
    trae `crops`, así que `cultivos` sale de resolver `certified_crops`, y tiene que
    dar lo mismo que daba `crops` antes: ids de la hoja, uno por término."""

    def test_toda_lista_certificada_resuelve_a_ids_de_la_hoja(self) -> None:
        hoja = {e["id"] for e in vocabulario((BUNDLE / "ontologia" / "cultivos.md").read_text())}
        con_lista = [r for r in indice() if isinstance(r["certificados"], list)]

        assert con_lista
        for r in con_lista:
            assert r["cultivos"], r["nombre"]
            assert set(r["cultivos"]) <= hoja, (r["nombre"], set(r["cultivos"]) - hoja)

    def test_un_escalar_deja_cultivos_vacio_y_el_alcance_lo_dice(self) -> None:
        escalares = [r for r in indice() if not isinstance(r["certificados"], list)]

        assert escalares
        for r in escalares:
            assert r["cultivos"] == [], r["nombre"]
            assert r["alcance"] in {"all_crops", "not_applicable"}, r["nombre"]


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
    def test_por_default_manda_a_objetivos(self) -> None:
        """Default desde el 14 de septiembre de 2026, por la pregunta compuesta.

        "salivazo en la caña, un enraizador y algo para el picudo" salía 1/3 con la
        plaga por grep: el OR de los tres pedidos casa en 38 documentos y la ficha de
        Biomet queda enterrada. Con `objetivos` en el REPL, 3/3.
        """
        assert "En el REPL tenés `objetivos`" in contrato()
        assert "`ontologia/objetivos.md` tiene los alias" not in contrato()

    def test_sin_ontologia_cambia_esa_vineta_y_solo_esa(self) -> None:
        sin = contrato(ontologia=False)

        assert "`ontologia/objetivos.md` tiene los alias" in sin
        assert "`objetivos`" not in sin
        assert sin.replace(BUSCAR, BUSCAR_EN_OBJETIVOS) == contrato()

    def test_el_tipo_de_producto_es_un_eje_en_las_dos_variantes(self) -> None:
        """La viñeta de `tags` va afuera de BUSCAR, así que `--sin-ontologia` no la toca.

        Antes de esta viñeta, "algo para el tratamiento del agua" salía 1 de 3: el
        modelo buscaba `corrector de pH|buffer` y la ficha de Novicor dice "corrector
        de dureza". El tag `corrector-de-dureza` estaba en `productos` desde siempre;
        lo que no estaba era una línea que dijera que ese eje existe.
        """
        for con in (contrato(), contrato(ontologia=False)):
            assert "el eje es `tags` en `productos`" in con
            assert "`ontologia/tipos-producto.md`" in con

    def test_el_pedido_lleva_el_contrato_que_le_piden(self) -> None:
        assert "En el REPL tenés `objetivos`" in pedido("¿zompopo?", ())
        assert "`objetivos`" not in pedido("¿zompopo?", (), ontologia=False)
