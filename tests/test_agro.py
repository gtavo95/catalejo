"""Lo que el asesor sirve como respuesta, que no siempre es lo que el modelo dijo."""

import pytest

from catalejo.core import Conversation, Fail, Log, Message, PlanOp, Role, activa, cerrado, proyectar
from catalejo.llm import Reply, Stub
from agro import (
    BUNDLE,
    BUSCAR,
    BUSCAR_EN_OBJETIVOS,
    NOTAS,
    PASOS,
    SEMILLA,
    SESION,
    SIN_CONSULTAR,
    VENTA,
    armar,
    claves,
    cliente,
    contrato,
    corpus,
    dijo_area,
    dijo_plaga,
    final,
    indice,
    objetivos,
    pedido,
    registro_sesion,
    resolver,
    responder,
    vocabulario,
)

SIN_FUNDAMENTO = Fail("grounding", "contestó sin haber consultado el contexto")
SIN_PASOS = Fail("loop", "tope de pasos: 12")
FUENTE_VINETA = "- Cerrá con una línea `FUENTE: productos/x.md`"


def dicho(texto: str) -> Message:
    return Message(Role.ASSISTANT, texto)


class TestFinal:
    def test_la_respuesta_normal_sale_tal_cual(self) -> None:
        out = Log(said=(Message(Role.USER, "¿dosis?"), dicho("1.0 L/Ha\nFUENTE: productos/x.md")))

        assert final(out) == "1.0 L/Ha\nFUENTE: productos/x.md"

    def test_el_ultimo_bloque_de_codigo_no_es_respuesta(self) -> None:
        out = Log(said=(dicho("ya veo"), dicho("```python\nprint(1)\n```")))

        assert final(out) == "ya veo"

    def test_sin_consultar_pero_sin_nada_que_consultar_se_sirve(self) -> None:
        """El turno en que se pregunta la plaga no tenía nada que fundar."""
        out = Log(said=(dicho("¿Qué plaga ves?"),), fails=(SIN_FUNDAMENTO,))

        assert final(out, fundar=False) == "¿Qué plaga ves?"

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

    def test_las_notas_se_piden_en_el_turno_del_usuario_en_las_dos_variantes(self) -> None:
        """Va acá y no en el preámbulo, que ya lo pedía en abstracto: en 9 corridas de
        las dos compuestas el modelo escribió `notas` 0 veces, y el run 1 de salivazo
        perdió Biomet del prompt por eso. Fuera de BUSCAR, así que `--sin-ontologia`
        no la toca."""
        for con in (contrato(), contrato(ontologia=False)):
            assert "`notas` ya existe" in con
            assert "al principio del bloque siguiente" in con
            assert "`notas.append(" in con
            assert "vuelve al pie de cada salida" in con

    def test_sin_notas_saca_esa_vineta_y_solo_esa(self) -> None:
        sin = contrato(notas=False)

        assert "notas" not in sin
        assert sin.replace(FUENTE_VINETA, NOTAS + FUENTE_VINETA) == contrato()
        assert "notas" not in pedido("¿zompopo?", (), notas=False)


class Guion(Stub):
    """Un `Stub` que además es un `Provider`: tiene nombre y se puede cerrar."""

    model = "guion"

    async def aclose(self) -> None:
        pass


class Mudo:
    """Un `Provider` que nunca se llama: armar el agente no habla con nadie."""

    model = "mudo"

    async def complete(self, conv: Conversation) -> Reply:
        raise AssertionError("armar el agente no debería llamar al modelo")

    async def aclose(self) -> None:
        pass


class TestNotasEnElWorkspace:
    async def test_armar_la_crea_vacia_y_el_primer_append_no_revienta(self) -> None:
        _, ws = armar("=== productos/x.md ===\nhola", Mudo(), ver=False)

        out = await ws.run("notas.append('x')")

        assert out.err == ""
        assert out.notas == ("x",)

    async def test_responder_la_vacia_por_pregunta(self) -> None:
        """El workspace vive toda la sesión; una nota de la pregunta anterior al pie
        de las salidas de esta sería un dato falso."""
        _, ws = armar("=== productos/x.md ===\nhola", Mudo(), ver=False)
        await ws.run("notas.append('de la pregunta anterior')")

        async def agente(seen: Log) -> Log:
            return Log(said=(dicho("Sin producto.\nFUENTE: ninguna"),), reads=1)

        await responder(agente, ws, "¿y ahora?", (), ver=False)

        assert (await ws.run("pass")).notas == ()


def pedido_del(pregunta: str, historia: tuple[tuple[str, str], ...] = ()) -> Log:
    return Log(said=(Message(Role.USER, pedido(pregunta, historia, venta=proyectar(SESION))),))


def hoja(ops: tuple[PlanOp, ...]) -> str:
    h = activa(proyectar(ops))
    assert h is not None
    return h.id


class TestCliente:
    def test_sin_historia_es_la_pregunta(self) -> None:
        assert cliente(pedido_del("mosca blanca")) == "mosca blanca"

    def test_con_historia_trae_las_preguntas_y_no_las_respuestas(self) -> None:
        log = pedido_del("dos", (("tengo una plaga", "¿mosca blanca, gusano?"),))

        assert cliente(log) == "tengo una plaga\ndos"

    def test_no_lee_el_contrato_ni_el_plan(self) -> None:
        texto = cliente(pedido_del("mosca blanca"))

        assert "Para contestar" not in texto and VENTA not in texto


OBJS = [
    {"id": "mosca-blanca", "etiqueta": "Mosca blanca", "alias": ["bemisia"]},
    {"id": "arana-roja", "etiqueta": "Araña roja", "alias": []},
    {"id": "broca", "etiqueta": "Broca del café", "alias": ["broca"]},
]


class TestCompuertasDeLaVenta:
    """Miran las palabras del cliente y nada más. Pisos, como todas."""

    def test_dijo_plaga_cierra_con_la_palabra_del_cliente(self) -> None:
        assert dijo_plaga(OBJS)(pedido_del("tengo mosca blanca en el tomate"))

    def test_no_cierra_con_tengo_una_plaga(self) -> None:
        assert not dijo_plaga(OBJS)(pedido_del("tengo una plaga en el tomate"))

    def test_no_cierra_con_lo_que_dijo_el_modelo(self) -> None:
        log = pedido_del("dos", (("tengo una plaga", "¿Es mosca blanca o araña roja?"),))

        assert not dijo_plaga(OBJS)(log)

    def test_pliega_caso_y_acentos(self) -> None:
        assert dijo_plaga(OBJS)(pedido_del("ARAÑA roja"))
        assert dijo_plaga(OBJS)(pedido_del("arana roja"))

    def test_no_casa_adentro_de_otra_palabra(self) -> None:
        assert not dijo_plaga(OBJS)(pedido_del("un brocado de tela"))

    def test_sin_vocabulario_no_cierra_nunca(self) -> None:
        assert not dijo_plaga([])(pedido_del("mosca blanca"))

    def test_sobre_el_bundle_de_verdad(self) -> None:
        gate = registro_sesion(objetivos(indice()))["dijo_plaga"]

        assert gate(pedido_del("mosca blanca"))
        assert not gate(pedido_del("tengo una plaga en el tomate"))

    def test_dijo_area_con_cifras_y_con_palabras(self) -> None:
        for dicho in ("2 manzanas", "dos manzanas", "1.5 ha", "3 hectáreas", "tengo 10mz"):
            assert dijo_area(pedido_del(dicho)), dicho

    def test_dijo_area_lo_que_no_atrapa(self) -> None:
        """Documentado, no perseguido: es un piso. "una manzana podrida" sí cuenta."""
        for dicho in ("manzana", "tengo tomate", "veinticinco manzanas", "manzana y media"):
            assert not dijo_area(pedido_del(dicho)), dicho


class TestLaSemillaDeLaVenta:
    def test_los_ids_no_se_repiten_entre_los_dos_planes(self) -> None:
        """`mark` y `skip` se enrutan por id: un id en los dos iría al equivocado."""
        assert {op.id for op in SESION}.isdisjoint(op.id for op in SEMILLA)

    def test_todo_lo_que_exige_esta_en_el_catalogo(self) -> None:
        assert all(i in PASOS for op in SESION for i in op.exige)

    def test_toda_compuerta_esta_en_el_registro(self) -> None:
        registro = registro_sesion(OBJS)

        assert all(op.completes_when in registro for op in SESION if op.completes_when)

    def test_arranca_con_plaga_activa_y_sin_exigir_nada(self) -> None:
        h = activa(proyectar(SESION))

        assert h is not None and h.id == "plaga" and h.exige == ()


class TestArmar:
    def test_plan_y_sesion_juntos_no_se_cablean(self) -> None:
        with pytest.raises(ValueError):
            armar("=== productos/x.md ===\nhola", Mudo(), ver=False, plan=True, sesion=True)


class TestPedidoConVenta:
    def test_lleva_la_venta_dibujada_y_sus_instrucciones(self) -> None:
        texto = pedido("mosca blanca", (), venta=proyectar(SESION))

        assert VENTA in texto and "[ ] diagnostico" in texto and "    [ ] plaga" in texto
        assert "dura toda la conversación" in texto
        assert "El turno no termina" not in texto and "[ ] buscar" not in texto

    def test_con_la_hoja_activa_sin_exigencia_lo_dice(self) -> None:
        con = pedido("tengo una plaga", (), venta=proyectar(SESION))
        sin = pedido("dos", (), venta=proyectar((*SESION, PlanOp("mark", "plaga", status="done"))))

        assert "no exige consultar nada" in con and "`plaga`" in con
        assert "no exige consultar nada" not in sin

    def test_dibuja_el_estado_que_trae(self) -> None:
        texto = pedido("dos", (), venta=proyectar((*SESION, PlanOp("mark", "plaga", status="done"))))

        assert "    [x] plaga" in texto


class TestLaVentaDePuntaAPunta:
    """Tres turnos con un modelo de guion sobre el corpus real, sin exigir a nadie de más."""

    async def test_tres_turnos(self) -> None:
        modelo = Guion(
            "```python\nprint(len(productos))\n```",
            "¿Qué plaga ves en el tomate?",
            "```python\nprint(read(catalogo, 'productos/metaveria-40-ew.md'))\n```",
            "Metaveria 40 EW, certificado en tomate. ¿Cuántas manzanas tenés?\nFUENTE: productos/metaveria-40-ew.md",
            "Para 2 manzanas van 1.4 a 3 L. ¿Alguna duda del producto?\nFUENTE: productos/metaveria-40-ew.md",
        )
        agente, ws = armar(corpus(), modelo, ver=False, sesion=True)
        try:
            r1, c1 = await responder(agente, ws, "tengo una plaga en el tomate", (), ver=False, course=())
            p1 = proyectar(c1)
            assert r1.startswith("¿Qué plaga")
            assert hoja(c1) == "plaga"
            assert cerrado(p1, p1[2]), "producto cierra con cualquier consulta: es el piso de REGISTRO"

            historia: tuple[tuple[str, str], ...] = (("tengo una plaga en el tomate", r1),)
            r2, c2 = await responder(agente, ws, "mosca blanca", historia, ver=False, course=c1)
            assert [op.id for op in c2] == ["plaga", "receta"]
            assert hoja((*c1, *c2)) == "cantidad"
            assert "FUENTE" in r2

            historia = (*historia, ("mosca blanca", r2))
            r3, c3 = await responder(agente, ws, "dos manzanas", historia, ver=False, course=(*c1, *c2))
            p3 = proyectar((*c1, *c2, *c3))
            assert [op.id for op in c3] == ["cantidad"]
            assert hoja((*c1, *c2, *c3)) == "dudas"
            assert not cerrado(p3, p3[0])
        finally:
            ws.cerrar()

    async def test_la_checklist_del_turno_se_siembra_y_se_exige(self) -> None:
        """Turno 2 con `producto` ya cerrado: la hoja activa es `receta`, el turno
        exige `dosis` y `fuente`, y contestar sin la ficha se veta una vez."""
        modelo = Guion(
            "Metaveria sirve. ¿Cuántas manzanas?",
            "```python\nprint(read(catalogo, 'productos/metaveria-40-ew.md'))\n```",
            "Metaveria 40 EW, 1 a 2.14 L/ha. ¿Cuántas manzanas?\nFUENTE: productos/metaveria-40-ew.md",
        )
        agente, ws = armar(corpus(), modelo, ver=False, sesion=True)
        previos = (*SESION, PlanOp("mark", "producto", status="done"))
        try:
            r, c = await responder(agente, ws, "mosca blanca", (), ver=False, course=previos)
        finally:
            ws.cerrar()

        vistos = "\n".join(m.text for m in modelo.visto[1])
        assert "[plan] faltan pasos:" in vistos and "`dosis`" in vistos
        assert [op.id for op in c] == ["plaga", "receta"]
        assert "FUENTE" in r
