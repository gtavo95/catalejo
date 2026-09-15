"""El plan como dato: qué significa cada op y cuál entra.

Todo acá es puro. No hay modelo, no hay REPL y no hay Log: si algo de esto
necesitara uno de los tres, la checklist no sería un tipo de dato.
"""

from functools import reduce

from hypothesis import given
from hypothesis import strategies as st

from catalejo.core import (
    Plan,
    PlanOp,
    Step,
    activa,
    admitidos,
    admitir,
    aplicar,
    cerrado,
    cerrar,
    completo,
    deja,
    hijos,
    pendientes,
    proyectar,
)

ops = st.builds(
    PlanOp,
    verb=st.sampled_from(["add_step", "mark", "skip", "revise"]),
    id=st.sampled_from(["a", "b", "c", ""]),
    intent=st.just(""),
    status=st.sampled_from(["todo", "active", "done", "skipped", "listo"]),
    completes_when=st.just(""),
)


def plan(*ids: str) -> Plan:
    return proyectar(tuple(PlanOp("add_step", i) for i in ids))


class TestCompleto:
    def test_el_plan_vacio_no_esta_completo(self) -> None:
        """`all` sobre la tupla vacía es True, y esto es la única fuente de DONE:
        sin el `bool(plan)`, el plan recién nacido del paso 1 dice que el turno
        terminó y el agente contesta sin haber hecho nada."""
        assert completo(()) is False

    def test_completo_cuando_no_queda_nada_abierto(self) -> None:
        p = proyectar(
            (
                PlanOp("add_step", "a"),
                PlanOp("add_step", "b"),
                PlanOp("mark", "a", status="done"),
                PlanOp("skip", "b"),
            )
        )

        assert completo(p) is True
        assert pendientes(p) == ()

    def test_un_paso_activo_no_completa(self) -> None:
        p = proyectar((PlanOp("add_step", "a"), PlanOp("mark", "a", status="active")))

        assert completo(p) is False


class TestAplicar:
    def test_agrega_un_paso_en_todo(self) -> None:
        assert aplicar((), PlanOp("add_step", "a", "buscar")) == (Step("a", "buscar", "todo"),)

    def test_un_verbo_desconocido_devuelve_el_mismo_plan(self) -> None:
        p = plan("a")

        assert aplicar(p, PlanOp("revise", "a")) == p

    def test_un_id_que_no_existe_devuelve_el_mismo_plan(self) -> None:
        p = plan("a")

        assert aplicar(p, PlanOp("mark", "z", status="done")) == p

    def test_un_estado_que_no_existe_devuelve_el_mismo_plan(self) -> None:
        p = plan("a")

        assert aplicar(p, PlanOp("mark", "a", status="casi")) == p

    def test_agregar_dos_veces_el_mismo_id_no_hace_nada(self) -> None:
        p = plan("a")

        assert aplicar(p, PlanOp("add_step", "a", "otra cosa")) == p

    def test_un_id_vacio_no_hace_nada(self) -> None:
        assert aplicar((), PlanOp("add_step", "")) == ()

    def test_marcar_lo_que_ya_estaba_asi_no_hace_nada(self) -> None:
        """Es lo que hace detectable el no-op: `admitir` compara y nada más."""
        p = plan("a")

        assert aplicar(p, PlanOp("mark", "a", status="todo")) == p


class TestLaLeyDelFold:
    """Plegar dos tramos por separado da lo mismo que plegarlos juntos.

    Es la ley que hace que nadie tenga que pasar el plan: `then` le da a cada
    célula `merge(seen, acc)`, así que cada una proyecta el tramo que le tocaba.
    """

    @given(st.lists(ops, max_size=6).map(tuple), st.lists(ops, max_size=6).map(tuple))
    def test_proyectar_es_asociativo(
        self, a: tuple[PlanOp, ...], b: tuple[PlanOp, ...]
    ) -> None:
        assert proyectar(a + b) == reduce(aplicar, b, proyectar(a))

    @given(st.lists(ops, max_size=6).map(tuple))
    def test_proyectar_no_depende_de_cuando_se_llame(self, a: tuple[PlanOp, ...]) -> None:
        assert proyectar(a) == proyectar(a)


class TestAdmitir:
    def test_el_lote_se_juzga_contra_el_plan_que_va_quedando(self) -> None:
        """Contra el plan de antes, el `mark` parece un no-op y se perdía."""
        lote = (PlanOp("add_step", "dosis"), PlanOp("mark", "dosis", status="active"))

        entran = admitidos(admitir(lote, (), flex="acotado"))

        assert entran == lote

    def test_rechaza_lo_que_no_cambia_nada(self) -> None:
        veredictos = admitir((PlanOp("mark", "z", status="done"),), plan("a"), flex="acotado")

        assert admitidos(veredictos) == ()
        assert veredictos[0].motivo == "no cambia nada del plan"

    def test_el_vocabulario_lo_cierra_la_flexibilidad(self) -> None:
        veredictos = admitir((PlanOp("add_step", "nuevo"),), (), flex="cerrado")

        assert admitidos(veredictos) == ()
        assert "`add_step` no se puede usar" in veredictos[0].motivo

    def test_un_done_sobre_un_paso_con_compuerta_se_rechaza_siempre(self) -> None:
        """Ni siquiera cuando la compuerta ya se cumple: el `done` lo emite
        `cerrar`, y que haya un solo emisor es toda la garantía."""
        p = proyectar((PlanOp("add_step", "a", completes_when="leyo"),))

        veredictos = admitir((PlanOp("mark", "a", status="done"),), p, flex="acotado")

        assert admitidos(veredictos) == ()
        assert "`leyo`" in veredictos[0].motivo

    def test_sobre_un_paso_ya_cerrado_el_motivo_es_el_verdadero(self) -> None:
        """"lo cierro yo cuando se cumpla" sobre un paso que la compuerta ya
        cerró es falso: ya se cumplió. El motivo tiene que ser el que es."""
        p = proyectar(
            (
                PlanOp("add_step", "a", completes_when="leyo"),
                PlanOp("mark", "a", status="done"),
            )
        )

        veredictos = admitir((PlanOp("mark", "a", status="done"),), p, flex="acotado")

        assert veredictos[0].motivo == "no cambia nada del plan"

    def test_el_modelo_si_puede_activar_un_paso_con_compuerta(self) -> None:
        p = proyectar((PlanOp("add_step", "a", completes_when="leyo"),))

        assert admitidos(admitir((PlanOp("mark", "a", status="active"),), p, flex="acotado"))

    def test_el_modelo_cierra_los_pasos_sin_compuerta(self) -> None:
        """La válvula. Con el registro vacío, TODO el plan cae acá y el plan
        vuelve a garantizar lo mismo que garantizaba antes, que es nada."""
        p = plan("a")

        entran = admitidos(admitir((PlanOp("mark", "a", status="done"),), p, flex="acotado"))

        assert completo(proyectar((PlanOp("add_step", "a"), *entran)))


class TestDeja:
    def test_firme_agrega_y_marca_pero_no_salta(self) -> None:
        assert deja("firme") == {"add_step", "mark"}

    def test_acotado_deja_agregar(self) -> None:
        assert deja("acotado") == {"add_step", "mark", "skip"}

    def test_cerrado_no_deja_agregar(self) -> None:
        assert deja("cerrado") == {"mark", "skip"}

    def test_una_flexibilidad_que_no_existe_da_la_mas_chica(self) -> None:
        """Un nombre mal escrito en el cableado tiene que QUITAR permisos."""
        assert deja("abierto") == deja("cerrado")


class TestCerrar:
    def test_cierra_el_paso_cuya_compuerta_se_cumple(self) -> None:
        p = proyectar((PlanOp("add_step", "a", completes_when="leyo"),))

        assert cerrar(p, frozenset({"leyo"})) == (PlanOp("mark", "a", status="done"),)

    def test_no_toca_el_paso_cuya_compuerta_no_se_cumple(self) -> None:
        p = proyectar((PlanOp("add_step", "a", completes_when="leyo"),))

        assert cerrar(p, frozenset()) == ()

    def test_no_toca_los_pasos_sin_compuerta(self) -> None:
        assert cerrar(plan("a"), frozenset({"leyo"})) == ()

    def test_no_reabre_ni_recierra_lo_ya_cerrado(self) -> None:
        p = proyectar(
            (PlanOp("add_step", "a", completes_when="leyo"), PlanOp("skip", "a"))
        )

        assert cerrar(p, frozenset({"leyo"})) == ()


VENTA = (
    PlanOp("add_step", "diagnostico", "armar el carrito"),
    PlanOp("add_step", "plaga", padre="diagnostico", completes_when="dijo_plaga"),
    PlanOp("add_step", "producto", padre="diagnostico", exige=("buscar", "fuente")),
    PlanOp("add_step", "pago", "cobrar"),
    PlanOp("add_step", "medio", padre="pago"),
)


def hoja(p: Plan) -> str:
    h = activa(p)
    assert h is not None
    return h.id


class TestEtapas:
    """Dos niveles con un campo, y todo lo demás calculado."""

    def test_add_step_lleva_padre_y_exige_al_paso(self) -> None:
        p = proyectar(VENTA)

        assert p[2] == Step("producto", "", "todo", "", "diagnostico", ("buscar", "fuente"))
        assert [h.id for h in hijos(p, "diagnostico")] == ["plaga", "producto"]

    def test_una_etapa_se_cierra_cuando_sus_hijos_se_cierran(self) -> None:
        p = proyectar((*VENTA, PlanOp("mark", "plaga", status="done"), PlanOp("skip", "producto")))

        assert cerrado(p, p[0])
        assert not cerrado(p, p[3])

    def test_una_etapa_saltada_esta_cerrada_aunque_tenga_hijos_abiertos(self) -> None:
        """Saltar la facturación es saltar sus datos."""
        p = proyectar((*VENTA, PlanOp("skip", "pago")))

        assert cerrado(p, p[3])
        assert pendientes(p) == (p[0], p[1], p[2])

    def test_la_hoja_activa_es_el_primer_hijo_abierto_de_la_primera_etapa_abierta(self) -> None:
        p = proyectar(VENTA)
        assert hoja(p) == "plaga"

        p = proyectar((*VENTA, PlanOp("mark", "plaga", status="done")))
        assert hoja(p) == "producto"

        p = proyectar((*VENTA, PlanOp("skip", "diagnostico")))
        assert hoja(p) == "medio"

    def test_una_hoja_agregada_despues_cuenta_en_su_etapa_y_no_al_final(self) -> None:
        """En orden plano quedaría después de `medio`; por `padre` sigue en diagnóstico."""
        tarde = PlanOp("add_step", "dudas", padre="diagnostico")
        p = proyectar((*VENTA, PlanOp("mark", "plaga", status="done"), PlanOp("skip", "producto"), tarde))

        assert hoja(p) == "dudas"
        assert not cerrado(p, p[0])

    def test_un_paso_sin_padre_y_sin_hijos_es_una_hoja_raiz(self) -> None:
        p = proyectar((PlanOp("add_step", "a"), PlanOp("add_step", "b")))

        assert hoja(p) == "a"

    def test_todo_cerrado_es_completo_y_sin_hoja_activa(self) -> None:
        p = proyectar((*VENTA, PlanOp("skip", "diagnostico"), PlanOp("skip", "pago")))

        assert completo(p)
        assert activa(p) is None

    def test_el_plan_plano_no_cambia(self) -> None:
        """Sin `padre` todo esto es lo de siempre: cerrado es el estado, y nada más."""
        p = proyectar((PlanOp("add_step", "a"), PlanOp("mark", "a", status="done")))

        assert completo(p) and pendientes(p) == () and activa(p) is None
