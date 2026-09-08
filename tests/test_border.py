"""El borde de un alcance y las reglas que se pueden aplicar ahí.

Lo que se prueba no es que `drop` vacíe un canal, que es evidente, sino dónde se
la puede mover sin cambiar nada. Esa es la única razón por la que la distinción
entre vaciar y colapsar importa.
"""

from hypothesis import given
from hypothesis import strategies as st

from catalejo.core import (
    ZERO,
    Cell,
    Channel,
    Fail,
    Log,
    Message,
    Role,
    Status,
    border,
    drop,
    fanout,
    loop,
    merge,
    mute,
    normal,
    quiet,
    then,
)

messages = st.builds(Message, role=st.sampled_from(Role), text=st.text(max_size=6))
fails = st.builds(Fail, who=st.text(max_size=3), reason=st.text(max_size=6))
logs = st.builds(
    Log,
    said=st.lists(messages, max_size=3).map(tuple),
    fails=st.lists(fails, max_size=3).map(tuple),
    vote=st.sampled_from(Status),
    spent=st.integers(min_value=0, max_value=1000),
    reads=st.integers(min_value=0, max_value=10),
)
canales: st.SearchStrategy[list[Channel]] = st.lists(
    st.sampled_from(["said", "fails", "vote", "spent", "reads"]),
    max_size=3,
    unique=True,
)


def dice(text: str) -> Cell:
    async def cell(seen: Log) -> Log:
        return Log(said=(Message(Role.ASSISTANT, text),))

    return cell


def trabaja(text: str, *, spent: int = 10) -> Cell:
    """Habla, gasta, falla y pide otra vuelta: toca los cinco canales."""

    async def cell(seen: Log) -> Log:
        return Log(
            said=(Message(Role.ASSISTANT, text),),
            fails=(Fail(text, "timeout"),),
            vote=Status.CONTINUE,
            spent=spent,
            reads=1,
        )

    return cell


def eco(vote: Status = Status.QUIET) -> Cell:
    """Repite cuántos dichos le llegaron. Depende de `said` de la entrada."""

    async def cell(seen: Log) -> Log:
        return Log(said=(Message(Role.ASSISTANT, str(len(seen.said))),), vote=vote)

    return cell


class TestDropEsHomomorfismo:
    """Vaciar la suma es sumar los vaciados. Esa es toda la licencia para
    aplicarla adentro o afuera de una acumulación."""

    @given(logs, logs, canales)
    def test_distribuye_sobre_merge(
        self, a: Log, b: Log, cs: list[Channel]
    ) -> None:
        h = drop(*cs)

        assert h(merge(a, b)) == merge(h(a), h(b))

    @given(logs, canales)
    def test_es_idempotente(self, a: Log, cs: list[Channel]) -> None:
        h = drop(*cs)

        assert h(h(normal(a))) == h(normal(a))

    @given(logs)
    def test_sin_canales_es_la_identidad(self, a: Log) -> None:
        assert drop()(normal(a)) == normal(a)

    @given(logs, canales)
    def test_manda_el_canal_al_vacio_y_no_toca_el_resto(
        self, a: Log, cs: list[Channel]
    ) -> None:
        out = drop(*cs)(normal(a))

        for canal in cs:
            assert getattr(out, canal) == getattr(ZERO, canal)
        for canal in ("said", "fails", "vote", "spent", "reads"):
            if canal not in cs:
                assert getattr(out, canal) == getattr(normal(a), canal)


class TestBorder:
    async def test_le_aplica_la_regla_a_lo_que_sale(self) -> None:
        out = await border(trabaja("a"), drop("said"))(ZERO)

        assert out.said == ()
        assert out.spent == 10

    async def test_la_celula_ve_lo_que_llego_sin_recortar(self) -> None:
        seen = Log(said=(Message(Role.USER, "hola"),))

        out = await border(eco(), drop("fails"))(seen)

        assert [m.text for m in out.said] == ["1"]

    async def test_afuera_del_loop_recorta_la_salida_y_no_los_pasos(self) -> None:
        """Un paso sigue viendo lo que dijo el anterior; lo recortado es lo que
        el loop entrega hacia afuera."""
        afuera = border(loop(eco(Status.CONTINUE), max_steps=3), drop("fails"))

        out = await afuera(ZERO)

        assert [m.text for m in out.said] == ["0", "1", "2"]
        assert out.fails == ()

    async def test_adentro_del_loop_le_tapa_el_paso_al_siguiente(self) -> None:
        adentro = loop(border(eco(Status.CONTINUE), drop("said")), max_steps=3)

        out = await adentro(ZERO)

        assert out.said == ()
        assert [f.who for f in out.fails] == ["loop"]


class TestMute:
    async def test_le_tira_lo_que_dijo(self) -> None:
        out = await mute(trabaja("a"))(ZERO)

        assert out.said == ()

    async def test_callarla_no_la_abarata(self) -> None:
        out = await mute(trabaja("a", spent=42))(ZERO)

        assert out.spent == 42
        assert out.fails == (Fail("a", "timeout"),)
        assert out.vote is Status.CONTINUE

    async def test_la_ley_levanta_a_fanout(self) -> None:
        """En `fanout` cada célula ve exactamente lo que llegó, así que mover el
        borde para adentro no le cambia la entrada a nadie."""
        seen = Log(said=(Message(Role.USER, "hola"),))
        afuera = mute(fanout(trabaja("a"), trabaja("b")))
        adentro = fanout(mute(trabaja("a")), mute(trabaja("b")))

        assert await afuera(seen) == await adentro(seen)

    async def test_la_ley_no_levanta_a_then(self) -> None:
        """Y no es un defecto de `mute`: `then` le muestra a cada célula lo que
        dijeron las anteriores, así que callar a la primera le cambia la entrada
        a la segunda. El homomorfismo vale sobre `merge`, no sobre el orden."""
        afuera = mute(then(dice("a"), eco()))
        adentro = then(mute(dice("a")), mute(eco()))

        assert await afuera(ZERO) == Log()
        assert await adentro(ZERO) == Log()

        vista_afuera = await then(dice("a"), eco())(ZERO)
        vista_adentro = await then(mute(dice("a")), eco())(ZERO)

        assert [m.text for m in vista_afuera.said] == ["a", "1"]
        assert [m.text for m in vista_adentro.said] == ["0"]


class TestQuiet:
    async def test_le_saca_el_voto(self) -> None:
        out = await quiet(trabaja("a"))(ZERO)

        assert out.vote is Status.QUIET

    async def test_deja_pasar_todo_lo_demas(self) -> None:
        out = await quiet(trabaja("a", spent=7))(ZERO)

        assert [m.text for m in out.said] == ["a"]
        assert out.spent == 7
        assert out.reads == 1

    async def test_una_observadora_no_tiene_al_loop_de_rehen(self) -> None:
        """Sin `quiet`, `trabaja` vota CONTINUE y el loop llega al tope de pasos.
        Con `quiet`, el paso vota QUIET y el loop corta en la primera vuelta."""
        rehen = await loop(trabaja("a"), max_steps=3)(ZERO)
        libre = await loop(quiet(trabaja("a")), max_steps=3)(ZERO)

        assert len(rehen.said) == 3
        assert len(libre.said) == 1
        assert libre.vote is Status.QUIET
