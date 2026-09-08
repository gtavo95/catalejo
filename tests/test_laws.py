"""Las leyes de merge, verificadas contra logs generados al azar.

En vez de probar "este agente con esta entrada da esto", se generan logs y se
verifica que la ley aguanta siempre. Eso encuentra los casos que no se te ocurren.
"""

from hypothesis import given
from hypothesis import strategies as st

from catalejo.core import ZERO, Fail, Log, Message, Role, Status, merge, normal

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


class TestMonoide:
    @given(logs)
    def test_neutro_por_derecha(self, a: Log) -> None:
        a = normal(a)
        assert merge(a, ZERO) == a

    @given(logs)
    def test_neutro_por_izquierda(self, a: Log) -> None:
        a = normal(a)
        assert merge(ZERO, a) == a

    @given(logs, logs, logs)
    def test_asociatividad(self, a: Log, b: Log, c: Log) -> None:
        assert merge(merge(a, b), c) == merge(a, merge(b, c))

    @given(logs)
    def test_normal_es_idempotente(self, a: Log) -> None:
        assert normal(normal(a)) == normal(a)


class TestConmutatividad:
    """Vale en todos los canales menos `said`. Es la frontera que dice qué se
    puede correr en paralelo."""

    @given(logs, logs)
    def test_vale_en_fails(self, a: Log, b: Log) -> None:
        assert merge(a, b).fails == merge(b, a).fails

    @given(logs, logs)
    def test_vale_en_vote(self, a: Log, b: Log) -> None:
        assert merge(a, b).vote == merge(b, a).vote

    @given(logs, logs)
    def test_vale_en_spent(self, a: Log, b: Log) -> None:
        assert merge(a, b).spent == merge(b, a).spent

    @given(logs, logs)
    def test_vale_en_reads(self, a: Log, b: Log) -> None:
        assert merge(a, b).reads == merge(b, a).reads

    def test_no_vale_en_said(self) -> None:
        a = Log(said=(Message(Role.USER, "hola"),))
        b = Log(said=(Message(Role.ASSISTANT, "chau"),))

        assert merge(a, b) != merge(b, a)


class TestIdempotencia:
    """Vale en fails y vote, y no en said, spent ni reads. Por eso reintentar una
    búsqueda caída es gratis y reintentar una respuesta no."""

    @given(logs)
    def test_vale_en_fails(self, a: Log) -> None:
        a = normal(a)
        assert merge(a, a).fails == a.fails

    @given(logs)
    def test_vale_en_vote(self, a: Log) -> None:
        assert merge(a, a).vote == a.vote

    def test_no_vale_en_said(self) -> None:
        a = normal(Log(said=(Message(Role.ASSISTANT, "hola"),)))

        assert len(merge(a, a).said) == 2

    def test_no_vale_en_reads(self) -> None:
        """Consultar dos veces son dos consultas."""
        a = Log(reads=1)

        assert merge(a, a).reads == 2

    def test_no_vale_en_spent(self) -> None:
        """Y está bien que no valga: reintentar cuesta plata, y el álgebra lo
        dice sola en vez de que haya que acordarse."""
        a = Log(spent=100)

        assert merge(a, a).spent == 200


class TestCanales:
    def test_said_concatena_en_orden(self) -> None:
        a = Log(said=(Message(Role.USER, "uno"),))
        b = Log(said=(Message(Role.ASSISTANT, "dos"),))

        assert [m.text for m in merge(a, b).said] == ["uno", "dos"]

    def test_fails_es_un_conjunto(self) -> None:
        caido = Fail("web", "timeout")

        out = merge(Log(fails=(caido,)), Log(fails=(caido,)))

        assert out.fails == (caido,)

    def test_el_voto_de_seguir_le_gana_al_de_terminar(self) -> None:
        listo = Log(vote=Status.DONE)
        falta = Log(vote=Status.CONTINUE)

        assert merge(listo, falta).vote is Status.CONTINUE
        assert merge(falta, listo).vote is Status.CONTINUE

    def test_quiet_es_el_neutro_del_voto(self) -> None:
        assert merge(Log(vote=Status.QUIET), Log(vote=Status.DONE)).vote is Status.DONE

    def test_spent_suma(self) -> None:
        assert merge(Log(spent=30), Log(spent=12)).spent == 42

    def test_reads_suma(self) -> None:
        assert merge(Log(reads=1), Log(reads=2)).reads == 3

    def test_cero_es_el_neutro_del_gasto(self) -> None:
        assert merge(Log(spent=0), Log(spent=42)).spent == 42
