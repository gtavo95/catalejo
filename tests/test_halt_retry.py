"""El voto que frena y el combinador que descarta.

Van juntos porque se cruzan: reintentar algo que alguien frenó a propósito es
exactamente lo que no hay que hacer.
"""

import asyncio

from catalejo.core import (
    ZERO,
    Cell,
    Fail,
    Log,
    Message,
    Role,
    Status,
    fanout,
    loop,
    merge,
    retry,
    then,
)


def dice(text: str, *, vote: Status = Status.CONTINUE) -> Cell:
    async def cell(seen: Log) -> Log:
        return Log(said=(Message(Role.ASSISTANT, text),), vote=vote)

    return cell


def juez(vote: Status) -> Cell:
    async def cell(seen: Log) -> Log:
        return Log(vote=vote)

    return cell


def inestable(falla_hasta: int, marcas: list[int], *, spent: int = 10) -> Cell:
    """Falla los primeros `falla_hasta` intentos y después anda.

    Habla, lee y vota en todos, así que se ve qué sobrevive de los perdidos.
    """
    intento = 0

    async def cell(seen: Log) -> Log:
        nonlocal intento
        intento += 1
        marcas.append(len(seen.fails))
        roto = intento <= falla_hasta
        return Log(
            said=(Message(Role.ASSISTANT, f"intento {intento}"),),
            fails=(Fail("web", f"timeout {intento}"),) if roto else (),
            vote=Status.CONTINUE if roto else Status.DONE,
            spent=spent,
            reads=1,
        )

    return cell


class TestHalt:
    def test_le_gana_a_todos(self) -> None:
        assert max(Status) is Status.HALT

    async def test_frena_el_loop(self) -> None:
        out = await loop(then(dice("a"), juez(Status.HALT)), max_steps=5)(ZERO)

        assert len(out.said) == 1
        assert out.vote is Status.HALT

    async def test_le_gana_al_continue_del_paso(self) -> None:
        """Sin esto, la que quiere seguir tiene al loop de rehén."""
        paso = then(dice("a", vote=Status.CONTINUE), juez(Status.HALT))

        assert (await paso(ZERO)).vote is Status.HALT

    async def test_no_corta_el_paso_que_esta_corriendo(self) -> None:
        """Frena el turno siguiente, no el actual. Lo que las otras dijeron
        aterriza igual, que es la diferencia con el error como cortocircuito."""
        out = await then(juez(Status.HALT), dice("igual hablo"))(ZERO)

        assert [m.text for m in out.said] == ["igual hablo"]

    async def test_una_sola_celula_alcanza_en_paralelo(self) -> None:
        out = await fanout(dice("a"), dice("b"), juez(Status.HALT))(ZERO)

        assert out.vote is Status.HALT
        assert len(out.said) == 2

    async def test_cruza_el_borde_del_sub_loop(self) -> None:
        """El "seguí pensando" de adentro no se escapa, pero "esto no sigue" sí."""
        hijo = loop(then(dice("a"), juez(Status.HALT)), max_steps=3)
        padre = loop(hijo, max_steps=3)

        out = await padre(ZERO)

        assert out.vote is Status.HALT
        assert len(out.said) == 1


class TestRetry:
    async def test_devuelve_el_intento_que_anduvo(self) -> None:
        out = await retry(inestable(1, []), attempts=3)(ZERO)

        assert [m.text for m in out.said] == ["intento 2"]
        assert out.vote is Status.DONE

    async def test_tira_lo_que_dijo_el_intento_perdido(self) -> None:
        out = await retry(inestable(2, []), attempts=3)(ZERO)

        assert [m.text for m in out.said] == ["intento 3"]

    async def test_el_gasto_del_perdido_no_se_perdona(self) -> None:
        out = await retry(inestable(2, [], spent=10), attempts=3)(ZERO)

        assert out.spent == 30

    async def test_las_fallas_de_los_perdidos_quedan(self) -> None:
        out = await retry(inestable(2, []), attempts=3)(ZERO)

        assert [f.reason for f in out.fails] == ["timeout 1", "timeout 2"]

    async def test_los_reads_del_perdido_no_le_sirven_de_piso_al_que_anduvo(
        self,
    ) -> None:
        """Si quedaran, un intento que leyó y reventó le dejaría el piso servido
        a `grounded` y el que aterrizó podría contestar de memoria."""
        out = await retry(inestable(2, []), attempts=3)(ZERO)

        assert out.reads == 1

    async def test_el_intento_ve_las_fallas_de_los_anteriores(self) -> None:
        marcas: list[int] = []

        await retry(inestable(2, marcas), attempts=3)(ZERO)

        assert marcas == [0, 1, 2]

    async def test_el_ultimo_intento_vuelve_entero(self) -> None:
        """Fallar hasta el final no es abortar: el que llamó se lleva lo que haya."""
        out = await retry(inestable(9, []), attempts=2)(ZERO)

        assert [m.text for m in out.said] == ["intento 2"]
        assert len(out.fails) == 2
        assert out.spent == 20

    async def test_no_reintenta_lo_que_alguien_freno(self) -> None:
        intentos = 0

        async def frenada(seen: Log) -> Log:
            nonlocal intentos
            intentos += 1
            return Log(fails=(Fail("juez", "inyección"),), vote=Status.HALT)

        out = await retry(frenada, attempts=5)(ZERO)

        assert intentos == 1
        assert out.vote is Status.HALT

    async def test_con_un_intento_es_correr_la_celula(self) -> None:
        seen = Log(said=(Message(Role.USER, "hola"),))
        una = inestable(9, [])

        assert await retry(una, attempts=1)(seen) == await inestable(9, [])(seen)

    async def test_con_cero_intentos_no_agrega_nada(self) -> None:
        assert await retry(dice("a"), attempts=0)(ZERO) == ZERO

    async def test_no_se_dispara_sin_fallas(self) -> None:
        marcas: list[int] = []

        await retry(inestable(0, marcas), attempts=5)(ZERO)

        assert marcas == [0]

    async def test_los_intentos_van_en_serie(self) -> None:
        """Cada uno ve lo que falló antes, así que no se pueden solapar."""
        vivos = 0
        pico = 0

        async def lenta(seen: Log) -> Log:
            nonlocal vivos, pico
            vivos += 1
            pico = max(pico, vivos)
            await asyncio.sleep(0)
            vivos -= 1
            return Log(fails=(Fail("web", "timeout"),))

        await retry(lenta, attempts=3)(ZERO)

        assert pico == 1
