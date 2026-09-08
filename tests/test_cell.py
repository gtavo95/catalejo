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
    identity,
    loop,
    merge,
    only,
    then,
)


def dice(text: str) -> Cell:
    async def cell(seen: Log) -> Log:
        return Log(said=(Message(Role.ASSISTANT, text),))

    return cell


def caido(who: str) -> Cell:
    async def cell(seen: Log) -> Log:
        return Log(fails=(Fail(who, "timeout"),))

    return cell


def espia(text: str, visto: list[int]) -> Cell:
    """Anota cuántos dichos vio antes de hablar."""

    async def cell(seen: Log) -> Log:
        visto.append(len(seen.said))
        return Log(said=(Message(Role.ASSISTANT, text),))

    return cell


class TestThen:
    async def test_sin_celulas_no_agrega_nada(self) -> None:
        assert await then()(ZERO) == ZERO

    async def test_con_una_celula_se_comporta_como_ella(self) -> None:
        seen = Log(said=(Message(Role.USER, "hola"),))

        assert await then(dice("a"))(seen) == await dice("a")(seen)

    async def test_devuelve_solo_lo_que_agrego(self) -> None:
        seen = Log(said=(Message(Role.USER, "hola"),))

        out = await then(dice("a"), dice("b"))(seen)

        assert [m.text for m in out.said] == ["a", "b"]

    async def test_cada_celula_ve_lo_que_agregaron_las_anteriores(self) -> None:
        visto: list[int] = []
        seen = Log(said=(Message(Role.USER, "hola"),))

        await then(espia("a", visto), espia("b", visto), espia("c", visto))(seen)

        assert visto == [1, 2, 3]

    async def test_identity_es_el_neutro(self) -> None:
        seen = Log(said=(Message(Role.USER, "hola"),))

        con = await then(dice("a"), identity(), dice("b"))(seen)
        sin = await then(dice("a"), dice("b"))(seen)

        assert con == sin

    async def test_asociatividad_en_el_resultado(self) -> None:
        seen = Log(said=(Message(Role.USER, "hola"),))
        f, g, h = dice("a"), dice("b"), dice("c")

        izq = await then(then(f, g), h)(seen)
        der = await then(f, then(g, h))(seen)

        assert izq == der

    async def test_asociatividad_en_lo_que_vio_cada_celula(self) -> None:
        """La ley que se rompe en silencio si `then` pasa `acc` en vez de
        `merge(seen, acc)`. Un test de resultado no la agarra: el resultado
        coincide igual y lo que cambia es que la última célula deja de ver lo que
        dijo la primera."""
        seen = Log(said=(Message(Role.USER, "hola"),))

        izq: list[int] = []
        await then(then(espia("a", izq), espia("b", izq)), espia("c", izq))(seen)

        der: list[int] = []
        await then(espia("a", der), then(espia("b", der), espia("c", der)))(seen)

        assert izq == der == [1, 2, 3]


class TestFanout:
    async def test_ninguna_celula_ve_a_las_otras(self) -> None:
        visto: list[int] = []
        seen = Log(said=(Message(Role.USER, "hola"),))

        await fanout(espia("a", visto), espia("b", visto), espia("c", visto))(seen)

        assert visto == [1, 1, 1]

    async def test_junta_en_el_orden_dado_no_en_el_que_terminaron(self) -> None:
        def lenta(text: str, delay: float) -> Cell:
            async def cell(seen: Log) -> Log:
                await asyncio.sleep(delay)
                return Log(said=(Message(Role.ASSISTANT, text),))

            return cell

        out = await fanout(lenta("a", 0.02), lenta("b", 0.0))(ZERO)

        assert [m.text for m in out.said] == ["a", "b"]

    async def test_conmuta_en_fails_y_vote(self) -> None:
        f = caido("web")

        async def juez(seen: Log) -> Log:
            return Log(vote=Status.CONTINUE)

        uno = await fanout(f, juez)(ZERO)
        otro = await fanout(juez, f)(ZERO)

        assert uno.fails == otro.fails
        assert uno.vote is otro.vote

    async def test_una_celula_caida_no_mata_el_turno(self) -> None:
        """La composición es total: el error es un canal, no un cortocircuito."""
        agente = then(fanout(dice("encontré esto"), caido("web")), dice("la respuesta"))

        out = await agente(ZERO)

        assert [m.text for m in out.said] == ["encontré esto", "la respuesta"]
        assert out.fails == (Fail("web", "timeout"),)


class TestOnly:
    async def test_corre_cuando_se_cumple(self) -> None:
        out = await only(lambda seen: True, dice("a"))(ZERO)

        assert [m.text for m in out.said] == ["a"]

    async def test_no_agrega_nada_cuando_no(self) -> None:
        assert await only(lambda seen: False, dice("a"))(ZERO) == ZERO

    async def test_cablearla_dos_veces_no_cuesta(self) -> None:
        """La idempotencia del canal es lo que lo hace gratis."""
        guarda = only(lambda seen: seen.vote is Status.CONTINUE, caido("web"))
        seen = Log(vote=Status.CONTINUE)

        una = await guarda(seen)
        dos = await then(guarda, guarda)(seen)

        assert una.fails == dos.fails


def guion(*votos: Status, cuesta: int = 0) -> Cell:
    """Una célula que vota según un guion, un voto por paso."""
    paso = 0

    async def cell(seen: Log) -> Log:
        nonlocal paso
        voto = votos[paso] if paso < len(votos) else Status.DONE
        paso += 1
        return Log(said=(Message(Role.ASSISTANT, f"paso {paso}"),), vote=voto, spent=cuesta)

    return cell


class TestLoop:
    async def test_un_paso_que_vota_done_corta(self) -> None:
        out = await loop(guion(Status.DONE), max_steps=10)(ZERO)

        assert [m.text for m in out.said] == ["paso 1"]
        assert out.fails == ()

    async def test_un_paso_que_vota_quiet_corta(self) -> None:
        """Nadie pidió otra vuelta, así que un cableado a medio hacer se detiene
        en vez de quemar tokens."""
        out = await loop(guion(Status.QUIET), max_steps=10)(ZERO)

        assert [m.text for m in out.said] == ["paso 1"]

    async def test_sigue_mientras_el_paso_pida_seguir(self) -> None:
        pasos = (Status.CONTINUE, Status.CONTINUE, Status.DONE)

        out = await loop(guion(*pasos), max_steps=10)(ZERO)

        assert [m.text for m in out.said] == ["paso 1", "paso 2", "paso 3"]
        assert out.fails == ()

    async def test_el_voto_acumulado_no_lo_frena(self) -> None:
        """`vote` se junta por máximo, así que el acumulado se queda en CONTINUE
        para siempre. Si el loop lo mirara a él, no terminaría nunca."""
        out = await loop(guion(Status.CONTINUE, Status.DONE), max_steps=50)(ZERO)

        assert len(out.said) == 2
        assert out.fails == ()

    async def test_cada_paso_ve_lo_que_agregaron_los_anteriores(self) -> None:
        visto: list[int] = []

        async def cell(seen: Log) -> Log:
            visto.append(len(seen.said))
            voto = Status.CONTINUE if len(visto) < 3 else Status.DONE
            return Log(said=(Message(Role.ASSISTANT, "x"),), vote=voto)

        await loop(cell, max_steps=10)(Log(said=(Message(Role.USER, "hola"),)))

        assert visto == [1, 2, 3]

    async def test_devuelve_solo_lo_que_agrego(self) -> None:
        seen = Log(said=(Message(Role.USER, "hola"),))

        out = await loop(guion(Status.DONE), max_steps=10)(seen)

        assert [m.text for m in out.said] == ["paso 1"]

    async def test_el_tope_de_pasos_corta_y_lo_reporta(self) -> None:
        eterna = guion(Status.CONTINUE, Status.CONTINUE, Status.CONTINUE, Status.CONTINUE)

        out = await loop(eterna, max_steps=3)(ZERO)

        assert len(out.said) == 3
        assert out.fails == (Fail("loop", "tope de pasos: 3"),)

    async def test_lo_cortado_por_el_tope_sigue_votando_continue(self) -> None:
        out = await loop(guion(Status.CONTINUE, Status.CONTINUE), max_steps=1)(ZERO)

        assert out.vote is Status.CONTINUE

    async def test_el_gasto_se_acumula_entre_pasos(self) -> None:
        pasos = (Status.CONTINUE, Status.CONTINUE, Status.DONE)

        out = await loop(guion(*pasos, cuesta=10), max_steps=10)(ZERO)

        assert out.spent == 30

    async def test_el_presupuesto_corta_y_lo_reporta(self) -> None:
        cara = guion(*(Status.CONTINUE,) * 10, cuesta=40)

        out = await loop(cara, max_steps=10, budget=100)(ZERO)

        assert out.spent == 120
        assert out.fails == (Fail("loop", "presupuesto agotado: 120/100"),)

    async def test_el_presupuesto_no_le_gana_a_un_paso_que_ya_termino(self) -> None:
        out = await loop(guion(Status.DONE, cuesta=999), max_steps=10, budget=1)(ZERO)

        assert out.fails == ()


class TestLoopAnidado:
    """Un loop es una célula, así que cabe adentro de otro. Ahí va a vivir la
    recursión de RLM."""

    async def test_el_sub_loop_reporta_su_propio_veredicto(self) -> None:
        interno = loop(guion(Status.CONTINUE, Status.DONE), max_steps=10)

        out = await interno(ZERO)

        assert len(out.said) == 2
        assert out.vote is Status.DONE

    async def test_el_seguí_de_adentro_no_se_le_escapa_al_de_afuera(self) -> None:
        """Sin resumir el voto en el borde, el sub-loop le reportaría CONTINUE al
        padre para siempre y el padre daría vueltas hasta el tope."""
        externo = loop(loop(guion(Status.CONTINUE, Status.DONE), max_steps=10), max_steps=5)

        out = await externo(ZERO)

        assert len(out.said) == 2
        assert out.fails == ()

    async def test_el_gasto_del_sub_loop_sube_al_padre(self) -> None:
        interno = loop(guion(Status.CONTINUE, Status.DONE, cuesta=10), max_steps=10)

        out = await loop(interno, max_steps=5)(ZERO)

        assert out.spent == 20


class TestUnTurno:
    async def test_el_turno_entero_es_una_linea(self) -> None:
        """Lo que el agente agrega se junta con lo que ya había, con la misma
        operación que usan todos los combinadores."""
        agente = then(fanout(dice("busqué"), caido("web")), dice("respondo"))
        seen = Log(said=(Message(Role.USER, "hola"),))

        seen = merge(seen, await agente(seen))

        assert [m.text for m in seen.said] == ["hola", "busqué", "respondo"]
