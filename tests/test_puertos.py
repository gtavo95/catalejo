"""Los puertos, verificados por mypy y no por una afirmación en un docstring.

Estas funciones no asertan nada en runtime a propósito: lo que prueba que un
adaptador cumple el puerto es que `uv run mypy` pase con ellas escritas. Si
mañana alguien le saca el `aclose` a `OpenAI`, o le pone a `complete` un
parámetro que el puerto no declara, el build se cae acá y no en la primera
corrida contra la API.

Es la forma de tipado estructural: ningún adaptador hereda de `Model` ni de
`Provider`, y aun así los cumplen. Por eso agregar un proveedor no toca ni una
línea del motor.
"""

from __future__ import annotations

from catalejo.core import Message, Role
from catalejo.llm import Gemini, Model, OpenAI, Provider, Stub


def acepta_model(_: Model) -> None:
    """Lo que piden `worker` y `recurse`: solo `complete`."""


def acepta_provider(_: Provider) -> None:
    """Lo que pide la aplicación: además el nombre del modelo y cerrar el cliente."""


class TestLosAdaptadoresCumplenElPuerto:
    def test_gemini(self) -> None:
        gem = Gemini(api_key="k")

        acepta_model(gem)
        acepta_provider(gem)

    def test_openai(self) -> None:
        oai = OpenAI(api_key="k")

        acepta_model(oai)
        acepta_provider(oai)

    def test_el_stub_es_model_y_no_provider(self) -> None:
        """El doble de test no tiene cliente que cerrar.

        Obligarlo a inventarse un `aclose` sería la abstracción filtrándose hacia
        el lado de los tests, que es justo lo que el puerto angosto evita.
        """
        acepta_model(Stub("hola"))

        assert not hasattr(Stub("hola"), "aclose")


class TestElMotorNoNombraProveedores:
    async def test_el_worker_corre_contra_cualquier_model(self) -> None:
        """La prueba de que el puerto alcanza: el álgebra entera se testea sin red."""
        from catalejo.core import Log
        from catalejo.rlm import Handle, worker

        out = await worker(Stub("listo"), Handle())(Log(said=(Message(Role.USER, "hola"),)))

        assert out.said[0].text == "listo"
