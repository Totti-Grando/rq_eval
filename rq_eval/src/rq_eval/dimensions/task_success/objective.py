"""§4 steps 1 + 3 — infer the objective and decompose outcomes [T3-gen]."""

from __future__ import annotations

from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.providers.base import GeneratorProvider


class ObjectiveInference:
    """[T3-gen] Infers the user's objective (intent, not literal words)."""

    def __init__(self, generator: GeneratorProvider, seed: int) -> None:
        """Inject the generator and pin the seed."""
        self._generator = generator
        self._seed = seed

    def infer(self, question: str) -> str:
        """Return the inferred objective text for ``question``."""
        return self._generator.generate(f"[[echo]] {{{{ {question} }}}}", seed=self._seed).text


class OutcomeDecomposer:
    """[T3-gen] Instantiates the template outcomes against this instance."""

    def __init__(self, generator: GeneratorProvider, seed: int) -> None:
        """Inject the generator and pin the seed."""
        self._generator = generator
        self._seed = seed

    def decompose(self, outcomes: list[Outcome], objective: str) -> list[Outcome]:
        """Return the required outcomes, instantiated for ``objective``."""
        instantiated: list[Outcome] = []
        for oc in outcomes:
            text = self._generator.generate(
                f"[[echo]] {{{{ {oc.text} }}}}", seed=self._seed
            ).text
            instantiated.append(Outcome(id=oc.id, text=text or oc.text, cues=oc.cues))
        return instantiated
