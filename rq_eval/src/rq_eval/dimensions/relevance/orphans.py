"""§3 relevance — orphan resolution (the veracity layer) [T1 + T2].

Claims with no path to any anchor are **not** simply off-topic — the orphan
bucket holds three different things, separated by reusing existing checks:

* **off-topic** — fails the on-topic test against the question → the true false
  positive; penalized.
* **stranded / veracity-bearing** — on-topic *and* the directional NLI
  ``orphan → anchor`` finds it **supports or contradicts** an anchor even though
  the answer drew no edge. The highest-value catch: a true, on-topic fact the
  answer stranded that *undercuts* its own conclusion. Kept **relevant**; a
  contradiction is routed to the Reasoning ``ConsistencyProvider`` (+ completeness).
* **independent-background** — on-topic, no entailment relation to any anchor →
  relevant context, kept.

Relevance owns *structure*, not *validity*: it never scores soundness itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.contracts import Claim
from rq_eval.providers.base import GroundingProvider
from rq_eval.providers.consistency import ConsistencyProvider

OFF_TOPIC = "off_topic"
STRANDED_CONTRADICTION = "stranded_contradiction"
STRANDED_SUPPORT = "stranded_support"
BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class OrphanVerdict:
    """Classification of one unreachable claim + whether it stays relevant."""

    claim_id: str
    kind: str
    relevant: bool
    route_reason: str | None = None


class OrphanResolver:
    """[T1 on-topic + T2 orphan→anchor NLI] Classify + route unreachable claims."""

    def __init__(self, grounding: GroundingProvider, consistency: ConsistencyProvider) -> None:
        """Inject the shared NLI verifier and the Reasoning consistency provider."""
        self._grounding = grounding
        self._consistency = consistency

    def classify(self, claim: Claim, on_topic: bool, anchors: list[Claim]) -> OrphanVerdict:
        """Classify an orphan; route a stranded contradiction to Reasoning."""
        if not on_topic:
            return OrphanVerdict(claim.id, OFF_TOPIC, relevant=False)
        supports = False
        for anchor in anchors:
            label = self._grounding.entails(claim.text, anchor.text).label
            if label == "C":
                receipt = self._consistency.route_contradiction(claim.text, anchor.text)
                return OrphanVerdict(
                    claim.id, STRANDED_CONTRADICTION, relevant=True,
                    route_reason=receipt.reason,
                )
            supports = supports or label == "E"
        if supports:
            return OrphanVerdict(claim.id, STRANDED_SUPPORT, relevant=True)
        return OrphanVerdict(claim.id, BACKGROUND, relevant=True)
