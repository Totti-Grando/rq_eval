"""§1 accuracy — the per-node axiom-truth check (RQ §1, Layer 1).

Per claim, three **truth-only** booleans — ``grounded ∧ source-adequate ∧
attributed`` — AND-ed into one per-node ``axiom`` verdict (the protected floor;
no graph, no edges). Numeric claims fold an exact-match into grounded; a **bare**
claim (no source) routes to the ScoringJudge unsourced residual. **Responsiveness
is deliberately not here** — accuracy is truth; relevance owns responsiveness.
Equal weight per node.
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim, ContextChunk
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.source_attribution.citations import resolve_explicit
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.base import AttributionProvider, SourceQualityProvider

_CODE = ("code", "rq_eval")
_RESIDUAL_TRUTH = "[[affirm]] Is this claim true (no source available to ground against)?"


@dataclass(frozen=True, slots=True)
class ClaimAccuracyDeps:
    """Collaborators injected into :class:`ClaimAccuracy` (keeps arity sane)."""

    grounding: GroundingGrader
    attribution: AttributionProvider  # set-op over §1's support set (no NLI)
    residual_truth: JudgeGrader
    t1: T1Tools
    source_quality: SourceQualityProvider
    logger: AtomLogger
    numeric_tolerance: float
    source_adequate_default: bool = True  # Nexa profile default when no cited source
    grounded_export: GroundednessExport | None = None  # §1 import; None -> compute locally


class ClaimAccuracy:
    """Computes + logs the per-claim axiom-truth booleans and the node verdict."""

    def __init__(self, deps: ClaimAccuracyDeps) -> None:
        """Inject the collaborators bundle."""
        self._d = deps

    def evaluate_claim(
        self, claim: Claim, chunks: list[ContextChunk], cited: dict[str, str]
    ) -> list[AtomRecord]:
        """Return the claim's component atoms + one ``axiom`` node verdict atom."""
        source_text = " ".join(c.text for c in chunks)
        grounded, g_atoms = self._grounded(claim, chunks, source_text)
        sa_ok, sa_atom = self._source_adequate(claim, chunks)
        at_ok, at_atom = self._attributed(claim, cited)
        axiom = grounded and sa_ok and at_ok
        node = self._d.logger.record(
            subject=claim.id, role="axiom", question="grounded ∧ source-adequate ∧ attributed?",
            tier="code", verdict=axiom,
            evidence=f"grounded={grounded} adequate={sa_ok} attributed={at_ok}",
            grader_id="accuracy.axiom", model=_CODE[0], model_version=_CODE[1],
        )
        return [*g_atoms, sa_atom, at_atom, node]

    def _grounded(
        self, claim: Claim, chunks: list[ContextChunk], source_text: str
    ) -> tuple[bool, list[AtomRecord]]:
        d = self._d
        out: list[AtomRecord] = []
        # numeric edge: a numeric claim must exact-match a number in the source
        numeric_ok = True
        if source_text and d.t1.extract_number(claim.text) is not None:
            numeric_ok = d.t1.numeric_match(claim.text, source_text, d.numeric_tolerance)
            out.append(
                d.logger.record(
                    subject=claim.id, role="numeric", question="numeric exact-match", tier="T1",
                    verdict=numeric_ok, evidence=f"tol={d.numeric_tolerance}",
                    grader_id="accuracy.numeric", model=_CODE[0], model_version=_CODE[1],
                )
            )
        # grounded: import the per-claim atom from groundedness (§1) when present
        if d.grounded_export is not None and d.grounded_export.has(claim.id):
            atom = d.grounded_export.atom(claim.id)  # the SAME atom groundedness logged
            out.append(atom)
            return atom.verdict and numeric_ok, out
        if not source_text:  # bare claim -> unsourced residual truth-judge [T3]
            residual = d.residual_truth.judge(
                subject=claim.id, role="unsourced_residual", question=_RESIDUAL_TRUTH,
                context=claim.text, reference=None, tier="T3",
            )
            out.append(residual)
            return residual.verdict and numeric_ok, out
        atom = d.grounding.check(
            subject=claim.id, role="grounded", source=source_text, claim=claim.text
        )
        out.append(atom)
        return atom.verdict and numeric_ok, out

    def _source_adequate(
        self, claim: Claim, chunks: list[ContextChunk]
    ) -> tuple[bool, AtomRecord]:
        d = self._d
        id2chunk = {c.id: c for c in chunks}
        if claim.citation and claim.citation in id2chunk:
            adequate = d.source_quality.adequate(
                id2chunk[claim.citation], claim.text, chunks, claim_id=claim.id
            )
            evidence = "provider"
        else:  # no cited source -> Nexa-profile default
            adequate = d.source_adequate_default
            evidence = "default"
        atom = d.logger.record(
            subject=claim.id, role="source_adequate", question="source adequate?", tier="T1",
            verdict=adequate, evidence=evidence, grader_id="accuracy.source_quality",
            model=_CODE[0], model_version=_CODE[1],
        )
        return adequate, atom

    def _attributed(self, claim: Claim, cited: dict[str, str]) -> tuple[bool, AtomRecord]:
        d = self._d
        # cited set C = explicit ids in the source sentence + the pipeline-resolved cite
        c = resolve_explicit(claim.source_sentence, set(cited))
        if claim.citation and claim.citation in cited:
            c.add(claim.citation)
        if c:  # source-referencing claim -> attribution is C∩S (no NLI)
            res = d.attribution.attributed(claim.id, c)
            atom = d.logger.record(
                subject=claim.id, role="attributed", question="attributed? (C∩S≠∅)", tier="T2",
                verdict=res.attributed,
                evidence=f"label={res.label} conf={res.confidence:.4f} C={sorted(c)}",
                grader_id="accuracy.attributed", model=_CODE[0], model_version=_CODE[1],
            )
            return res.attributed, atom
        # no citation -> excluded from attribution (routes to the residual), don't penalize
        atom = d.logger.record(
            subject=claim.id, role="attributed", question="attributed?", tier="T2",
            verdict=True, evidence="no citation (excluded)", grader_id="accuracy.attributed",
            model=_CODE[0], model_version=_CODE[1],
        )
        return True, atom
