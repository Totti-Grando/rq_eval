"""§1 steps 1–4, 6–7 — the four booleans per claim (+ residual/inferred).

Per claim, in code:
  grounded ∧ source_adequate ∧ attributed ∧ responsive  → claim_correct
with the numeric edge (exact-match, not NLI) folded into grounded, the unsourced
residual truth-judge, and the inferred-claim flag. ``responsive`` is imported
from relevance — never recomputed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim, ContextChunk
from rq_eval.dimensions.accuracy.importance import ImportanceWeights
from rq_eval.dimensions.accuracy.stubs import InferenceValidityStub
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.base import SourceQualityProvider

_CODE = ("code", "rq_eval")
_RESIDUAL_TRUTH = "[[affirm]] Is this claim true (no source available to ground against)?"


@dataclass(frozen=True, slots=True)
class ClaimAccuracyDeps:
    """Collaborators injected into :class:`ClaimAccuracy` (keeps arity sane)."""

    grounding: GroundingGrader
    attribution: GroundingGrader
    residual_truth: JudgeGrader
    t1: T1Tools
    source_quality: SourceQualityProvider  # real §3 provider (no more stub)
    inference: InferenceValidityStub
    weights: ImportanceWeights
    logger: AtomLogger
    grounding_tau: float
    numeric_tolerance: float
    source_adequate_default: bool = True  # Nexa profile default when no cited source
    grounded_export: GroundednessExport | None = None  # §1 import; None -> compute locally


class ClaimAccuracy:
    """Computes + logs the per-claim accuracy booleans."""

    def __init__(self, deps: ClaimAccuracyDeps) -> None:
        """Inject the collaborators bundle."""
        self._d = deps

    def evaluate_claim(
        self,
        claim: Claim,
        chunks: list[ContextChunk],
        cited: dict[str, str],
        export: ResponsivenessExport,
    ) -> list[AtomRecord]:
        """Return the ordered atoms whose per-subject AND is claim_correct."""
        d = self._d
        w = d.weights.weight(claim.id)
        source_text = " ".join(c.text for c in chunks)
        atoms: list[AtomRecord] = []

        atoms.extend(self._grounded(claim, chunks, source_text, w))
        atoms.append(self._source_adequate(claim, chunks, w))
        atoms.append(self._attributed(claim, cited, w))
        atoms.append(self._responsive(claim, export, w))
        return atoms

    def _grounded(
        self, claim: Claim, chunks: list[ContextChunk], source_text: str, w: float
    ) -> list[AtomRecord]:
        d = self._d
        out: list[AtomRecord] = []
        # numeric edge: a numeric claim must exact-match a number in the source
        if source_text and d.t1.extract_number(claim.text) is not None:
            num_ok = d.t1.numeric_match(claim.text, source_text, d.numeric_tolerance)
            out.append(
                d.logger.record(
                    subject=claim.id, role="numeric", question="numeric exact-match", tier="T1",
                    verdict=num_ok, weight=w, evidence=f"tol={d.numeric_tolerance}",
                    grader_id="accuracy.numeric", model=_CODE[0], model_version=_CODE[1],
                )
            )
        # grounded: import the per-claim atom from groundedness (§1) when present
        if d.grounded_export is not None and d.grounded_export.has(claim.id):
            grounded = d.grounded_export.atom(claim.id)  # the SAME atom groundedness logged
            out.append(grounded)
        elif not source_text:
            # unsourced residual: truth-judge [T3]
            out.append(
                d.residual_truth.judge(
                    subject=claim.id, role="grounded", question=_RESIDUAL_TRUTH,
                    context=claim.text, weight=w, tier="T3",
                )
            )
            return out
        else:
            grounded = d.grounding.check(
                subject=claim.id, role="grounded", source=source_text, claim=claim.text, weight=w
            )
            out.append(grounded)
        # inferred flag: grounded by the whole context but by no single chunk
        if grounded.verdict and chunks:
            max_single = max(d.grounding.raw(c.text, claim.text) for c in chunks)
            if max_single < d.grounding_tau:
                valid = d.inference.valid(claim.text, source_text)
                out.append(
                    d.logger.record(
                        subject=claim.id, role="inferred", question="inference valid?", tier="T2",
                        verdict=valid, weight=w, evidence=f"max_single={max_single:.4f}",
                        grader_id="accuracy.inference_validity", model=_CODE[0],
                        model_version=_CODE[1],
                    )
                )
        return out

    def _source_adequate(
        self, claim: Claim, chunks: list[ContextChunk], w: float
    ) -> AtomRecord:
        d = self._d
        id2chunk = {c.id: c for c in chunks}
        if claim.citation and claim.citation in id2chunk:
            adequate = d.source_quality.adequate(id2chunk[claim.citation], claim.text, chunks)
            evidence = "provider"
        else:  # no cited source -> Nexa-profile default
            adequate = d.source_adequate_default
            evidence = "default"
        return d.logger.record(
            subject=claim.id, role="source_adequate", question="source adequate?", tier="T1",
            verdict=adequate, weight=w, evidence=evidence, grader_id="accuracy.source_quality",
            model=_CODE[0], model_version=_CODE[1],
        )

    def _attributed(self, claim: Claim, cited: dict[str, str], w: float) -> AtomRecord:
        if claim.citation and claim.citation in cited:
            return self._d.attribution.check(
                subject=claim.id, role="attributed", source=cited[claim.citation],
                claim=claim.text, weight=w,
            )
        return self._d.logger.record(
            subject=claim.id, role="attributed", question="attributed?", tier="T2",
            verdict=True, weight=w, evidence="no citation", grader_id="accuracy.attributed",
            model=_CODE[0], model_version=_CODE[1],
        )

    def _responsive(self, claim: Claim, export: ResponsivenessExport, w: float) -> AtomRecord:
        if export.has(claim.id):
            return export.atom(claim.id)  # the SAME atom relevance logged
        return self._d.logger.record(
            subject=claim.id, role="responsive", question="responsive? (no export)", tier="T2",
            verdict=True, weight=w, evidence="fallback", grader_id="accuracy.responsive_fallback",
            model=_CODE[0], model_version=_CODE[1],
        )
