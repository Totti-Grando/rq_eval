"""R7 — reforms end-to-end: fixtures + explanation + whole-run replay."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.contracts import Claim
from rq_eval.dimensions.relevance.claim_responsiveness import ClaimResponsiveness
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.fixtures import FixtureSuite
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.relevance_grader import RelevanceGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.base import EntailmentResult
from rq_eval.providers.factory import ProviderFactory
from rq_eval.runner import Evaluator
from rq_eval.scoring.formulas import default_registry


def _run(case_name: str, tmp_path: Path):
    cfg = load_config()
    case = next(c for c in FixtureSuite().cases() if c.name == case_name)
    store = JsonlAtomStore(tmp_path / f"{case_name}.jsonl")
    result = Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())
    return result, store


def test_explanation_summary_and_whole_run_replays(tmp_path: Path) -> None:
    """The read-only ExplanationJudge runs and the ReplayVerifier still reproduces all scores."""
    result, store = _run("aligned", tmp_path)
    assert result.summary  # ExplanationJudge produced prose
    # the summary is not in the store and not referenced by any formula
    assert result.summary not in {a.evidence for a in store.all()}
    assert ReplayVerifier(default_registry()).verify_run(list(result.results.values()), store)


def test_self_citation_flags_disinterest(tmp_path: Path) -> None:
    _result, store = _run("self_citation", tmp_path)
    di = next(a for a in store.all() if a.role == "sq_disinterested")
    assert di.verdict is False and di.tier == "T1"  # COI rule, no judge


def test_world_knowledge_unit_deferred_in_run(tmp_path: Path) -> None:
    """A double-NLI disagreement logs the decidability residual during a real run."""
    _result, store = _run("missing_facet", tmp_path)
    # bottom-up units for facets absent from the answer flip with the corpus -> residual fires
    residuals = [a for a in store.all() if a.role == "decidability_residual"]
    assert residuals and all(a.tier == "T3" for a in residuals)


class _FixedGrounding:
    """Grounding provider returning a fixed label (to dissociate NLI from lexical)."""

    def __init__(self, label: str) -> None:
        self._label = label

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        return EntailmentResult(label=self._label, raw_score=1.0 if self._label == "E" else 0.0)


def _responsive(tmp_path: Path, label: str) -> bool:
    cfg = load_config()
    store = JsonlAtomStore(tmp_path / f"{label}.jsonl")
    logger = AtomLogger(store, FixedClock())
    on_topic = RelevanceGrader(
        ProviderFactory(cfg).build().relevance, cfg.thresholds.relevance_tau, logger,
        ("mock", "mock"), "relevance.on_topic", 1,
    )
    on_ask_nli = GroundingGrader(_FixedGrounding(label), logger, ("nli", "nli"), "on_ask_nli", 1)
    cr = ClaimResponsiveness(
        on_topic, on_ask_nli, T1Tools(), logger, ("mock", "mock"), 1,
        lexical_min_overlap=1.1,  # lexical flag can never fire -> on_ask is NLI-only
    )
    export = ResponsivenessExport()
    claim = Claim(
        id="c1", text="Barcelona won the match", source_sentence="x", verifiable=True,
        decontextualized=True, extractor_version="claim-extractor-v1",
    )
    signals = cr.compute("Who won the match", [claim], export)
    return signals[0].responsive.verdict


def test_nli_label_drives_responsive(tmp_path: Path) -> None:
    """Flipping the on-ask NLI label flips responsive (lexical held off)."""
    assert _responsive(tmp_path, "E") is True
    assert _responsive(tmp_path, "N") is False
