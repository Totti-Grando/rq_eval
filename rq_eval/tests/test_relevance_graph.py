"""U2 — relevance graph substrate: edges, anchors, and the consistency stub."""

from __future__ import annotations

from rq_eval.config import load_config
from rq_eval.contracts import Claim
from rq_eval.dimensions.relevance.anchors import AnchorSelector
from rq_eval.dimensions.relevance.edges import Edge, EdgeBuilder
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.consistency import StubConsistencyProvider
from rq_eval.providers.factory import ProviderFactory


def _claim(cid: str, text: str) -> Claim:
    return Claim(
        id=cid, text=text, source_sentence=text, verifiable=True,
        decontextualized=True, extractor_version="claim-extractor-v1",
    )


def _grounding() -> object:
    return ProviderFactory(load_config()).build().grounding


def test_marker_without_entailment_is_not_an_edge() -> None:
    """A stated 'because' link whose premise doesn't entail the conclusion → no edge."""
    anchor = _claim("c1", "GDP fell three percent")
    asserted = _claim("c2", "Air quality worsened because of weak exports")
    builder = EdgeBuilder(_grounding(), T1Tools(), edge_tau=0.5)  # type: ignore[arg-type]
    edges = builder.build([anchor, asserted])
    # the marker is present (candidate prior) but it never becomes a confirmed edge
    assert T1Tools().has_discourse_marker(asserted.text)
    assert not any(e.src == "c2" and e.dst == "c1" for e in edges)


def test_entailment_confirms_support_edge() -> None:
    """A premise whose tokens cover the conclusion is a confirmed support edge."""
    premise = _claim("c1", "GDP fell three percent in 2024 due to weak exports")
    conclusion = _claim("c2", "GDP fell")
    builder = EdgeBuilder(_grounding(), T1Tools(), edge_tau=0.5)  # type: ignore[arg-type]
    edges = builder.build([premise, conclusion])
    assert any(e.src == "c1" and e.dst == "c2" for e in edges)


def test_centrality_promotes_low_seed_claim_to_anchor() -> None:
    """A borderline-on-ask claim heavily supported by others is promoted by centrality."""
    cfg = load_config()
    selector = AnchorSelector(
        _calibrator(cfg.relevance.anchor_alpha, cfg.conformal.min_calibration_n),
        centrality_min=cfg.relevance.anchor_centrality_min,
    )
    edges = [Edge("p1", "hub", 0.9, False), Edge("p2", "hub", 0.9, False),
             Edge("p3", "hub", 0.9, False)]
    result = selector.select(
        claim_ids=["p1", "p2", "p3", "hub"], seed_ids={"p1"}, edges=edges,
        confidences={"p1": 0.8, "hub": 0.1},
    )
    assert "hub" in result.anchor_ids  # never seeded, promoted by in-degree 3
    assert result.centrality["hub"] == 3


def test_anchor_recall_carries_conformal_band() -> None:
    """The anchor set reports a conformal recall band (abstains below min-n)."""
    cfg = load_config()
    selector = AnchorSelector(
        _calibrator(cfg.relevance.anchor_alpha, cfg.conformal.min_calibration_n),
        centrality_min=cfg.relevance.anchor_centrality_min,
    )
    result = selector.select(["a1"], {"a1"}, [], {"a1": 0.9})
    assert 0.0 < result.band_low < 1.0
    assert result.band_low == 1.0 - cfg.relevance.anchor_alpha
    assert result.abstained  # too few anchors to certify recall -> honest abstain


def test_consistency_stub_defaults() -> None:
    """The forward-declared stub never penalizes and always routes contradictions."""
    stub = StubConsistencyProvider()
    assert stub.edge_sound("premise", "conclusion") is True
    receipt = stub.route_contradiction("stranded claim", "anchor")
    assert receipt.routed is True


def _calibrator(alpha: float, min_n: int) -> object:
    from rq_eval.scoring.conformal import ConformalCalibrator

    return ConformalCalibrator(alpha=alpha, min_n=min_n)
