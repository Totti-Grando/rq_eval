"""Live NLP — spaCy (en_core_web_lg) + coreferee (B2, live path).

Sentence segmentation [T1] and coreference resolution [T2] exactly as §0
specifies. spaCy/coreferee are imported lazily so this file is import-safe
without them installed; install.sh provisions the models on the target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rq_eval.providers.base import CorefResult, NlpProvider

if TYPE_CHECKING:
    from rq_eval.config import Config

_MODEL = "en_core_web_lg"


class SpacyNlpProvider(NlpProvider):
    """spaCy segmentation + coreferee coreference resolution."""

    def __init__(self, cfg: Config) -> None:
        """Store config; the spaCy pipeline is loaded lazily on first use."""
        self._cfg = cfg
        self._nlp: Any | None = None

    def _pipeline(self) -> Any:
        if self._nlp is None:
            import coreferee  # noqa: F401, PLC0415 - registers the pipe component
            import spacy  # noqa: PLC0415

            nlp = spacy.load(_MODEL)
            if "coreferee" not in nlp.pipe_names:
                nlp.add_pipe("coreferee")
            self._nlp = nlp
        return self._nlp

    def segment(self, text: str) -> list[str]:
        """[T1] spaCy sentence segmentation."""
        doc = self._pipeline()(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    def parse_clauses(self, sentence: str) -> list[str]:
        """[T1] ClausIE/PredPatt-style clause decomposition over the dependency parse.

        Each clause head (the ROOT verb plus any coordinated / subordinate /
        complement clause verbs) yields its subtree span as one content unit;
        overlapping shorter spans are dropped. No generation — this reads the
        dependency arcs only. Falls back to the whole sentence if no clause head
        is found.
        """
        doc = self._pipeline()(sentence)
        _clause_deps = {"conj", "advcl", "ccomp", "xcomp", "relcl", "acl"}
        heads = [
            tok
            for tok in doc
            if tok.pos_ in {"VERB", "AUX"}
            and (tok.dep_ == "ROOT" or tok.dep_ in _clause_deps)
        ]
        spans: list[tuple[int, int, str]] = []
        for head in heads:
            toks = sorted(head.subtree, key=lambda t: t.i)
            text = " ".join(t.text for t in toks).strip()
            if text:
                spans.append((toks[0].i, toks[-1].i, text))
        # drop a span fully contained inside another (keep the widest clauses)
        kept = [
            s
            for s in spans
            if not any(o is not s and o[0] <= s[0] and o[1] >= s[1] for o in spans)
        ]
        clauses = [text for _, _, text in sorted(set(kept))]
        return clauses if clauses else [sentence.strip()]

    def resolve_coref(self, text: str, context: str = "") -> CorefResult:
        """[T2] Resolve references in ``text`` using coreferee over context+text."""
        doc = self._pipeline()((context + "\n" + text) if context else text)
        resolved = text
        for chain in doc._.coref_chains:
            for mention in chain:
                for token_idx in mention.token_indexes:
                    head = doc._.coref_chains.resolve(doc[token_idx])
                    if head:
                        resolved = resolved.replace(
                            doc[token_idx].text, " ".join(h.text for h in head)
                        )
        return CorefResult(resolved_text=resolved.strip())
