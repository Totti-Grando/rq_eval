"""T1 grader toolbox — deterministic, pure checks (build order B5).

Numeric exact-match (never cosine — "$1.2B" vs "$1.3B" must fail), citation-id
set-membership, atomicity / conjunction split, and length counts. All pure and
property-tested; no model calls.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_MAGNITUDE = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}
_CONJUNCTION = re.compile(r";|\band\b|\bbut\b|\bwhereas\b", re.IGNORECASE)
# opinion / hedge / hypothetical markers → not plausibly provable (VeriScore filter)
_HEDGE = re.compile(
    r"\b(maybe|perhaps|possibly|probably|likely|arguably|i think|i believe|in my opinion|"
    r"might|may|could|should|would|seems?|appears?|hypothetically|if\b)\b",
    re.IGNORECASE,
)
# Claimify-style abstractive placeholder: a bracketed span containing a space
# (e.g. "[a celebrity]") — distinct from a citation token like "[chunk-1]".
_ABSTRACTIVE = re.compile(r"\[[^\]]*\s[^\]]*\]")
_LEADING_PRONOUN = re.compile(
    r"^(he|she|it|they|this|that|these|those|him|her|them|his|its|their)\b", re.IGNORECASE
)
# discourse markers that *propose* a premise→conclusion edge (candidate prior,
# §3): noisy on their own, so an edge is only confirmed by entailment.
_DISCOURSE = re.compile(
    r"\b(because|since|due to|owing to|as a result|therefore|thus|hence|"
    r"consequently|tied to|linked to|caused by|leads? to|results? in|"
    r"resulting in|driven by|so that)\b",
    re.IGNORECASE,
)
# nested/abstractive predicates a positional parse can't cleanly triple ->
# routed to the pinned generator residual (Evidence §0)
_NESTING = re.compile(r"\b(that|which|who|whom|whose|whether)\b", re.IGNORECASE)
# deictic adverbs marking an indexical (incomplete) proposition (§0.3 type-3)
_DEIXIS = re.compile(r"\b(here|there|now|then|today|yesterday|tomorrow|currently|recently)\b",
                     re.IGNORECASE)
# a spatiotemporal filler in a sibling: a date/time, or a Capitalized place/entity
_FILLER = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|[A-Z][a-zA-Z]+)\b")
# capitalized function words that open sentences — never a real entity filler
_NOT_FILLER = frozenset(
    "The A An It This That These Those They He She We You I In On At For And But Or".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on for and or is are was were be been it its this that with as at by "
    "from into about how why what which who when where whom does did do can could should would "
    "will was were the".split()
)
# leading interrogative/auxiliary words dropped when templating an ask hypothesis
_LEADING_ASK = frozenset(
    "what which who whom whose when where why how is are was were do does did can could should "
    "would will".split()
)


class T1Tools:
    """Pure deterministic string/number checks used across dimensions."""

    def extract_number(self, text: str) -> float | None:
        """Parse the first number (commas, $, %, magnitude words) or None.

        "$1.2B" -> 1.2e9, "12 percent" -> 12.0, "1,200" -> 1200.0.
        """
        match = _NUM.search(text)
        if match is None:
            return None
        value = float(match.group().replace(",", ""))
        tail = text[match.end() :].strip().lower()
        for suffix, factor in _MAGNITUDE.items():
            if tail.startswith(suffix):
                return value * factor
        return value

    def extract_numbers(self, text: str) -> list[float]:
        """Parse every number in ``text`` (commas, $, %, magnitude words)."""
        out: list[float] = []
        for match in _NUM.finditer(text):
            value = float(match.group().replace(",", ""))
            tail = text[match.end() :].strip().lower()
            for suffix, factor in _MAGNITUDE.items():
                if tail.startswith(suffix):
                    value *= factor
                    break
            out.append(value)
        return out

    def numeric_match(self, a: str, b: str, tolerance: float) -> bool:
        """True iff ``a``'s number matches *any* number in ``b`` within ``tolerance``.

        |na - nb| <= tolerance · max(|na|, |nb|); tolerance 0.0 == exact. Checking
        against every number in ``b`` (not just its first) is what makes the edge
        correct for multi-figure sources. False if ``a`` has no number.
        """
        na = self.extract_number(a)
        if na is None:
            return False
        return any(
            abs(na - nb) <= tolerance * max(abs(na), abs(nb)) for nb in self.extract_numbers(b)
        )

    def citation_member(self, citation_id: str | None, allowed: set[str]) -> bool:
        """True iff ``citation_id`` is a non-empty member of ``allowed``."""
        return citation_id is not None and citation_id in allowed

    def is_atomic(self, text: str) -> bool:
        """True iff the text has no clause-joining conjunction / semicolon."""
        return _CONJUNCTION.search(text) is None

    def conjunction_split(self, text: str) -> list[str]:
        """Split on conjunctions/semicolons into candidate atomic parts."""
        return [p.strip() for p in _CONJUNCTION.split(text) if p.strip()]

    def is_verifiable(self, text: str) -> bool:
        """[T1] True iff the span is plausibly provable (VeriScore filter).

        Opinions, hedges, and hypotheticals carry lexical markers ("maybe",
        "might", "I think", "if …") and are routed away from truth scoring; the
        ambiguous remainder is left for an optional fixed ``[T2]`` classifier.
        """
        return _HEDGE.search(text) is None

    def is_abstractive_implied(self, text: str) -> bool:
        """[T1] True iff the span carries an abstractive placeholder to flag.

        Claimify's bracketed ``[a celebrity]`` marks a fact *implied* but not
        stated. These are flagged (routed out), never generated — the design's
        deterministic-decomposition rule. A bracketed citation token (no space,
        e.g. ``[chunk-1]``) is not abstractive.
        """
        return _ABSTRACTIVE.search(text) is not None

    def has_leading_pronoun(self, text: str) -> bool:
        """[T1] True iff the text still opens with an unresolved pronoun/mention.

        Used as the structural self-contained check after coref resolution (no
        judge): a resolved claim should not begin with "it"/"they"/"this"/… .
        """
        return _LEADING_PRONOUN.match(text.strip()) is not None

    def has_discourse_marker(self, text: str) -> bool:
        """[T1] True iff the text carries a premise→conclusion discourse marker.

        A cheap *candidate* prior for a support edge ("because", "tied to", …);
        noisy on its own, so an edge is confirmed only by entailment (§3).
        """
        return _DISCOURSE.search(text) is not None

    def is_indexical(self, text: str) -> bool:
        """[T1] True iff the claim is an *incomplete* proposition (§0.3 type-3).

        A leading unresolved pronoun ("it is dark") or a deictic adverb
        ("here"/"now"/…) marks a free spatiotemporal/deictic slot the claim needs
        a sibling to fill before it is groundable. (Live uses spaCy POS/dep tags.)
        """
        return self.has_leading_pronoun(text) or _DEIXIS.search(text) is not None

    def find_filler(self, text: str) -> str | None:
        """[T1] A spatiotemporal filler (date/time/Capitalized entity) in ``text``, or None.

        Skips capitalized function words that merely open a sentence ("The", "It", …)
        so the filler is a real place/date/entity, not sentence casing.
        """
        for m in _FILLER.finditer(text):
            token = m.group(1)
            if token not in _NOT_FILLER:
                return token
        return None

    def parse_triplet(self, text: str) -> tuple[str, str, str] | None:
        """[T1] Parse a clause into a subject|predicate|object tuple, or None.

        The deterministic parse-first path for Evidence §0 triplets (PredPatt/
        OpenIE-style). Returns ``None`` for the residual the parser can't cleanly
        triple — nested predicates ("that"/"which"/…) or empty text — which the
        caller routes to the pinned generator. The mock uses a positional
        heuristic (subject | predicate | rest); the live path uses the parse.
        """
        if _NESTING.search(text):
            return None
        toks = text.split()
        if not toks:
            return None
        subject = toks[0]
        predicate = toks[1] if len(toks) > 1 else ""
        obj = " ".join(toks[2:])
        return (subject, predicate, obj)

    def word_count(self, text: str) -> int:
        """Number of whitespace-delimited tokens."""
        return len(text.split())

    def ask_hypothesis(self, question: str) -> str:
        """[T1] Template a question's specific ask into a declarative hypothesis.

        Drops the trailing '?' and a single leading interrogative/auxiliary word
        (no generation), leaving the content the answer must address — the NLI
        hypothesis for the on-ask check (DIVER-QA).
        """
        text = question.strip().rstrip("?").strip()
        words = text.split()
        if words and words[0].lower() in _LEADING_ASK:
            words = words[1:]
        return " ".join(words) if words else text

    def key_term_overlap(self, a: str, b: str) -> float:
        """[T1] Coverage of ``a``'s content words by ``b`` ∈ [0, 1] (lexical flag)."""
        ta = {t for t in _WORD.findall(a.lower()) if t not in _STOP}
        if not ta:
            return 0.0
        tb = {t for t in _WORD.findall(b.lower()) if t not in _STOP}
        return len(ta & tb) / len(ta)
