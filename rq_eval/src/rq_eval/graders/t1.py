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

    def numeric_match(self, a: str, b: str, tolerance: float) -> bool:
        """True iff both parse to numbers within relative ``tolerance``.

        |na - nb| <= tolerance · max(|na|, |nb|); tolerance 0.0 == exact.
        Returns False if either side has no number.
        """
        na, nb = self.extract_number(a), self.extract_number(b)
        if na is None or nb is None:
            return False
        scale = max(abs(na), abs(nb))
        return abs(na - nb) <= tolerance * scale

    def citation_member(self, citation_id: str | None, allowed: set[str]) -> bool:
        """True iff ``citation_id`` is a non-empty member of ``allowed``."""
        return citation_id is not None and citation_id in allowed

    def is_atomic(self, text: str) -> bool:
        """True iff the text has no clause-joining conjunction / semicolon."""
        return _CONJUNCTION.search(text) is None

    def conjunction_split(self, text: str) -> list[str]:
        """Split on conjunctions/semicolons into candidate atomic parts."""
        return [p.strip() for p in _CONJUNCTION.split(text) if p.strip()]

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
