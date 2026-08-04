"""§3 disinterest — the [T1] conflict-of-interest rule (R4).

Deterministic-first: a source is conflicted (disinterested = False) if its
domain/author is on the pinned COI denylist, or (affiliation rule) the source's
org matches the claim's subject entity — the self-citation case. Clearly
independent sources (internal corpus) are decisively disinterested. Everything
else is *ambiguous* and left to the sampled residual judge.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rq_eval.config import load_yaml
from rq_eval.contracts import ContextChunk

if TYPE_CHECKING:
    from rq_eval.config import Config

_ENTITY = re.compile(r"\b([A-Z][a-zA-Z0-9]+)")  # proper-noun-ish subject tokens


class CoiRule:
    """Loads the COI oracle and decides disinterest where the rule is decisive."""

    def __init__(self, cfg: Config) -> None:
        """Load the pinned denylist + the affiliation-rule toggle."""
        data = load_yaml(cfg.resolve(cfg.source_quality.coi_denylist))
        if not isinstance(data, dict):
            raise ValueError("coi_denylist.yaml must be a mapping")
        self._version = str(data["version"])
        self._deny_domains = {str(d).lower() for d in data.get("domains", [])}
        self._deny_authors = {str(a).lower() for a in data.get("authors", [])}
        self._affiliation = cfg.source_quality.affiliation_rule

    @property
    def version(self) -> str:
        """The pinned COI-oracle version."""
        return self._version

    def decide(self, source: ContextChunk, claim: str) -> tuple[bool | None, str]:
        """Return (disinterested, reason); ``None`` = ambiguous (defer to judge)."""
        domain = (source.domain or "").lower()
        author = (source.author or "").lower()
        if domain in self._deny_domains or author in self._deny_authors:
            return False, "denylist"
        if self._affiliation and self._affiliation_conflict(source, claim):
            return False, "affiliation"
        if source.domain is None and source.author is None:
            return True, "internal"  # internal corpus -> decisively independent
        return None, "ambiguous"

    def _affiliation_conflict(self, source: ContextChunk, claim: str) -> bool:
        org = f"{source.domain or ''} {source.author or ''}".lower()
        if not org.strip():
            return False
        return any(len(t) > 2 and t.lower() in org for t in _ENTITY.findall(claim))
