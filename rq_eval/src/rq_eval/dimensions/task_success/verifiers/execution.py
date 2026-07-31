"""[T1] executable/test verifier — run it, or a heuristic fallback (§4).

The design routes 'the fix runs' / 'the SQL returns the right rows' to real
execution (sandbox exec / unit test / recompute) — the big determinism win for
code/SQL. Running model-produced code is unsafe without a sandbox, so this is
gated by ``task_success.execution_sandbox``:

* sandbox OFF (default) -> a deterministic T1 heuristic on the answer text
  (code signals present AND a run/pass claim);
* sandbox ON -> delegate to the injected :class:`ExecutionSandbox` (wired on the
  target machine); falls back to the heuristic if none is provided.
"""

from __future__ import annotations

from typing import Protocol

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.dimensions.task_success.verifiers.base import Verifier, VerifyContext


class ExecutionSandbox(Protocol):
    """Runs an artifact and returns whether it satisfies the outcome."""

    def run(self, outcome: Outcome, ctx: VerifyContext) -> bool:
        """Execute and return the pass/fail result."""
        ...


class ExecutionVerifier(Verifier):
    """[T1] Executes the artifact (sandbox) or applies a text heuristic."""

    def __init__(
        self, logger: AtomLogger, sandbox_enabled: bool, sandbox: ExecutionSandbox | None = None
    ) -> None:
        """Inject logger, the sandbox toggle, and an optional sandbox impl."""
        self._logger = logger
        self._sandbox_enabled = sandbox_enabled
        self._sandbox = sandbox

    def verify(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """Run the artifact if a sandbox is enabled+available, else heuristic."""
        if self._sandbox_enabled and self._sandbox is not None:
            achieved = self._sandbox.run(outcome, ctx)
            method = "sandbox"
        else:
            achieved = self._heuristic(outcome, ctx.answer)
            method = "heuristic"
        return self._logger.record(
            subject=f"outcome:{outcome.id}", role="outcome",
            question=f"would run? {outcome.text}", tier="T1",
            verdict=achieved, weight=outcome.weight,
            evidence=f"verifier=executable method={method}",
            grader_id="task_success.executable", model="code", model_version="rq_eval",
        )

    @staticmethod
    def _heuristic(outcome: Outcome, answer: str) -> bool:
        low = answer.lower()
        signals = [str(s).lower() for s in outcome.params.get("signals", [])]
        run_claims = [str(s).lower() for s in outcome.params.get("run_claims", [])]
        has_code = any(s in low for s in signals) if signals else False
        claims_run = any(s in low for s in run_claims) if run_claims else False
        return has_code and claims_run
