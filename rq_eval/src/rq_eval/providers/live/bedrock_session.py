"""Lazy Bedrock client factory (build order B2, live path).

boto3 is imported inside methods so the mock path never imports it and live
providers can be *constructed* on a machine without boto3 — the import only
happens when a client is actually requested (i.e. on the target machine).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rq_eval.config import Config


class BedrockSession:
    """Builds and caches boto3 Bedrock clients from config (region/profile)."""

    def __init__(self, cfg: Config) -> None:
        """Store config; defer all boto3 work until a client is requested."""
        self._cfg = cfg
        self._runtime: Any | None = None
        self._control: Any | None = None

    def _session(self) -> Any:
        import boto3  # noqa: PLC0415 - lazy so mock mode never imports boto3

        return boto3.Session(
            region_name=self._cfg.aws.region,
            profile_name=self._cfg.aws.profile or None,
        )

    def runtime(self) -> Any:
        """bedrock-runtime client (Converse / InvokeModel / ApplyGuardrail)."""
        if self._runtime is None:
            self._runtime = self._session().client("bedrock-runtime")
        return self._runtime

    def control(self) -> Any:
        """Bedrock control-plane client (model/guardrail metadata for smoke)."""
        if self._control is None:
            self._control = self._session().client("bedrock")
        return self._control
