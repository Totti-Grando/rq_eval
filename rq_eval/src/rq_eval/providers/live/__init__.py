"""Live provider implementations — Bedrock/Titan/Guardrail/fairseq/spaCy.

All heavy third-party imports are lazy (inside methods), so these classes are
import- and construct-safe on a machine without the live dependencies; the
imports fire only when a method is actually called (on the target machine).
"""
