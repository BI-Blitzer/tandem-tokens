"""All ollamaq exception types. Every subsystem raises only these (or builtins)."""

from __future__ import annotations


class OllamaqError(Exception):
    """Base class for every error raised by ollamaq."""


class ConfigError(OllamaqError):
    """models.yaml missing, unparseable, or internally inconsistent."""


class UnknownModelError(ConfigError):
    """A job named a model that is not defined in models.yaml."""


class ModelDisabledError(ConfigError):
    """A job (or routing entry) targets a model with enabled: false."""


class UnknownTaskTagError(ConfigError):
    """model='auto' but the task_tag has no routing entry."""


class JobSpecError(OllamaqError):
    """A job spec file is missing, not JSON, or fails schema validation."""


class OllamaConnectionError(OllamaqError):
    """The Ollama server could not be reached at all."""


class OllamaHTTPError(OllamaqError):
    """The Ollama server answered with a non-200 status."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Ollama HTTP {status_code}: {body}")


class VerifierError(OllamaqError):
    """A verifier subprocess violated the protocol (bad JSON, timeout, crash)."""


class NotImplementedPhaseError(OllamaqError, NotImplementedError):
    """Feature is specified but scheduled for a later phase (see PLAN.md section 4)."""

    def __init__(self, phase: int, feature: str) -> None:
        self.phase = phase
        self.feature = feature
        super().__init__(
            f"{feature} is not implemented until Phase {phase} - see PLAN.md section 4"
        )
