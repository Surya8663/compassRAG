"""
Circuit breaker state machine managing primary generation resilience (`GenerationCircuitBreaker`).
Transitions between CLOSED, OPEN, and HALF_OPEN states with consecutive failure counting
and time-based recovery windows.
"""

import logging
import time
from enum import StrEnum
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    """Possible states of the generation circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Exception raised when attempting primary execution while circuit breaker is OPEN."""
    pass


class GenerationCircuitBreaker:
    """
    Circuit breaker state machine for the Generation Service.
    Monitors primary LLM execution and prevents cascading failures when degraded,
    routing traffic to fallback mode while attempting periodic recovery trials.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0
        self._lock = Lock()

    def can_execute(self) -> bool:
        """
        Checks whether the primary circuit allows execution.
        Raises `CircuitBreakerOpenError` if state is OPEN and within reset window.
        If OPEN and `reset_timeout` has elapsed, transitions to HALF_OPEN and returns True.
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                elapsed = time.monotonic() - self.last_failure_time
                if elapsed >= self.reset_timeout:
                    logger.warning(
                        "Circuit breaker reset timeout (%.2fs) elapsed. Transition -> HALF_OPEN.",
                        self.reset_timeout,
                    )
                    self.state = CircuitState.HALF_OPEN
                    return True
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Active reset window ({elapsed:.2f}s / "
                    f"{self.reset_timeout:.2f}s)."
                )
            # HALF_OPEN state allows trial request
            return True

    def record_success(self) -> None:
        """Records a successful primary call, resetting failures to 0 and state to CLOSED."""
        with self._lock:
            if self.state != CircuitState.CLOSED or self.failure_count > 0:
                logger.info("Circuit breaker success recorded. State -> CLOSED, failures -> 0.")
                self.state = CircuitState.CLOSED
                self.failure_count = 0

    def record_failure(self, exc: Exception | None = None) -> None:
        """Records an execution failure. Trips to OPEN if threshold reached or if in HALF_OPEN."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            if self.state == CircuitState.HALF_OPEN:
                logger.error(
                    "Failure recorded during HALF_OPEN trial (`%s`). Tripping to OPEN.", exc
                )
                self.state = CircuitState.OPEN
            elif (
                self.failure_count >= self.failure_threshold
                and self.state == CircuitState.CLOSED
            ):
                logger.error(
                    "Failure count (%d) reached threshold (%d). Tripping CLOSED -> OPEN (`%s`).",
                    self.failure_count,
                    self.failure_threshold,
                    exc,
                )
                self.state = CircuitState.OPEN
            else:
                logger.warning(
                    "Primary generation failure recorded (%d / %d): %s",
                    self.failure_count,
                    self.failure_threshold,
                    exc,
                )
