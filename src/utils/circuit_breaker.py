import time
from functools import wraps
from typing import Any, Callable

from src.utils.logger import logger


class CircuitBreakerOpenException(Exception):
    pass


class AsyncCircuitBreaker:
    """
    A lightweight, zero-dependency async Circuit Breaker.
    Prevents cascading failures when a downstream service is down.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def _update_state(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit Breaker transitioned to HALF_OPEN state.")

    def record_success(self):
        if self.state in ["HALF_OPEN", "OPEN"]:
            logger.info("Circuit Breaker recovered. Transitioning to CLOSED.")
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            if self.state != "OPEN":
                logger.error(
                    f"Circuit Breaker tripped! Transitioning to OPEN state for {self.recovery_timeout}s."
                )
            self.state = "OPEN"

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            self._update_state()
            if self.state == "OPEN":
                raise CircuitBreakerOpenException(
                    f"Circuit Breaker is OPEN. Calls to {func.__name__} are blocked."
                )

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise e

        return wrapper


# Use AsyncCircuitBreaker as a per-instance attribute on each model class.
# Do NOT create a module-level singleton — a shared breaker causes unrelated
# models to trip each other's circuit.
# Example: self._breaker = AsyncCircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
