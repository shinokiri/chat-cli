import httpx
from time import sleep
from typing import Callable, Iterable
from openai import APIConnectionError, APITimeoutError


RETRY_EXCEPTIONS = (
    httpx.TransportError,
    APIConnectionError,
    APITimeoutError,
)


def stream_with_retry(
    stream_factory: Callable[[], Iterable],
    *,
    max_retries: int = 15,
    sleep_seconds: float = 1.0,
):
    """
    Run a streaming generator with retry.

    - stream_factory: a zero-arg callable that returns a stream object
    - Retries only on network-like errors
    - Re-raises the last exception if all retries fail
    """

    for attempt in range(max_retries):
        try:
            with stream_factory() as stream:
                yield from stream
            return  # success, exit completely

        except RETRY_EXCEPTIONS:
            if attempt == max_retries - 1:
                raise
            sleep(sleep_seconds * min(max_retries, 2 ** attempt))  # exponential backoff