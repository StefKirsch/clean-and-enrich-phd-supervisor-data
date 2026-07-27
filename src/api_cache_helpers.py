import logging
from time import monotonic, sleep

import pyalex.api
import requests
import requests_cache
from requests.exceptions import RequestException


_LOGGER = logging.getLogger("openalex.http")
_ORIGINAL_GET_REQUESTS_SESSION = pyalex.api._get_requests_session
_PYALEX_SESSION_PATCHED = False


class OpenAlexDailyLimitError(RequestException):
    """Raised when OpenAlex reports that the daily API budget is exhausted."""

    def __init__(self, reset_seconds=None, response=None):
        self.reset_seconds = reset_seconds
        reset_message = (
            f" It resets in approximately {reset_seconds} seconds."
            if reset_seconds is not None
            else ""
        )
        super().__init__(
            "OpenAlex daily API budget exhausted."
            f"{reset_message} Cached requests remain available, but new API "
            "queries cannot complete until the budget resets or prepaid "
            "credit is available.",
            response=response,
        )


def _number_from_header(headers, name):
    value = headers.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seconds_from_header(headers, name="X-RateLimit-Reset"):
    value = _number_from_header(headers, name)
    return max(0, int(value)) if value is not None else None


def _daily_budget_exhausted(response):
    remaining = _number_from_header(
        response.headers,
        "X-RateLimit-Remaining",
    )
    if remaining is not None:
        return remaining <= 0

    # OpenAlex documents rate-limit headers on every response, but retaining a
    # conservative body fallback keeps quota handling useful through proxies
    # that strip non-standard headers.
    body = response.text.lower()
    return "daily" in body and ("budget" in body or "limit" in body)


def get_openalex_rate_limit_status(api_key, timeout=(5, 20)):
    """Return a fresh, sanitized summary from OpenAlex's rate-limit endpoint."""
    if not api_key:
        return None

    try:
        # The main cache never expires, so this status request must bypass it.
        with requests_cache.disabled():
            response = requests.get(
                "https://api.openalex.org/rate-limit",
                params={"api_key": api_key},
                timeout=timeout,
            )
    except RequestException:
        # Do not leak the API key through a Requests exception containing the
        # fully prepared URL.
        raise RequestException(
            "Could not check the current OpenAlex API budget."
        ) from None

    if not response.ok:
        raise RequestException(
            "The OpenAlex rate-limit check returned HTTP "
            f"{response.status_code}."
        )

    rate_limit = response.json().get("rate_limit", {})
    return {
        "daily_budget_usd": rate_limit.get("daily_budget_usd"),
        "daily_used_usd": rate_limit.get("daily_used_usd"),
        "daily_remaining_usd": rate_limit.get("daily_remaining_usd"),
        "prepaid_remaining_usd": rate_limit.get("prepaid_remaining_usd"),
        "resets_at": rate_limit.get("resets_at"),
        "resets_in_seconds": rate_limit.get("resets_in_seconds"),
        "endpoint_costs_usd": rate_limit.get("endpoint_costs_usd", {}),
    }


def ensure_openalex_budget_available(api_key, timeout=(5, 20)):
    """Fail before extraction when neither daily nor prepaid budget remains."""
    status = get_openalex_rate_limit_status(api_key, timeout=timeout)
    if status is None:
        return None

    daily_remaining = status.get("daily_remaining_usd")
    prepaid_remaining = status.get("prepaid_remaining_usd")
    if (
        daily_remaining is not None
        and float(daily_remaining) <= 0
        and (prepaid_remaining is None or float(prepaid_remaining) <= 0)
    ):
        raise OpenAlexDailyLimitError(
            reset_seconds=status.get("resets_in_seconds")
        )

    return status


def initialize_request_cache():
    """
    Initialize the request cache to store responses from OpenAlex in a SQLite database on disk (openalex_cache.sqlite).

    The cache is set to never expire, so it will continue to store responses indefinitely.  
    To clear the cache, call `requests_cache.clear()`.  To delete the cache, call `requests_cache.uninstall_cache()`.
    """
    requests_cache.install_cache(
        cache_name='openalex_cache',
        backend='sqlite',
        expire_after=requests_cache.NEVER_EXPIRE

    )


def configure_pyalex_http_timeout(
    connect_timeout=5,
    read_timeout=20,
    respect_retry_after=True,
    max_retry_backoff=10,
    max_rate_limit_retries=3,
):
    """
    Add finite timeouts, bounded retry delays, and request timing logs to PyAlex.

    PyAlex 0.18 does not pass a timeout to ``requests``. A network request can
    therefore wait indefinitely. This wrapper retains PyAlex's retry-enabled
    session while supplying a default timeout to every request it makes.

    HTTP 429 responses are handled separately so daily-budget exhaustion can
    stop immediately, while short-lived rate limits use bounded retries.
    """
    global _PYALEX_SESSION_PATCHED

    if _PYALEX_SESSION_PATCHED:
        return

    timeout = (connect_timeout, read_timeout)
    policy_logged = False

    def get_requests_session_with_timeout():
        nonlocal policy_logged

        session = _ORIGINAL_GET_REQUESTS_SESSION()

        # PyAlex mounts an HTTPAdapter containing urllib3's Retry object.
        # Bound both header-driven and exponential retry sleeps.
        for adapter in session.adapters.values():
            retries = getattr(adapter, "max_retries", None)
            if retries is not None:
                # urllib3 does not cap Retry-After sleeps. The wrapper handles
                # 429 Retry-After values itself below with max_retry_backoff.
                retries.respect_retry_after_header = False
                retries.backoff_max = max_retry_backoff
                # Let the wrapper below inspect 429 quota headers before any
                # retry. The adapter continues to retry PyAlex's 5xx codes.
                retries.status_forcelist = {
                    status
                    for status in retries.status_forcelist
                    if status != 429
                }

        original_request = session.request

        if not policy_logged:
            _LOGGER.info(
                "OpenAlex HTTP policy active: timeout=%s "
                "respect_retry_after=%s max_retry_backoff=%ss "
                "max_rate_limit_retries=%s",
                timeout,
                respect_retry_after,
                max_retry_backoff,
                max_rate_limit_retries,
            )
            policy_logged = True

        def request_with_timeout(method, url, **kwargs):
            kwargs.setdefault("timeout", timeout)
            started_at = monotonic()
            _LOGGER.info(
                "OpenAlex request started: method=%s url=%s timeout=%s",
                method,
                url,
                timeout,
            )
            try:
                rate_limit_attempt = 0
                while True:
                    response = original_request(method, url, **kwargs)
                    if response.status_code != 429:
                        break

                    reset_seconds = _seconds_from_header(response.headers)
                    if _daily_budget_exhausted(response):
                        _LOGGER.error(
                            "OpenAlex daily API budget exhausted: "
                            "remaining=%s reset_seconds=%s",
                            response.headers.get("X-RateLimit-Remaining"),
                            reset_seconds,
                        )
                        raise OpenAlexDailyLimitError(
                            reset_seconds=reset_seconds,
                            response=response,
                        )

                    if rate_limit_attempt >= max_rate_limit_retries:
                        response.raise_for_status()

                    retry_after = _seconds_from_header(
                        response.headers,
                        "Retry-After",
                    )
                    exponential_delay = min(
                        max_retry_backoff,
                        2 ** rate_limit_attempt,
                    )
                    delay = (
                        min(retry_after, max_retry_backoff)
                        if respect_retry_after and retry_after is not None
                        else exponential_delay
                    )
                    rate_limit_attempt += 1
                    _LOGGER.warning(
                        "OpenAlex returned a transient HTTP 429; retrying "
                        "in %ss (attempt %s/%s, remaining=%s)",
                        delay,
                        rate_limit_attempt,
                        max_rate_limit_retries,
                        response.headers.get("X-RateLimit-Remaining"),
                    )
                    sleep(delay)
            except OpenAlexDailyLimitError:
                raise
            except Exception:
                _LOGGER.exception(
                    "OpenAlex request failed after %.1fs: method=%s url=%s",
                    monotonic() - started_at,
                    method,
                    url,
                )
                raise

            from_cache = getattr(response, "from_cache", False)
            _LOGGER.info(
                "OpenAlex request finished in %.1fs: status=%s cache=%s "
                "daily_remaining=%s reset_seconds=%s",
                monotonic() - started_at,
                response.status_code,
                from_cache,
                (
                    "cached"
                    if from_cache
                    else response.headers.get("X-RateLimit-Remaining")
                ),
                (
                    "cached"
                    if from_cache
                    else response.headers.get("X-RateLimit-Reset")
                ),
            )
            return response

        session.request = request_with_timeout
        return session

    pyalex.api._get_requests_session = get_requests_session_with_timeout
    _PYALEX_SESSION_PATCHED = True

