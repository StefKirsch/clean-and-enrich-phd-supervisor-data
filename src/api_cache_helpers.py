import logging
from time import monotonic

import pyalex.api
import requests_cache


_LOGGER = logging.getLogger("openalex.http")
_ORIGINAL_GET_REQUESTS_SESSION = pyalex.api._get_requests_session
_PYALEX_SESSION_PATCHED = False

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
    respect_retry_after=False,
    max_retry_backoff=10,
):
    """
    Add finite timeouts, bounded retry delays, and request timing logs to PyAlex.

    PyAlex 0.18 does not pass a timeout to ``requests``. A network request can
    therefore wait indefinitely. This wrapper retains PyAlex's retry-enabled
    session while supplying a default timeout to every request it makes.

    urllib3 normally honors a server's ``Retry-After`` header without capping
    the requested delay. Disabling that behavior makes retries use the bounded
    exponential backoff configured in PyAlex instead.
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
                retries.respect_retry_after_header = respect_retry_after
                retries.backoff_max = max_retry_backoff

        original_request = session.request

        if not policy_logged:
            _LOGGER.info(
                "OpenAlex HTTP policy active: timeout=%s "
                "respect_retry_after=%s max_retry_backoff=%ss",
                timeout,
                respect_retry_after,
                max_retry_backoff,
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
                response = original_request(method, url, **kwargs)
            except Exception:
                _LOGGER.exception(
                    "OpenAlex request failed after %.1fs: method=%s url=%s",
                    monotonic() - started_at,
                    method,
                    url,
                )
                raise

            _LOGGER.info(
                "OpenAlex request finished in %.1fs: status=%s cache=%s url=%s",
                monotonic() - started_at,
                response.status_code,
                getattr(response, "from_cache", False),
                url,
            )
            return response

        session.request = request_with_timeout
        return session

    pyalex.api._get_requests_session = get_requests_session_with_timeout
    _PYALEX_SESSION_PATCHED = True

