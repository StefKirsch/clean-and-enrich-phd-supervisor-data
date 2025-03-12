import requests_cache

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

