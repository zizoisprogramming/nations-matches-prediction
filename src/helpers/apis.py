import asyncio

# After this many Sofascore 403s in a row we're being blocked (e.g. datacenter IP),
# so stop instead of spending minutes on requests that will all fail.
MAX_CONSECUTIVE_403 = 20
_consecutive_403 = 0

# Sofascore sometimes throttles by letting requests hang rather than refusing them.
# Use a short timeout with retries, and after several requests in a row fail
# completely, pause so it can cool down.
REQUEST_TIMEOUT_MS = 10_000
ATTEMPTS = 3
FAILURES_BEFORE_COOLDOWN = 5
COOLDOWN_SECONDS = 60
_consecutive_failures = 0


class SofascoreBlocked(Exception):
    pass


class RequestFailed(Exception):
    """The request itself failed (timeout, network error, 403/429/5xx), as opposed to
    Sofascore answering that the resource doesn't exist."""


async def _api_get(page, url: str, raise_on_failure: bool = False):
    """
    GET a Sofascore API URL through the browser page and return the JSON, or None.
    A 404 always returns None. Other failures return None too, unless raise_on_failure
    is set, in which case they raise RequestFailed so callers can tell "no data"
    apart from "couldn't ask".
    """
    global _consecutive_403, _consecutive_failures
    last_error = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = await page.goto(url, timeout=REQUEST_TIMEOUT_MS)
            if response.status == 200:
                _consecutive_403 = 0
                _consecutive_failures = 0
                return await response.json()
            body = (await response.text())[:120].replace("\n", " ")
            print(f"  ⚠️  HTTP {response.status} for {url}: {body}")
            if response.status == 404:
                _consecutive_failures = 0
                return None
            if response.status == 403:
                _consecutive_403 += 1
                if _consecutive_403 >= MAX_CONSECUTIVE_403:
                    raise SofascoreBlocked(f"{_consecutive_403} Sofascore 403s in a row, last: {body}")
                last_error = f"HTTP 403: {body}"
                break  # retrying a 403 right away doesn't help
            last_error = f"HTTP {response.status}"
        except SofascoreBlocked:
            raise
        except Exception as e:
            last_error = str(e).splitlines()[0]
            print(f"  ⚠️  request failed for {url} (attempt {attempt}/{ATTEMPTS}): {last_error}")
        if attempt < ATTEMPTS:
            await asyncio.sleep(5 * attempt)

    _consecutive_failures += 1
    if _consecutive_failures >= FAILURES_BEFORE_COOLDOWN:
        print(f"  ⏳ {_consecutive_failures} Sofascore requests in a row failed, pausing {COOLDOWN_SECONDS}s")
        await asyncio.sleep(COOLDOWN_SECONDS)
        _consecutive_failures = 0
    if raise_on_failure:
        raise RequestFailed(f"{url}: {last_error}")
    return None
