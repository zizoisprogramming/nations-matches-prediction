# After this many Sofascore 403s in a row we're being blocked (e.g. datacenter IP),
# so stop instead of spending minutes on requests that will all fail.
MAX_CONSECUTIVE_403 = 20
_consecutive_403 = 0


class SofascoreBlocked(Exception):
    pass


async def _api_get(page, url: str):
    global _consecutive_403
    try:
        response = await page.goto(url)
        if response.status == 200:
            _consecutive_403 = 0
            return await response.json()
        body = (await response.text())[:120].replace("\n", " ")
        print(f"  ⚠️  HTTP {response.status} for {url}: {body}")
        if response.status == 403:
            _consecutive_403 += 1
            if _consecutive_403 >= MAX_CONSECUTIVE_403:
                raise SofascoreBlocked(f"{_consecutive_403} Sofascore 403s in a row, last: {body}")
    except SofascoreBlocked:
        raise
    except Exception as e:
        print(f"  ⚠️  request failed for {url}: {e}")
    return None
