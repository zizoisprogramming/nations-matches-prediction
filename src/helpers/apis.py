


async def _api_get(page, url: str):
    try:
        response = await page.goto(url)
        if response.status == 200:
            return await response.json()
    except Exception:
        pass
    return None