async def _api_get(page, url: str):
    try:
        response = await page.goto(url)
        if response.status == 200:
            return await response.json()
        print(f"  ⚠️  HTTP {response.status} for {url}")
    except Exception as e:
        print(f"  ⚠️  request failed for {url}: {e}")
    return None
