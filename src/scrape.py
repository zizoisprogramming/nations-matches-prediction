import sys
import argparse
import random
import time
import pandas as pd
from playwright.sync_api import sync_playwright
from difflib import SequenceMatcher
import numpy as np
import re
from time import sleep
import datetime

from src.helpers.constants import REQ_TO_TRAIN, DATA_PATH, NEW_DATA_PATH, TEST_DATA_PATH

TARGET_COMPETITIONS = [
    "CAF Africa Cup of Nations", "UEFA European Championship", "FIFA World Cup",
    "CONMEBOL Copa America", "UEFA Nations League", "AFC Asian Cup", "Concacaf Nations League"
]


MIN_REQUEST_INTERVAL = {
    "fifa": 3.0,
    "espn": 5.0,   
}

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "Washington D.C."
]

_last_request_time = {"fifa": 0.0, "espn": 0.0}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _throttle(site_key: str):
    """Sleep just long enough to respect MIN_REQUEST_INTERVAL for this site, plus jitter."""
    min_interval = MIN_REQUEST_INTERVAL.get(site_key, 3.0)
    elapsed = time.monotonic() - _last_request_time.get(site_key, 0.0)
    if elapsed < min_interval:
        sleep(min_interval - elapsed + random.uniform(0.2, 1.0))
    _last_request_time[site_key] = time.monotonic()


def _random_user_agent() -> str:
    return random.choice(USER_AGENTS)



CHALLENGE_TITLE_MARKERS = [
    "just a moment", "attention required", "are you a robot", "access denied",
    "please verify you are a human", "checking your browser", "request unsuccessful",
    "captcha",
]


def _diagnose_blocked(page) -> str:
    """
    Only called *after* we've already failed to find the real content on the page.
    Returns a short reason string if the page's <title> matches a known bot-check
    page, else "" (meaning: probably just no data for this date / selector drift).
    """
    try:
        title = (page.title() or "").strip()
    except Exception:
        title = ""
    print(f"  ℹ️  page title was: {title!r}")
    lowered = title.lower()
    for marker in CHALLENGE_TITLE_MARKERS:
        if marker in lowered:
            return f"title matches bot-check marker {marker!r}"
    return ""


def _check_response(response, site_key: str):
    """Raise if the navigation response itself signals throttling/blocking."""
    if response is None:
        return
    if response.status in (403, 429, 503):
        raise RuntimeError(f"{site_key} returned HTTP {response.status} (likely rate-limited or blocked)")


def with_retries(scrape_fn, *args, site_key: str, max_retries: int = 4, base_delay: float = 8.0, **kwargs):
    """
    Generic retry wrapper with exponential backoff + jitter for any scrape_* function.

    Retries on:
      - exceptions raised by the scrape function (including the blocked/HTTP-status
        checks the scrapers now raise themselves)
      - empty results, since an empty list on a date that should have matches often
        means we got throttled rather than there genuinely being nothing scheduled.
    """
    delay = base_delay
    result = []
    for attempt in range(1, max_retries + 1):
        try:
            result = scrape_fn(*args, **kwargs)
        except Exception as e:
            print(f"  ⚠️  {site_key} scrape attempt {attempt}/{max_retries} raised: {e}")
            result = []

        if result:
            return result

        if attempt < max_retries:
            sleep_for = delay + random.uniform(0, delay * 0.3)
            print(f"  ⏳ {site_key} scrape attempt {attempt}/{max_retries} returned nothing, "
                  f"retrying in {sleep_for:.1f}s...")
            sleep(sleep_for)
            delay *= 2

    return result


def is_target_competition(name: str) -> bool:
    
    name = name.lower()
    for comp in TARGET_COMPETITIONS:
        if _text_similarity(name, comp.lower()) > 0.9:
            return True

    return False

def is_usa_location(location: str) -> bool:
    location = location.lower()
    for state in US_STATES:
        if _text_similarity(location, state.lower()) > 0.9:
            return True

    return False

def scrape_fifa_matches(date_str: str):
    """
    date_str: YYYY-MM-DD  (e.g. 2026-07-19)
    """
    url = f"https://www.fifa.com/en/match-centre?date={date_str}"

    _throttle("fifa")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=_random_user_agent()
        )
        page = context.new_page()

        print(f"Navigating to {url} ...")
        # domcontentloaded (not networkidle): avoids hanging on background
        # requests that may never let the network go fully idle.
        response = page.goto(url, wait_until="domcontentloaded", timeout=120000)
        _check_response(response, "fifa")
        try:
            page.wait_for_selector('div[class*="fixtures-competition_competiotionCard__"]', timeout=15000)
        except Exception:
            reason = _diagnose_blocked(page)
            browser.close()
            if reason:
                raise RuntimeError(f"fifa blocked: {reason}")
            print("No competition cards found on this date (or page structure changed).")
            return []
        try:
            # Wait for match rows to appear (the grid container from your screenshot)
            try:
                page.wait_for_selector('div[class*="match-row_rowContents__"]', timeout=15000)
            except Exception:
                print("No match rows found on this date (or page structure changed).")
                browser.close()
                return []

            # Extract all competition cards
            cards = page.query_selector_all('div[class*="fixtures-competition_competiotionCard__"]')
            matches = []

            for card in cards:
                
                try:
                    # --- Competition name (lives on the card, not the match row) ---
                    name_el = card.query_selector('a[class*="fixtures-competition_competitionName__"]')
                    competition_name = name_el.inner_text().strip() if name_el else ""
                    
                    if not is_target_competition(competition_name):
                        continue
                    

                    # --- Match rows within this competition card ---
                    rows = card.query_selector_all('div[class*="match-row_rowContents__"]')

                    for idx, row in enumerate(rows, 1):
                        try:
                            # --- Home Team ---
                            home_el = row.query_selector('a[class*="match-row_homeTeam__"]')
                            home_name = ""
                            if home_el:
                                name_span = home_el.query_selector('span[class*="match-row_teamFullName__"]')
                                home_name = name_span.inner_text().strip() if name_span else ""

                            # --- Away Team ---
                            away_el = row.query_selector('a[class*="match-row_awayTeam__"]')
                            away_name = ""
                            if away_el:
                                name_span = away_el.query_selector('span[class*="match-row_teamFullName__"]')
                                away_name = name_span.inner_text().strip() if name_span else ""

                            # --- Scores ---
                            score_el = row.query_selector('a[class*="match-row_scores__"]')
                            home_score, away_score = None, None
                            if score_el:
                                spans = score_el.query_selector_all("span")
                                if len(spans) >= 3:
                                    home_score = spans[0].inner_text().strip()
                                    away_score = spans[2].inner_text().strip()
                                elif len(spans) == 1:
                                    # Sometimes scores are in a single element
                                    txt = spans[0].inner_text().strip()
                                    if "-" in txt:
                                        parts = txt.split("-")
                                        home_score, away_score = parts[0].strip(), parts[1].strip()

                            # --- Match Status (FT, Live, etc.) ---
                            status_el = row.query_selector('span[class*="live-match-indicator_badge__"]')
                            status = status_el.inner_text().strip() if status_el else "TBD"

                            matches.append({
                                "match_number": idx,
                                "tournament": competition_name,
                                "home_team": home_name,
                                "away_team": away_name,
                                "home_score": home_score,
                                "away_score": away_score,
                                "status": status,
                                "date": date_str
                            })

                        except Exception as e:
                            print(f"  ⚠️  Skipped row {idx}: {e}")
                            continue

                except Exception as e:
                    print(f"  ⚠️  Skipped competition card: {e}")
                    continue
        except Exception as e:
            print(e)

        browser.close()
        return matches

def scrape_espn(date_str: str):
    """
    date_str: YYYY-MM-DD (e.g. 2026-07-15)
 
    Scrapes ESPN's football fixtures page for a given date and returns every
    match found, grouped by competition, with the venue/city/region split out
    from the "LOCATION" column (e.g. "Mercedes-Benz Stadium, Atlanta, Georgia").
    """
    espn_date = date_str.replace("-", "")  # YYYYMMDD, e.g. 20260715
    url = f"https://www.espn.co.uk/football/fixtures/_/date/{espn_date}"

    _throttle("espn")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=_random_user_agent()
        )
        page = context.new_page()
 
        print(f"Navigating to {url} ...")
        # domcontentloaded (not networkidle): ESPN's page has ongoing background
        # requests (ads/analytics/live scores) that can prevent the network from
        # ever going idle, which was causing false "timeout" failures.
        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        _check_response(response, "espn")
 
        matches = []
 
        try:
            page.wait_for_selector('div.mt3', timeout=15000)
        except Exception:
            reason = _diagnose_blocked(page)
            browser.close()
            if reason:
                raise RuntimeError(f"espn blocked: {reason}")
            print("No fixtures container found on this date (or page structure changed).")
            return []

        try:
            # div.mt3 is the container holding one direct-child <div> per
            # tournament (each with its own title + table of fixtures).
            blocks = page.query_selector_all('div.mt3 > div')
 
            for block in blocks:
                try:
                    title_el = block.query_selector('div[class*="Table__Title"]')
                    competition_name = title_el.inner_text().strip() if title_el else ""
 
                    rows = block.query_selector_all('tbody[class*="Table__TBODY"] tr')
 
                    if not rows:
                        continue  # empty/non-tournament div, skip
 
                    for row in rows:
                        try:
                            # --- Home team lives in the events__col cell ---
                            home_col = row.query_selector('td[class*="events__col"]')
                            home_name = ""
                            if home_col:
                                links = [a for a in home_col.query_selector_all("a") if a.inner_text().strip()]
                                if links:
                                    home_name = links[-1].inner_text().strip()

                            # --- Away team + score live in the colspan__col cell ---
                            away_col = row.query_selector('td[class*="colspan__col"]')
                            away_name = ""
                            home_score, away_score = None, None
                            if away_col:
                                score_link = away_col.query_selector("a")
                                if score_link:
                                    score_text = score_link.inner_text().strip()
                                    sm = re.match(r"(\d+)\s*-\s*(\d+)", score_text)
                                    if sm:
                                        home_score, away_score = sm.group(1), sm.group(2)
                                team_links = [
                                    a for a in away_col.query_selector_all('span[class*="Table__Team"] a')
                                    if a.inner_text().strip()
                                ]
                                if team_links:
                                    away_name = team_links[-1].inner_text().strip()

                            if not home_name and not away_name:
                                continue  # header/spacer row, skip

                            # --- Match status (oddly lives in the teams__col cell, e.g. "FT") ---
                            status_el = row.query_selector('td[class*="teams__col"]')
                            status = status_el.inner_text().strip() if status_el else ""

                            # --- Venue / location ---
                            venue_el = row.query_selector('td[class*="venue__col"]')
                            location_text = venue_el.inner_text().strip() if venue_el else ""

                            venue_name, city, region = "", "", ""
                            if location_text:
                                parts = [p.strip() for p in location_text.split(",")]
                                if len(parts) >= 3:
                                    venue_name = ", ".join(parts[:-2])
                                    city = parts[-2]
                                    region = parts[-1]
                                elif len(parts) == 2:
                                    venue_name, city = parts[0], parts[1]
                                else:
                                    venue_name = location_text

                            # --- Attendance ---
                            att_el = row.query_selector('td[class*="attendance__col"]')
                            attendance = att_el.inner_text().strip() if att_el else ""

                            if is_usa_location(region):
                                city = region
                                region = "United States"

                            matches.append({
                                "home_team": home_name,
                                "away_team": away_name,
                                "home_score": home_score,
                                "away_score": away_score,
                                "status": status,
                                "venue": venue_name,
                                "city": city,
                                "country": region,
                                "attendance": attendance,
                            })

                        except Exception as e:
                            print(f"  ⚠️  Skipped ESPN row: {e}")
                            continue
 
                except Exception as e:
                    print(f"  ⚠️  Skipped ESPN competition block: {e}")
                    continue
 
        except Exception as e:
            print(e)
 
        browser.close()
        return matches
    
def  _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower().strip(), (b or "").lower().strip()).ratio()
 
 
def find_espn_match(date_str: str, match_data:dict, espn_matches=None, threshold: float = 0.6):

    if espn_matches is None:
        espn_matches = scrape_espn(date_str)
 
    best_match = None
    best_score = 0.0
 
    for m in espn_matches:
        # Try both orderings, since we don't know which given name is home/away.
        score_normal = ( _text_similarity(match_data["home_team"], m["home_team"]) +  _text_similarity(match_data["away_team"], m["away_team"])) / 2
        score_swapped = ( _text_similarity(match_data["home_team"], m["away_team"]) +  _text_similarity(match_data["away_team"], m["home_team"])) / 2
        score = max(score_normal, score_swapped)
 
        if score > best_score:
            best_score = score
            best_match = m
 
    if best_match and best_score >= threshold:
        return {
            "home_team": match_data["home_team"],
            "away_team": match_data["away_team"],
            "date": match_data["date"],
            "home_score": match_data["home_score"],
            "away_score": match_data["away_score"],
            "tournament": match_data["tournament"],
            "city": best_match["city"],
            "country": best_match["country"]
        }
 
    return None

 
def post_scrape(data: dict):
    ts = pd.to_datetime(data['date'])
    data['year'] = ts.year
    data['month'] = ts.month
    data['day'] = ts.day

def label_data(df: pd.DataFrame):
    df['result'] = np.select(
        [
            df['home_score'] > df['away_score'],
            df['home_score'] < df['away_score']
        ],
        [
            1,  
            2   
        ],
        default=0 
    )
def main():
    parser = argparse.ArgumentParser(description="Scrape FIFA Match Centre for a specific date")
    parser.add_argument("mode", help="Weekly, monthly or daily")
    args = parser.parse_args()
    if not mode in ["weekly", "monthly", "daily"]:
        print("Mode must be one of: weekly, monthly or daily")
        return

    df = pd.read_csv(DATA_PATH).drop_duplicates()
    label_data(df)
    test_df = pd.read_csv(TEST_DATA_PATH).drop_duplicates()

    if len(df) >= REQ_TO_TRAIN:
        n_from_df = int(0.7 * REQ_TO_TRAIN)
        n_from_test = int(0.3 * REQ_TO_TRAIN)

        df_new = df.iloc[:n_from_df]
        df_rest = df.iloc[n_from_df:]

        test_new = test_df.iloc[:n_from_test]
        test_rest = test_df.iloc[n_from_test:]

        new_data = pd.concat([df_new, test_new], ignore_index=True)
        new_data.to_csv(NEW_DATA_PATH, index=False)

        updated_test_df = pd.concat([test_rest, df_rest], ignore_index=True)
        updated_test_df.to_csv(TEST_DATA_PATH, index=False)

    mode = args.mode
    full_matches = []
    span_dict = {
        "weekly": 7,
        "monthly": 31,
        "daily": 1
    }
    try:
        span = span_dict[mode]
    except KeyError:
        print(f"Mode {mode} not supported")
        sys.exit(1)

    date = datetime.date.today() 
    for _ in range(span):
        date_str = date.strftime("%Y-%m-%d")
        matches = with_retries(scrape_fifa_matches, date_str, site_key="fifa", max_retries=4, base_delay=8.0)
        if not matches:
            continue
        for match in matches:
            post_scrape(match)

        espn_matches = with_retries(scrape_espn, date_str, site_key="espn", max_retries=4, base_delay=10.0)
        for match in matches:
            target = find_espn_match(date_str, match, espn_matches)
            if target is not None:
                post_scrape(target)
                full_matches.append(target)
        date -= datetime.timedelta(days=1)

    if not matches:
        print("\nNo matches found.")
        sys.exit(0)

    print(f"  FIFA Matches for {mode}")
    print(f"{'='*60}")
    for m in full_matches:
        score_line = f"{m['home_score']} - {m['away_score']}" if m['home_score'] else "vs"
        print(f"\n  Match: {m['home_team']} {score_line} {m['away_team']}")
        print("Date: ", m['date'])
        print(f"{m['tournament']} - {m['city']}, {m['country']}")
    print(f"\n{'='*60}")

    df = pd.concat([df, pd.DataFrame([m for m in full_matches])], ignore_index=True)

if __name__ == "__main__":
    main()