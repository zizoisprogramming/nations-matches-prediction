import sys
import argparse
from playwright.sync_api import sync_playwright


def scrape_fifa_matches(date_str: str):
    """
    date_str: YYYY-MM-DD  (e.g. 2026-07-19)
    """
    url = f"https://www.fifa.com/en/match-centre?date={date_str}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0"
        )
        page = context.new_page()

        print(f"Navigating to {url} ...")
        page.goto(url, wait_until="networkidle", timeout=60000)

        # Wait for match rows to appear (the grid container from your screenshot)
        try:
            page.wait_for_selector('div[class*="match-row_rowContents__"]', timeout=15000)
        except Exception:
            print("No match rows found on this date (or page structure changed).")
            browser.close()
            return []

        # Extract all match rows
        rows = page.query_selector_all('div[class*="match-row_rowContents__"]')
        matches = []

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
                    "home": home_name,
                    "away": away_name,
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": status,
                })

            except Exception as e:
                print(f"  ⚠️  Skipped row {idx}: {e}")
                continue

        browser.close()
        return matches


# def main():
#     parser = argparse.ArgumentParser(description="Scrape FIFA Match Centre for a specific date")
#     parser.add_argument("date", help="Date in YYYY-MM-DD format")
#     args = parser.parse_args()

#     matches = scrape_fifa_matches(args.date)

#     if not matches:
#         print("\nNo matches found.")
#         sys.exit(0)

#     print(f"\n{'='*60}")
#     print(f"  FIFA Matches for {args.date}")
#     print(f"{'='*60}")
#     for m in matches:
#         score_line = f"{m['home_score']} - {m['away_score']}" if m['home_score'] else "vs"
#         print(f"\n  Match {m['match_number']}: {m['home']} {score_line} {m['away']}")
#         print(f"  Status: {m['status']}")
#     print(f"\n{'='*60}")
#     print(f"  Total matches: {len(matches)}")
#     print(f"{'='*60}")


# if __name__ == "__main__":
#     main()