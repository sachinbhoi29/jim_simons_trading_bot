import feedparser
import urllib.parse
import csv
from datetime import datetime, timedelta, timezone, time
import pandas as pd
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
IST = ZoneInfo("Asia/Kolkata")

# Keywords to query separately
ALL_QUERIES = [
    "stock market", "NSE", "BSE", "Sensex", "Nifty", "RBI", "budget",
    "IPO", "SEBI", "global markets", "US Fed", "Federal Reserve",
    "crude oil", "OPEC", "geopolitical", "war", "tariffs", "inflation"
]


def build_rss_url(query):
    base_url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en"
    }
    encoded_params = urllib.parse.urlencode(params)
    return f"{base_url}?{encoded_params}"




def fetch_news_for_all_keywords(start_dt, end_dt):
    all_news = []
    print(f"\n📅 Fetching news from {start_dt} to {end_dt} IST in parallel...\n")

    def task(keyword):
        print(f"   🔍 Querying: \"{keyword}\"")
        return fetch_news_generic(keyword, start_dt, end_dt)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(task, keyword): keyword for keyword in ALL_QUERIES}
        for future in as_completed(futures):
            try:
                news = future.result()
                all_news.extend(news)
            except Exception as e:
                print(f"⚠️ Error fetching news for '{futures[future]}': {e}")

    return all_news


def fetch_news_generic(keyword, start_dt, end_dt):
    # Format date range for query
    # if os.name == "nt":  # Windows
    start_str = start_dt.strftime("%#d %b %Y")
    end_str = end_dt.strftime("%#d %b %Y")
    # else:  # Linux/Mac
        # start_str = start_dt.strftime("%-d %b %Y")
        # end_str = end_dt.strftime("%-d %b %Y")

    # Build a more specific query
    query = f'{keyword} news from {start_str} to {end_str}'
    print('query',query)
    rss_url = build_rss_url(query)
    feed = feedparser.parse(rss_url)

    start_dt_utc = start_dt.astimezone(timezone.utc)
    end_dt_utc = end_dt.astimezone(timezone.utc)
    news_items = []

    for entry in feed.entries:
        try:
            published_utc = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            continue

        if start_dt_utc <= published_utc <= end_dt_utc:
            news_items.append({
                "query": keyword,
                "title": entry.title,
                "link": entry.link,
                "published": published_utc.isoformat(),
                "summary": entry.get("summary", "").strip().replace('\n', ' ')
            })

    return news_items


def save_to_csv(news_items, filename):
    if not news_items:
        print("No news to save.")
        return

    keys = news_items[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(news_items)

    print(f"\n✅ Saved {len(news_items)} news items to {filename}")


# -------------------------
# Main block
# -------------------------
if __name__ == "__main__":
    # Define date and time window
    start_date = datetime(2025, 8, 20)
    end_date = datetime(2025, 8, 21)
    start_time = time(15, 15)
    end_time = time(9, 15)

    # Combine with timezone
    start_dt = datetime.combine(start_date, start_time, IST)
    end_dt = datetime.combine(end_date, end_time, IST)

    # Fetch and save news
    all_news = fetch_news_for_all_keywords(start_dt, end_dt)

    if all_news:
        df = pd.DataFrame(all_news)
        df['published'] = pd.to_datetime(df['published'], utc=True).dt.tz_convert(IST)
        df['published'] = df['published'].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        date_str = f"{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%Y%m%d_%H%M')}"
        filename = f"news_market_combined_{date_str}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n📁 News saved to file: {filename}")
    else:
        print("\n⚠️ No news items found in selected time window.")
