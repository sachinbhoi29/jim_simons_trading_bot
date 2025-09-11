import feedparser
import urllib.parse
import csv
from datetime import datetime, timedelta, timezone
import pandas as pd
from zoneinfo import ZoneInfo
import re

# Constants
IST = ZoneInfo("Asia/Kolkata")

# Static queries
INDIA_MARKET_QUERY = (
    '"stock market" OR "NSE" OR "BSE" OR "Sensex" OR "Nifty" OR '
    '"RBI" OR "budget" OR "IPO" OR "SEBI"'
)

GLOBAL_MARKET_QUERY = (
    '"stock market" OR "NSE" OR "BSE" OR "Sensex" OR "Nifty" OR "RBI" OR "budget" OR "IPO" OR "SEBI" '
    'OR "global markets" OR "US Fed" OR "Federal Reserve" OR "crude oil" OR "OPEC" OR "geopolitical" '
    'OR "war" OR "tariffs" OR "inflation"'
)




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


def fetch_news_generic(query, start_dt=None, end_dt=None):
    if not end_dt:
        end_dt = datetime.now(IST)
    if not start_dt:
        start_dt = end_dt - timedelta(days=1)

    start_dt_utc = start_dt.astimezone(timezone.utc)
    end_dt_utc = end_dt.astimezone(timezone.utc)

    rss_url = build_rss_url(query)
    feed = feedparser.parse(rss_url)
    news_items = []

    for entry in feed.entries:
        try:
            published_utc = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            continue

        if start_dt_utc <= published_utc <= end_dt_utc:
            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "published": published_utc.isoformat(),
                "summary": entry.get("summary", "").strip().replace('\n', ' ')
            })

    return news_items


def fetch_news_for_generic_stock(start_dt=None, end_dt=None):
    return fetch_news_generic(INDIA_MARKET_QUERY, start_dt, end_dt)


def fetch_news_for_global_category(start_dt=None, end_dt=None):
    if not end_dt:
        end_dt = datetime.now(IST)
    if not start_dt:
        start_dt = end_dt - timedelta(days=1)

    start_dt_utc = start_dt.astimezone(timezone.utc)
    end_dt_utc = end_dt.astimezone(timezone.utc)

    rss_url = build_rss_url(GLOBAL_MARKET_QUERY)
    feed = feedparser.parse(rss_url)
    news_items = []

    def categorize_news(title, summary):
        text = f"{title} {summary}".lower()
        if re.search(r"us fed|federal reserve|global market|opec|crude oil|inflation|geopolitical|war|tariff", text):
            return "Global"
        elif re.search(r"sensex|nifty|nse|bse|rbi|sebi|budget", text):
            return "India Market"
        elif re.search(r"ipo|earnings|profit|eps|dividend|stock split|merger|m&a|ceo", text):
            return "Stocks/Corporate"
        else:
            return "Other"

    for entry in feed.entries:
        try:
            published_utc = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            continue

        if start_dt_utc <= published_utc <= end_dt_utc:
            title = entry.title
            summary = entry.get("summary", "").strip().replace('\n', ' ')
            category = categorize_news(title, summary)

            news_items.append({
                "published": published_utc.isoformat(),
                "title": title,
                "category": category,
                "summary": summary,
                "link": entry.link,
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

    print(f"Saved {len(news_items)} news items to {filename}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    end_dt = datetime.now(IST)  
    start_dt = end_dt - timedelta(days=1)

    print(f"Fetching news from {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')} IST")

    # Generic stock market news
    news = fetch_news_for_generic_stock(start_dt, end_dt)

    # Or fetch categorized global market news
    # news = fetch_news_for_global_category(start_dt, end_dt)

    if news:
        df = pd.DataFrame(news)
        df['published'] = pd.to_datetime(df['published'], utc=True).dt.tz_convert(IST)
        df['published'] = df['published'].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        date_str = end_dt.strftime("%Y%m%d_%H%M")
        filename = f"generic_stock_news_{date_str}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved news items to {filename}")
    else:
        print("No news items found in selected time window.")
