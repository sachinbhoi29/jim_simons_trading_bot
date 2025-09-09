import feedparser
import urllib.parse
import csv
from datetime import datetime, timedelta, timezone
import pandas as pd
from zoneinfo import ZoneInfo

def fetch_news_for_today_and_yesterday():
    base_url = "https://news.google.com/rss/search"
    
    # Keep query simpler & generic for stock market
    query = '"stock market" OR "NSE" OR "BSE" OR "Sensex" OR "Nifty" OR "RBI" OR "budget" OR "IPO" OR "SEBI"'
    
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en"
    }
    encoded_params = urllib.parse.urlencode(params)
    RSS_URL = f"{base_url}?{encoded_params}"

    feed = feedparser.parse(RSS_URL)
    news_items = []
    
    today_utc = datetime.now(timezone.utc).date()
    yesterday_utc = today_utc - timedelta(days=1)

    for entry in feed.entries:
        try:
            # Use feedparser’s built-in parsed time (more reliable than manual strptime)
            published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).date()
        except Exception:
            continue

        if published_dt in (today_utc, yesterday_utc):
            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published,
                "summary": entry.get("summary", "").strip().replace('\n', ' ')
            })
    return news_items


def fetch_news_for_today_and_yesterday_global_category():
    import re
    
    base_url = "https://news.google.com/rss/search"
    
    query = (
        '"stock market" OR "NSE" OR "BSE" OR "Sensex" OR "Nifty" OR "RBI" OR "budget" OR "IPO" OR "SEBI" '
        'OR "global markets" OR "US Fed" OR "Federal Reserve" OR "crude oil" OR "OPEC" OR "geopolitical" '
        'OR "war" OR "tariffs" OR "inflation"'
    )
    
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en"
    }
    encoded_params = urllib.parse.urlencode(params)
    RSS_URL = f"{base_url}?{encoded_params}"

    feed = feedparser.parse(RSS_URL)
    news_items = []
    
    today_utc = datetime.now(timezone.utc).date()
    yesterday_utc = today_utc - timedelta(days=1)

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
            published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).date()
        except Exception:
            continue

        if published_dt in (today_utc, yesterday_utc):
            title = entry.title
            summary = entry.get("summary", "").strip().replace('\n', ' ')
            category = categorize_news(title, summary)

            news_items.append({
                "published": entry.published,
                "title": title,
                "category": category,
                "summary": summary,
                "link": entry.link, 
            })
    return news_items    


def save_to_csv(news_items, filename):
    keys = news_items[0].keys() if news_items else ["title", "link", "published"]
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(news_items)

news = fetch_news_for_today_and_yesterday()
print(f"Fetched {len(news)} news items published today.")
for n in news[:5]:
    print(n)

if news:
    df = pd.DataFrame(news)

    df['published'] = pd.to_datetime(df['published'], utc=True)
    df['published'] = df['published'].dt.tz_convert('Asia/Kolkata')
    
    df['published'] = df['published'].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Save using today's IST date
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d")
    filename = f"generic_stock_news_{today_str}.csv"
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Saved news items to {filename}")
else:
    print("No news items found for today to save.")