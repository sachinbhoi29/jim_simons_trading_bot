import feedparser
import urllib.parse
import pandas as pd
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo
import re

# Timezone
IST = ZoneInfo("Asia/Kolkata")

# Pick just two stocks
STOCKS = ["ICICI Bank", "Reliance"]

# Query template
BASE_QUERY_TEMPLATE = '"{}" stock OR "{}" shares OR "{}" NSE OR "{}" BSE OR "{}" earnings OR "{}" news'

def build_rss_url(query):
    base_url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en"
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def fetch_news_for_stock(stock_name, start_dt, end_dt):
    """Fetch news for a given stock within a time window"""
    start_dt_utc = start_dt.astimezone(timezone.utc)
    end_dt_utc = end_dt.astimezone(timezone.utc)

    query = BASE_QUERY_TEMPLATE.format(*([stock_name] * 6))
    rss_url = build_rss_url(query)

    feed = feedparser.parse(rss_url)
    stock_news = []

    for entry in feed.entries:
        try:
            published_str = entry.published
            published_dt_utc = datetime.strptime(
                published_str, "%a, %d %b %Y %H:%M:%S %Z"
            ).replace(tzinfo=timezone.utc)

            if start_dt_utc <= published_dt_utc <= end_dt_utc:
                stock_news.append({
                    "date": start_dt.date().isoformat(),
                    "stock": stock_name,
                    "title": entry.title,
                    "link": entry.link,
                    "published": published_dt_utc.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "summary": entry.get("summary", "").strip().replace('\n', ' ')
                })
        except Exception:
            continue

    return stock_news


# ---------------- MAIN ----------------
if __name__ == "__main__":
    start_date = datetime(2025, 8, 15, tzinfo=IST)
    end_date = datetime(2025, 9, 30, tzinfo=IST)

    # Daily 9:15 AM to next day 9:15 AM window
    start_time = time(9, 15)
    end_time = time(9, 15)

    results = {stock: [] for stock in STOCKS}

    current_date = start_date
    while current_date <= end_date:
        day_start = datetime.combine(current_date.date(), start_time, IST)
        day_end = day_start + timedelta(days=1)

        print(f"\nFetching news for {current_date.date()}...")

        for stock in STOCKS:
            stock_news = fetch_news_for_stock(stock, day_start, day_end)
            results[stock].extend(stock_news)
            print(f"{stock}: {len(stock_news)} articles")

        current_date += timedelta(days=1)

    # Save into Excel (1 sheet per stock)
    filename = "stock_news_aug15_sep30.xlsx"
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for stock, news_items in results.items():
            if news_items:
                df = pd.DataFrame(news_items)
                df.sort_values(by="published", inplace=True)
                df.to_excel(writer, sheet_name=stock.replace(" ", "_")[:30], index=False)

    print(f"\nSaved news for {len(STOCKS)} stocks into {filename}")
