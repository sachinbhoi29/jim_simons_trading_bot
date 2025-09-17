import feedparser
import urllib.parse
import csv
from datetime import datetime

# Define your backtesting date here (UTC)
BACKTEST_DATE = datetime(2025, 7, 29).date()

# Stock to backtest
STOCK = "Larsen "

BASE_QUERY_TEMPLATE = (
    '"{stock}" AND ('
    '"NSE earnings" OR "Q1" OR "Q2" OR "Q3" OR "Q4" OR "quarterly earnings" OR "financial results" OR '
    '"net profit" OR "net loss" OR "EPS" OR "revenue growth" OR "top line" OR "bottom line" OR '
    '"dividend" OR "interim dividend" OR "final dividend" OR "dividend cut" OR '
    '"bonus issue" OR "stock split" OR "buyback" OR "buyback announcement" OR '
    '"CEO resignation" OR "CFO resignation" OR "MD resignation" OR "new CEO" OR '
    '"promoter stake" OR "pledge of shares" OR "unpledge of shares" OR '
    '"fundraising" OR "board meeting" OR "AGM" OR "EGM" OR '
    '"merger" OR "acquisition" OR "M&A deal" OR "demerger" OR "spin-off" OR '
    '"joint venture" OR "strategic partnership" OR '
    '"SEBI notice" OR "IT raid" OR "ED raid" OR "court order" OR "litigation" OR '
    '"RBI restriction" OR "NCLT" OR "bankruptcy" OR "credit rating downgrade" OR '
    '"auditor comment" OR "auditor resignation"'
    ')'
)

def build_rss_url(query):
    base_url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en"
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def fetch_news_for_date(stock_name, target_date):
    """Fetch news for a specific stock and date."""
    query = BASE_QUERY_TEMPLATE.format(stock=stock_name)
    rss_url = build_rss_url(query)
    print(f"🔍 Backtesting news for: {stock_name} on {target_date}\n→ RSS URL: {rss_url}\n")

    feed = feedparser.parse(rss_url)
    news_items = []

    for entry in feed.entries:
        try:
            published_str = entry.published
            published_dt = datetime.strptime(published_str, "%a, %d %b %Y %H:%M:%S %Z").date()
        except Exception:
            continue

        if published_dt == target_date:
            news_items.append({
                "stock": stock_name,
                "title": entry.title,
                "link": entry.link,
                "published": published_str,
                "summary": entry.get("summary", "").strip().replace('\n', ' ')
            })

    print(f"   → {len(news_items)} news items found on {target_date}.\n")
    return news_items

def save_to_csv(news_items, filename):
    if not news_items:
        print("⚠️ No news to save.")
        return

    keys = news_items[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(news_items)
    print(f"✅ Saved {len(news_items)} news items to {filename}")

# Main execution
if __name__ == "__main__":
    print("🕵️ Backtesting stock-specific news from Google News RSS...\n")

    news = fetch_news_for_date(STOCK, BACKTEST_DATE)

    if news:
        filename = f"backtest_news_{STOCK.replace(' ', '_')}_{BACKTEST_DATE}.csv"
        save_to_csv(news, filename)
    else:
        print("❌ No news items found for the given date.")
