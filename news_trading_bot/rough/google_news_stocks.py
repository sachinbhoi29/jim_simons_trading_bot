import feedparser
import urllib.parse
import csv
from datetime import datetime, timedelta, timezone
import feedparser
from zoneinfo import ZoneInfo
import pandas as pd

# Sample list of 10 NSE stock names\

LARGE_CAP_STOCKS = [
    "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Hindustan Unilever",
    "SBI", "Larsen & Toubro", "Axis Bank", "Bajaj Finance", "Kotak Mahindra Bank",
    "ITC", "Bharti Airtel", "Maruti Suzuki", "Sun Pharma", "Wipro", "HCL Technologies",
    "Mahindra & Mahindra", "NTPC", "Power Grid", "Tata Motors", "UltraTech Cement",
    "Adani Ports", "Cipla", "Dr. Reddy's", "Nestle India", "Bajaj Finserv",
    "Divi's Laboratories", "JSW Steel", "Tata Steel", "Coal India", "Grasim",
    "HDFC Life", "Tech Mahindra", "UPL", "Britannia", "Eicher Motors", "Hindalco",
    "ONGC", "BPCL", "Apollo Hospitals", "Hero MotoCorp", "SBI Life", "Adani Enterprises",
    "Bajaj Auto", "IndusInd Bank", "ICICI Lombard", "Tata Consumer Products"
    "Asian Paints", "Shree Cement", "Dabur", "Pidilite Industries", "Godrej Consumer Products",
    "Havells India", "Torrent Pharma", "Colgate-Palmolive India", "Berger Paints",
    "DLF", "Zee Entertainment", "Cholamandalam Investment", "Mphasis", "Mindtree",
    "GAIL", "Indian Oil Corporation", "Petronet LNG", "REC", "PNB Housing Finance",
    "Bank of Baroda", "Federal Bank", "IDFC First Bank", "Gland Pharma", "Alkem Laboratories",
    "Max Healthcare", "ICICI Prudential Life", "ABB India", "Siemens", "Cummins India",
    "BHEL", "Ashok Leyland", "Container Corporation of India", "IRCTC", "LIC Housing Finance",
    "SRF", "Adani Green Energy", "Adani Transmission", "Adani Total Gas"
    ]

MID_CAP_STOCKS = [
    "Aurobindo Pharma", "Bank of Baroda", "Canara Bank", "Federal Bank", "L&T Finance",
    "Gland Pharma", "GMR Airports", "Gujarat Gas", "Indigo (InterGlobe Aviation)",
    "Page Industries", "Mphasis", "Dixon Technologies", "Polycab", "Voltas",
    "TVS Motor", "Balkrishna Industries", "Crompton Greaves", "Biocon", "Max Financial",
    "Cholamandalam Investment", "ICICI Securities", "Persistent Systems",
    "Muthoot Finance", "Astral Poly", "Jubilant FoodWorks", "ABB India", "Alkem Labs",
    "PI Industries", "Tata Elxsi", "Coromandel International", "Manappuram Finance",
    "Aditya Birla Capital", "Alembic Pharma", "Bata India", "LTIMindtree",
    "Godrej Properties", "Deepak Nitrite", "Torrent Pharma", "Supreme Industries",
    "Thermax", "RBL Bank", "Sundaram Finance", "Shriram Finance", "Laurus Labs",
    "United Breweries", "JSW Energy", "IDFC First Bank", "IEX", "Zydus Lifesciences"
        # Newly added mid-caps
    "KEI Industries", "Blue Star", "Trent", "Adani Wilmar", "Adani Power",
    "Gujarat Fluorochemicals", "Castrol India", "Indian Hotels Company",
    "Radico Khaitan", "APL Apollo Tubes", "Relaxo Footwears", "Fine Organic",
    "Hatsun Agro", "Prestige Estates", "Oberoi Realty", "Varun Beverages",
    "Emami", "Zee Entertainment", "BHEL", "IRCTC", "Container Corporation of India",
    "Indraprastha Gas", "Jindal Steel & Power", "Cummins India", "Kajaria Ceramics",
    "RITES", "SKF India", "Hindustan Aeronautics (HAL)", "CESC", "Endurance Technologies",
    "Grindwell Norton", "Gujarat Narmada Valley Fertilizers (GNFC)",
    "Navin Fluorine", "Linde India", "Indian Energy Exchange (IEX)", "AIA Engineering",
    "Kansai Nerolac Paints", "Schaeffler India", "Birlasoft", "Coforge",
    "Syngene International", "Narayana Hrudayalaya", "Fortis Healthcare",
    "Indraprastha Medical", "PVR Inox", "INOX Leisure", "Spandana Sphoorty",
    "City Union Bank", "Karur Vysya Bank", "South Indian Bank"
    ]

SMALL_CAP_STOCKS = ["Laurus Labs", "Godfrey Phillips", "Delhivery", "Aster DM Healthcare", 
                "Piramal Pharma", "Aptus Value Housing Finance", "Clean Science Tech", "COHANCE Lifesciences", 
                "City Union Bank", "PNC Infratech", "Tata Chemicals", "Hindustan Copper", 
                "Navin Fluorine International", "BEML", "Great Eastern Shipping", 
                "Schneider Electric Infrastructure", "Gujarat Mineral Devt Corp", 
                "CreditAccess Grameen", "Craftsman Automation", "Schaeffler India"

    # Newly added small caps
    "Mazagon Dock Shipbuilders", "Cochin Shipyard", "Garden Reach Shipbuilders",
    "Rail Vikas Nigam (RVNL)", "Ircon International", "IRFC", "HUDCO",
    "Engineers India", "NBCC India", "RITES",
    "Tanla Platforms", "Route Mobile", "Subex", "Intellect Design Arena",
    "HFCL", "Tejas Networks", "Sterlite Technologies",
    "Deepak Fertilisers", "Balrampur Chini Mills", "Dhampur Sugar Mills",
    "EID Parry", "Avanti Feeds", "Venky’s", "KRBL",
    "Somany Ceramics", "Orient Cement", "India Cements", "HeidelbergCement India",
    "Jindal Saw", "Welspun Corp", "Ratnamani Metals",
    "Finolex Cables", "Polycab Wires (moved up to mid/large now but started small)",
    "Borosil Renewables", "Sterling and Wilson Renewable Energy", "Inox Wind",
    "India Glycols", "Gujarat Alkalis", "DCW", "Thirumalai Chemicals",
    "Ruchira Papers", "JK Paper", "West Coast Paper Mills",
    "Dishman Carbogen Amcis", "Sequent Scientific", "Caplin Point Laboratories",
    "Granules India", "Natco Pharma", "Neuland Laboratories",
    "Welspun India", "Raymond", "Arvind Fashions", "Future Lifestyle Fashions",
    "Himadri Speciality Chemicals", "Oriental Carbon & Chemicals",
    "PNB Gilts", "Ujjivan Small Finance Bank", "Equitas Small Finance Bank",
    "CSB Bank", "DCB Bank", "Repco Home Finance"
                ]


STOCKS = LARGE_CAP_STOCKS + MID_CAP_STOCKS +SMALL_CAP_STOCKS

# Base query template to customize per stock
BASE_QUERY_TEMPLATE = (
    '"{stock}" AND ('
    '"NSE earnings" OR "quarterly earnings" OR "financial results" OR '
    '"net profit" OR "EPS" OR "revenue growth" OR "top line" OR "bottom line" OR '
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
    """Build the Google News RSS URL for a given query."""
    base_url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en"
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def fetch_news_for_stock_today_and_yesterday(stock_name):
    """Fetch today's and yesterday's news for a specific stock using a dynamic query."""
    query = BASE_QUERY_TEMPLATE.format(stock=stock_name)
    rss_url = build_rss_url(query)
    print(f"🔍 Searching news for: {stock_name}\n→ RSS URL: {rss_url}\n")

    feed = feedparser.parse(rss_url)
    today_utc = datetime.now(timezone.utc).date()
    yesterday_utc = today_utc - timedelta(days=1)
    stock_news = []

    for entry in feed.entries:
        try:
            published_str = entry.published
            published_dt = datetime.strptime(
                published_str, "%a, %d %b %Y %H:%M:%S %Z"
            ).date()
        except Exception:
            continue

        if published_dt in (today_utc, yesterday_utc):
            stock_news.append({
                "stock": stock_name,
                "title": entry.title,
                "link": entry.link,
                "published": published_str,
                "summary": entry.get("summary", "").strip().replace('\n', ' ')
            })

    print(f"   → {len(stock_news)} news items found for today & yesterday.\n")
    return stock_news

def save_to_csv(news_items, filename):
    """Save the collected news items to a CSV file."""
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
    print("📡 Fetching stock-specific news from Google News (today only)...\n")
    all_news = []

    for stock in STOCKS:
        news = fetch_news_for_stock_today_and_yesterday(stock)
        all_news.extend(news)

    if all_news:
        df = pd.DataFrame(all_news)

        # Convert "published" column to datetime and change to IST
        df['published'] = pd.to_datetime(df['published'], utc=True)
        df['published'] = df['published'].dt.tz_convert('Asia/Kolkata')
        
        # Optional: Format datetime as string (e.g., "2024-09-08 13:45:00 IST")
        df['published'] = df['published'].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Save using today's IST date
        today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d")
        filename = f"stock_news_{today_str}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved news items to {filename}")
    else:
        print("No news items found for today to save.")