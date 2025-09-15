import feedparser
import urllib.parse
import csv
import pandas as pd
from datetime import datetime, timedelta, timezone,time
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


# Constants
IST = ZoneInfo("Asia/Kolkata")

earnings_keywords = [
    "quarterly results", "Q1 results", "Q2 results", "Q3 results", "Q4 results", "quarterly ", 
    "Q1 ", "Q2 ", "Q3 ", "Q4 ",
    "quarterly earnings", "net profit", "revenue", "EBITDA", "EPS", "financial results",
    "topline", "bottomline", "Q1FY", "Q2FY", "Q3FY", "Q4FY", "fy2025", "fy25"
]


# Sample list of 10 NSE stock names\
LARGE_CAP_STOCKS = [
    "Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", "Hindustan Unilever",
    "SBI", "Larsen & Toubro", "Axis Bank", "Bajaj Finance", "Kotak Mahindra Bank",
    "ITC", "Bharti Airtel", "Maruti Suzuki", "Sun Pharma", "Wipro", "HCL Technologies",
    "Mahindra", "NTPC", "Power Grid", "Tata Motors", "UltraTech Cement",
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
    "Mankind","Aurobindo Pharma", "Bank of Baroda", "Canara Bank", "Federal Bank", "L&T Finance",
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
# BASE_QUERY_TEMPLATE = (
#     '"{stock}" AND ('
#     '"NSE earnings" OR "quarterly earnings" OR "financial results" OR '
#     '"net profit" OR "EPS" OR "revenue growth" OR "top line" OR "bottom line" OR '
#     '"dividend" OR "interim dividend" OR "final dividend" OR "dividend cut" OR '
#     '"bonus issue" OR "stock split" OR "buyback" OR "buyback announcement" OR '
#     '"CEO resignation" OR "CFO resignation" OR "MD resignation" OR "new CEO" OR '
#     '"promoter stake" OR "pledge of shares" OR "unpledge of shares" OR '
#     '"fundraising" OR "board meeting" OR "AGM" OR "EGM" OR '
#     '"merger" OR "acquisition" OR "M&A deal" OR "demerger" OR "spin-off" OR '
#     '"joint venture" OR "strategic partnership" OR '
#     '"SEBI notice" OR "IT raid" OR "ED raid" OR "court order" OR "litigation" OR '
#     '"RBI restriction" OR "NCLT" OR "bankruptcy" OR "credit rating downgrade" OR '
#     '"auditor comment" OR "auditor resignation"'
#     ')'
# )
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

def fetch_news_for_stock(stock_name, start_dt=None, end_dt=None):
    if end_dt is None:
        end_dt = datetime.now(IST)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=1)

    start_dt_utc = start_dt.astimezone(timezone.utc)
    end_dt_utc = end_dt.astimezone(timezone.utc)

    # query = BASE_QUERY_TEMPLATE.format(stock=stock_name)
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
                    "stock": stock_name,
                    "title": entry.title,
                    "link": entry.link,
                    "published": published_str,
                    "summary": entry.get("summary", "").strip().replace('\n', ' ')
                })
        except Exception:
            continue

    return stock_news

def save_to_csv(news_items, filename):
    if not news_items:
        print("No news to save.")
        return

    keys = news_items[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(news_items)
    print(f"Saved {len(news_items)} news items to {filename}")

def very_important_news(df,date_str):
    pattern = re.compile("|".join([re.escape(k) for k in earnings_keywords]), re.IGNORECASE)
    df_earnings = df[df['title'].str.contains(pattern) | df['summary'].str.contains(pattern)]
    earnings_filename = f"stock_news_earnings_{date_str}.csv"
    df_earnings.to_csv(earnings_filename, index=False, encoding='utf-8')
    print(f"Saved earnings-related news to {earnings_filename}")

# Main execution
if __name__ == "__main__":
    # end_dt = datetime.now(IST)
    # start_dt = end_dt - timedelta(days=1)

    # Define dates
    start_date = datetime(2025, 9, 10)
    end_date = datetime(2025, 9, 11)
    # Define time of day (e.g., 09:15 AM)
    start_time_of_day = time(15, 15)
    end_time_of_day = time(9, 15)
    # Combine date + time with IST timezone
    start_dt = datetime.combine(start_date, start_time_of_day, IST)
    end_dt = datetime.combine(end_date, end_time_of_day, IST)
    print(f"Time window: {start_dt} to {end_dt}")

    all_news = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_stock = {
            executor.submit(fetch_news_for_stock, stock, start_dt, end_dt): stock
            for stock in STOCKS
        }

        for future in as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                stock_news = future.result()
                all_news.extend(stock_news)
                print(f"Fetched {len(stock_news)} news items for {stock}")
            except Exception as e:
                print(f"Error fetching news for {stock}: {e}")

    if all_news:
        df = pd.DataFrame(all_news)

        df['published'] = pd.to_datetime(df['published'], utc=True)
        df['published'] = df['published'].dt.tz_convert(IST)
        df['published'] = df['published'].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        date_str = end_dt.strftime("%Y%m%d_%H%M")
        very_important_news(df,date_str)
        filename = f"stock_news_{date_str}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved news items to {filename}")
    else:
        print("No news items found in selected time window.")
