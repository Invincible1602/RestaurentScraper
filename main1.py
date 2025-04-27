import requests
import csv
import time
import re
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup

URLS = [
    "https://www.belairediner.nyc/menus",
    "https://flamebroilerusa.com/menu/",
    "https://www.nikosgrilldanbury.com/",
    "https://www.cheflebanesemediterraneangrill.com/order",
    "https://www.asia-kitchen.com/menuview.html"
]

OUTPUT_CSV = "webscrap.csv"

FIELDNAMES = [
    "Restaurant",
    "Section",
    "Item",
    "Description",
    "Price"
]

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', str(text))
    text = ' '.join(text.split()).strip()
    return text

def clean_price(price_str: str) -> str:
    if not price_str:
        return ""
    cleaned = re.sub(r'[^[0-9.]', '', str(price_str))
    return cleaned

def can_fetch(url: str) -> bool:
    parser = RobotFileParser()
    parsed = requests.utils.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser.set_url(robots_url)
    parser.read()
    return parser.can_fetch("*", url)

def scrape_restaurant(url: str):
    if not can_fetch(url):
        print(f"Skipping {url} (disallowed by robots.txt)")
        return
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return
    soup = BeautifulSoup(resp.text, "html.parser")
    restaurant_elem = soup.select_one("h1.restaurant-name")
    restaurant = clean_text(restaurant_elem.get_text(strip=True)) if restaurant_elem else ""
    for item in soup.select("div.menu-item"):
        section_elem = item.select_one("div.section-name")
        section = clean_text(section_elem.get_text(strip=True)) if section_elem else ""
        item_title_elem = item.select_one("h2.item-title")
        item_name = clean_text(item_title_elem.get_text(strip=True)) if item_title_elem else ""
        desc_elem = item.select_one("p.item-desc")
        description = clean_text(desc_elem.get_text(strip=True)) if desc_elem else ""
        price_elem = item.select_one("span.price")
        price_raw = price_elem.get_text(strip=True) if price_elem else ""
        price = clean_price(price_raw)
        yield {
            "Restaurant": restaurant,
            "Section": section,
            "Item": item_name,
            "Description": description,
            "Price": price
        }

def main():
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        for url in URLS:
            print(f"Scraping {url} …")
            for row in scrape_restaurant(url):
                writer.writerow(row)
            time.sleep(1)
    print(f"✅ Scraping and preprocessing complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
