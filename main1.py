import requests
import csv
import time
import re
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup

# List of restaurant URLs to scrape
URLS = [
    "https://www.belairediner.nyc/menus ",
    "https://flamebroilerusa.com/menu/ ",
    "https://www.nikosgrilldanbury.com/ ",
    "https://www.cheflebanesemediterraneangrill.com/order",
    "https://www.asia-kitchen.com/menuview.html"
]

# CSV output file
OUTPUT_CSV = "restaurants_preprocessed.csv"

# Fields for CSV
FIELDNAMES = [
    "Restaurant Name",
    "Location",
    "Menu Item",
    "Description",
    "Price",
    "Features",
    "Operating Hours",
    "Contact"
]

# Preprocessing functions
def clean_text(text: str) -> str:
    """
    Remove HTML tags, normalize whitespace, lowercase.
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Normalize whitespace and lowercase
    text = ' '.join(text.split()).lower()
    return text


def clean_price(price_str: str) -> float:
    """
    Remove currency symbols and convert to float. Returns 0.0 on failure.
    """
    if not price_str:
        return 0.0
    # Remove non-numeric characters except dot
    cleaned = re.sub(r'[^[0-9.]', '', str(price_str))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def can_fetch(url: str) -> bool:
    """
    Check robots.txt to see if scraping is allowed.
    """
    parser = RobotFileParser()
    parsed = requests.utils.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser.set_url(robots_url)
    parser.read()
    return parser.can_fetch("*", url)


def scrape_restaurant(url: str):
    """
    Scrape one restaurant page and yield preprocessed row dicts.
    """
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

    # Extract restaurant-level info
    name = clean_text(soup.select_one("h1.restaurant-name").get_text(strip=True))  # placeholder selector
    location_elem = soup.select_one("p.location")
    location = clean_text(location_elem.get_text(strip=True)) if location_elem else ""
    hours_elem = soup.select_one("div.hours")
    hours = clean_text(hours_elem.get_text(strip=True)) if hours_elem else ""
    contact_elem = soup.select_one("span.contact")
    contact = clean_text(contact_elem.get_text(strip=True)) if contact_elem else ""
    features = [clean_text(li.get_text(strip=True)) for li in soup.select("ul.features li")]

    # Loop through each menu item
    for item in soup.select("div.menu-item"):
        title_elem = item.select_one("h2.item-title")
        title = clean_text(title_elem.get_text(strip=True)) if title_elem else ""
        desc_elem = item.select_one("p.item-desc")
        desc = clean_text(desc_elem.get_text(strip=True)) if desc_elem else ""
        price_elem = item.select_one("span.price")
        price_raw = price_elem.get_text(strip=True) if price_elem else ""
        price = clean_price(price_raw)

        yield {
            "Restaurant Name": name,
            "Location": location,
            "Menu Item": title,
            "Description": desc,
            "Price": price,
            "Features": "; ".join(features),
            "Operating Hours": hours,
            "Contact": contact
        }


def main():
    # Open CSV and write header
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        # Iterate over all URLs
        for url in URLS:
            print(f"Scraping {url} …")
            for row in scrape_restaurant(url):
                writer.writerow(row)
            time.sleep(1)  # polite delay between requests

    print(f"Scraping and preprocessing complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
