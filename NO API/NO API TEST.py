import asyncio
import csv
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from playwright.async_api import async_playwright
from PIL import Image

###############################################################################
# CONFIGURATION
###############################################################################
LAT_MIN = 38.210786
LAT_MAX = 38.223611
LON_MIN = -85.764151
LON_MAX = -85.755796
IMAGE_AMT = 10  # Number of points to generate in the grid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "Panoramas")
os.makedirs(IMAGE_DIR, exist_ok=True)
DEBUG_DIR = os.path.join(SCRIPT_DIR, "Debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

PANOLOG = os.path.join(SCRIPT_DIR, "PanoLog.csv")

TIMEOUT = 60000  # Timeout in milliseconds
RETRIES = 3  # Number of retries for navigation

###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def get_pano_id_from_url(url):
    """Extract pano ID from a Street View URL."""
    match = re.search(r'!1s([^!]+)!', url)
    return match.group(1) if match else None

def generate_coordinates(lat_min, lat_max, lon_min, lon_max, total_points):
    """Generate a grid of lat/lon coordinates."""
    area = (lat_max - lat_min) * (lon_max - lon_min)
    step = (area / total_points) ** 0.5
    latitudes = np.arange(lat_min, lat_max, step)
    longitudes = np.arange(lon_min, lon_max, step)

    coords = []
    for lat in latitudes:
        for lon in longitudes:
            coords.append((round(lat, 5), round(lon, 5)))
    return coords

async def safe_goto(page, url, retries, timeout):
    """Navigate to a URL with retries."""
    for attempt in range(retries):
        try:
            print(f"Attempting to navigate (attempt {attempt + 1}): {url}")
            await page.goto(url, timeout=timeout, wait_until="networkidle")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                return False

async def scrape_pano_id(page, lat, lon):
    """Scrape pano ID from Street View."""
    url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    success = await safe_goto(page, url, retries=RETRIES, timeout=TIMEOUT)

    if not success:
        print(f"Failed to load URL: {url}")
        return None, None

    # Take a debug screenshot
    debug_path = os.path.join(DEBUG_DIR, f"debug_{lat}_{lon}.png")
    await page.screenshot(path=debug_path)

    # Extract pano ID from the URL
    pano_id = get_pano_id_from_url(page.url)
    return pano_id, page.url

###############################################################################
# PANOLOG CSV HELPERS
###############################################################################
def load_panolog(csv_path):
    """Load existing pano data from CSV."""
    data = {}
    if os.path.isfile(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                data[(lat, lon)] = {
                    "pano_id": row["pano_id"],
                    "status": row["status"]
                }
    return data

def save_panolog(csv_path, data_dict):
    """Save updated pano data to CSV."""
    fieldnames = ["latitude", "longitude", "pano_id", "status"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (lat, lon), info in data_dict.items():
            writer.writerow({"latitude": lat, "longitude": lon, "pano_id": info["pano_id"], "status": info["status"]})

###############################################################################
# LIVE PROGRESS GRID
###############################################################################
def draw_progress_grid(coords, panolog_data, lat_min, lat_max, lon_min, lon_max):
    """Draw a live progress grid."""
    grid_size = int(len(coords) ** 0.5)
    grid = np.zeros((grid_size, grid_size), dtype=int)

    lat_steps = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_steps = np.linspace(lon_min, lon_max, grid_size + 1)

    for (lat, lon) in coords:
        lat_idx = np.searchsorted(lat_steps, lat) - 1
        lon_idx = np.searchsorted(lon_steps, lon) - 1
        if panolog_data.get((lat, lon), {}).get("status") == "Valid":
            grid[lat_idx, lon_idx] = 1

    plt.close("all")  # Ensure all previous figures are closed
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="Greys", origin="upper")
    plt.title("Grid Coverage Progress")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Coverage")
    plt.grid(color="black", linewidth=0.5)
    plt.pause(0.1)  # Allow real-time updates

###############################################################################
# MAIN SCRIPT LOGIC
###############################################################################
async def main():
    coords = generate_coordinates(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_AMT)
    total_points = len(coords)
    panolog_data = load_panolog(PANOLOG)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()

        for idx, (lat, lon) in enumerate(coords, 1):
            print(f"\nScraping ({lat}, {lon}) ({idx}/{total_points})...")
            if panolog_data.get((lat, lon), {}).get("status") == "Valid":
                print("  -> Already scraped. Skipping.")
                continue

            pano_id, url = await scrape_pano_id(page, lat, lon)
            if pano_id:
                print(f"  -> Found Pano ID: {pano_id}")
                panolog_data[(lat, lon)] = {"pano_id": pano_id, "status": "Valid"}
            else:
                print("  -> No Pano ID found.")
                panolog_data[(lat, lon)] = {"pano_id": "", "status": "Invalid"}

            draw_progress_grid(coords, panolog_data, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            save_panolog(PANOLOG, panolog_data)

        await browser.close()

    print("\nAll scraping complete!")

###############################################################################
# ENTRY POINT
###############################################################################
if __name__ == "__main__":
    asyncio.run(main())









