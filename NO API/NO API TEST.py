import asyncio
import csv
import re
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from playwright.async_api import async_playwright

###############################################################################
# CONFIGURATION
###############################################################################
LAT_MIN = 38.210786
LAT_MAX = 38.223611
LON_MIN = -85.764151
LON_MAX = -85.755796

IMAGE_AMT = 2500  # total approximate points in the grid
N_WORKERS = 4     # number of parallel tasks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(SCRIPT_DIR, "Debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

PANOLOG = os.path.join(SCRIPT_DIR, "PanoLog.csv")
FINAL_CSV = os.path.join(SCRIPT_DIR, "FinalMetadata.csv")

TIMEOUT = 4000
RETRIES = 3
FALLBACK_RETRIES = 2

# If PyCharm or your environment doesn’t show a window by default
matplotlib.use("Qt5Agg")
plt.ion()  # interactive mode for live updates

###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def get_pano_id_from_url(url):
    """Extract the pano_id from final Street View URL."""
    match = re.search(r'!1s([^!]+)!2', url)
    return match.group(1) if match else None

def factor_rows_cols(total_points):
    rows = int(np.sqrt(total_points))
    cols = rows
    while rows * cols < total_points:
        cols += 1
    return rows, cols

def generate_coordinates(lat_min, lat_max, lon_min, lon_max, total_points):
    """Create a grid of lat/lon in row-major order."""
    rows, cols = factor_rows_cols(total_points)
    lat_steps = np.linspace(lat_min, lat_max, rows)
    lon_steps = np.linspace(lon_min, lon_max, cols)

    coords = []
    for lat in lat_steps:
        for lon in lon_steps:
            coords.append((round(lat, 5), round(lon, 5)))
    return coords, rows, cols

def split_coords_among_workers(coords, n):
    """Split the coordinate list into n sublists, each ~equal in size."""
    chunk_size = len(coords) // n
    subsets = []
    start = 0
    for i in range(n):
        end = start + chunk_size
        if i == n - 1:
            end = len(coords)  # last chunk picks up remainder
        subsets.append(coords[start:end])
        start = end
    return subsets

async def safe_goto(page, url, retries, timeout):
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
    url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    print(f"scrape_pano_id - desired URL: {url}")
    success = await safe_goto(page, url, RETRIES, TIMEOUT)
    if not success:
        print("safe_goto => false, no success")
        return None, None

    print(f"scrape_pano_id - arrived URL: {page.url}")
    await page.wait_for_timeout(2000)
    final_url = page.url
    print(f"scrape_pano_id - final URL after wait: {final_url}")

    pano_id = get_pano_id_from_url(final_url)
    print(f"scrape_pano_id - extracted pano_id: {pano_id}")
    return pano_id, final_url

###############################################################################
# PANOLOG CSV
###############################################################################
def load_panolog(csv_path):
    data = {}
    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                data[(lat, lon)] = {
                    "pano_id": row["pano_id"],
                    "status": row["status"]
                }
    return data

def save_panolog(csv_path, data_dict):
    fieldnames = ["latitude", "longitude", "pano_id", "status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (lat, lon), info in data_dict.items():
            writer.writerow({
                "latitude": lat,
                "longitude": lon,
                "pano_id": info["pano_id"],
                "status": info["status"]
            })

###############################################################################
# COVERAGE & PLOTTING
###############################################################################
def init_coverage(rows, cols):
    return np.zeros((rows, cols), dtype=int)

def update_coverage_in_array(coverage, rows, cols, coords, lat, lon):
    """Just sets coverage[r,c] = 1 for the coordinate."""
    idx = coords.index((lat, lon))
    r = idx // cols
    c = idx % cols
    coverage[r, c] = 1

def draw_progress_grid(fig, ax, coverage):
    ax.clear()
    cax = ax.imshow(coverage, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    ax.set_title("Grid Coverage Progress (0=white, 1=black)")
    ax.grid(color="black", linewidth=0.5)

    if not hasattr(draw_progress_grid, "cbar"):
        draw_progress_grid.cbar = fig.colorbar(cax, ax=ax)
    else:
        draw_progress_grid.cbar.update_normal(cax)

    fig.canvas.draw()
    fig.canvas.flush_events()

###############################################################################
# Shared update coverage + plot function
###############################################################################
async def update_coverage_and_plot(coverage_lock, coverage, rows, cols, coords,
                                   lat, lon, fig, ax):
    """
    Lock-protected coverage update + coverage redraw.
    So multiple workers won't try to draw coverage at once.
    """
    async with coverage_lock:
        # 1) Update coverage array
        update_coverage_in_array(coverage, rows, cols, coords, lat, lon)
        # 2) Redraw coverage
        draw_progress_grid(fig, ax, coverage)

###############################################################################
# FINAL CSV
###############################################################################
def create_final_csv_from_panolog(panolog_data, output_csv):
    fieldnames = ["latitude", "longitude", "pano_id", "status"]
    rows = []
    for (lat, lon), info in panolog_data.items():
        rows.append({
            "latitude": lat,
            "longitude": lon,
            "pano_id": info["pano_id"],
            "status": info["status"],
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nFinal CSV created: {output_csv}")
    print(f"Total entries: {len(rows)}")

###############################################################################
# WORKER FUNCTION
###############################################################################
async def worker(worker_id, coords_subset, coverage_lock, coverage,
                 rows, cols, all_coords, panolog_data, page, fig, ax):
    """
    Each worker processes its subset of coords using a single shared 'page' or each worker's page.
    They call 'update_coverage_and_plot' with a lock to ensure the coverage array and plotting
    is not done concurrently.
    """
    print(f"Worker {worker_id} started with {len(coords_subset)} coords.")
    for (lat, lon) in coords_subset:
        # If already valid, skip
        if panolog_data.get((lat, lon), {}).get("status") == "Valid":
            continue

        pano_id, _ = await scrape_pano_id(page, lat, lon)
        if pano_id:
            panolog_data[(lat, lon)] = {"pano_id": pano_id, "status": "Valid"}
            # Protected update + plot
            await update_coverage_and_plot(
                coverage_lock, coverage, rows, cols, all_coords,
                lat, lon, fig, ax
            )
        else:
            panolog_data[(lat, lon)] = {"pano_id": "", "status": "Invalid"}

    print(f"Worker {worker_id} finished.")

###############################################################################
# MAIN
###############################################################################
async def main():
    coords, rows, cols = generate_coordinates(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_AMT)
    total_points = len(coords)
    print(f"Generated {total_points} points => {rows} rows x {cols} cols")

    panolog_data = load_panolog(PANOLOG)

    coverage = init_coverage(rows, cols)
    # Fill from existing log
    for (lat, lon), info in panolog_data.items():
        if info["status"] == "Valid" and (lat, lon) in coords:
            update_coverage_in_array(coverage, rows, cols, coords, lat, lon)

    # Start the coverage plot
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_progress_grid(fig, ax, coverage)

    # Split coords into subsets for N_WORKERS
    subsets = split_coords_among_workers(coords, N_WORKERS)

    # coverage_lock so only one draw at a time
    coverage_lock = asyncio.Lock()

    async with async_playwright() as pw:
        # Launch one browser
        browser = await pw.chromium.launch(headless=False)

        # Create pages
        pages = []
        for w_id in range(N_WORKERS):
            p = await browser.new_page()
            pages.append(p)

        # Make worker tasks
        tasks = []
        for w_id, subset in enumerate(subsets, start=1):
            t = asyncio.create_task(worker(
                worker_id=w_id,
                coords_subset=subset,
                coverage_lock=coverage_lock,
                coverage=coverage,
                rows=rows,
                cols=cols,
                all_coords=coords,
                panolog_data=panolog_data,
                page=pages[w_id-1],
                fig=fig,
                ax=ax
            ))
            tasks.append(t)

        # Wait for concurrency
        await asyncio.gather(*tasks)
        await browser.close()

    # Final coverage update on screen
    draw_progress_grid(fig, ax, coverage)
    save_panolog(PANOLOG, panolog_data)

    print("\nScraping complete! Building final CSV...")
    create_final_csv_from_panolog(panolog_data, FINAL_CSV)

    # Keep the plot
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())




