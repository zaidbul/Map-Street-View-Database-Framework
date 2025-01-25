import asyncio
import csv
import re
import os
import io
import requests
import aiohttp
import hashlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from playwright.async_api import async_playwright

###############################################################################
# CONFIG / CONSTANTS
###############################################################################
LAT_MIN = 38.210786
LAT_MAX = 38.223611
LON_MIN = -85.764151
LON_MAX = -85.755796

IMAGE_AMT = 4000               # total approximate points in the grid
N_WORKERS = 6                # number of parallel scraping workers
TIMEOUT = 5000
RETRIES = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Where we store logs
PANOLOG = os.path.join(SCRIPT_DIR, "PanoLog.csv")

# We will store a new column "md5" here (we're no longer clearing this each run)
PANODOWNLOAD_LOG = os.path.join(SCRIPT_DIR, "PanoDownload.csv")

FINAL_CSV = os.path.join(SCRIPT_DIR, "FinalMetadata.csv")

# Where we store stitched panos
STITCH_DIR = os.path.join(SCRIPT_DIR, "StitchedPanos")
os.makedirs(STITCH_DIR, exist_ok=True)

# For real-time coverage
matplotlib.use("Qt5Agg")
plt.ion()  # interactive mode for coverage

# ThreadPool for CPU-bound stitching so we don't block the event loop
stitch_executor = ThreadPoolExecutor(max_workers=4)

###############################################################################
# 1) UTILITY: EXTRACT PANO ID
###############################################################################
def get_pano_id_from_url(url):
    """
    Example Street View URL might contain !1s{pano_id}!2...
    We extract that group.
    """
    match = re.search(r'!1s([^!]+)!2', url)
    return match.group(1) if match else None

###############################################################################
# 2) ASYNC TILE DOWNLOAD + CV2 STITCH LOGIC (BATCHED)
###############################################################################
CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

async def fetch_tile(session, cbk0_url, panoid, fallback=False):
    """
    Fetches a tile from cbk0 or falls back to googleusercontent.com for the full image if needed.
    """
    async with semaphore:
        for attempt in range(RETRIES):
            try:
                async with session.get(cbk0_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 400 and not fallback:
                        # Fall back to googleusercontent.com for the full image
                        print(f"[Fallback] cbk0 failed with 400. Switching to googleusercontent for panoid {panoid}.")
                        return await fetch_full_image(session, panoid)
                    else:
                        print(f"Tile fetch failed with status {response.status} on attempt {attempt + 1}.")
            except Exception as e:
                print(f"Error fetching tile {cbk0_url} on attempt {attempt + 1}: {e}")
            await asyncio.sleep(2)
        print(f"Failed to fetch tile {cbk0_url} after {RETRIES} attempts.")
        return None

async def fetch_full_image(session, panoid):
    """
    Fetches the entire panorama image from googleusercontent.com for a given panoid.
    """
    fallback_url = f"https://lh3.googleusercontent.com/p/{panoid}=w13312-h6656"
    try:
        async with session.get(fallback_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                print(f"[Fallback Success] Fetched full panorama for panoid {panoid}.")
                return await response.read()
            else:
                print(f"[Fallback Failed] Status {response.status} for panoid {panoid}.")
    except Exception as e:
        print(f"[Fallback Error] Error fetching full panorama for panoid {panoid}: {e}")
    return None




async def download_tiles_async(panoid, zoom, max_x, max_y):
    """
    Downloads tiles for cbk0 and falls back to googleusercontent if necessary.
    """
    tile_url_template = (
        "https://cbk0.google.com/cbk?output=tile&panoid="
        f"{panoid}&zoom={zoom}&x={{x}}&y={{y}}"
    )

    tile_dict = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        coords_list = []
        for x in range(max_x):
            for y in range(max_y):
                cbk0_url = tile_url_template.format(x=x, y=y)
                coords_list.append((x, y))
                tasks.append(fetch_tile(session, cbk0_url, panoid))

        # Gather all tile fetches in parallel
        results = await asyncio.gather(*tasks)

    # Store results in tile_dict
    for (x, y), data in zip(coords_list, results):
        tile_dict[(x, y)] = data

    # Check if all are None => total failure
    if all(v is None for v in tile_dict.values()):
        print(f"[Tile Download Failure] All tiles failed for panoid {panoid} at zoom {zoom}.")
        return None

    return tile_dict

def stitch_tiles_cv2(tile_dict, max_x, max_y, tile_size=512):
    """
    Stitches tile bytes in tile_dict into a single panorama using OpenCV (cv2).
    tile_dict[(x,y)] = tile_bytes

    Returns: a 3D NumPy array (H,W,3) or None if all tiles are invalid.
    """
    if not tile_dict or all(v is None for v in tile_dict.values()):
        return None

    pano_width = tile_size * max_x
    pano_height = tile_size * max_y

    # Create blank output. OpenCV uses BGR order, but that's fine for saving.
    panorama = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)

    for (x, y), tile_bytes in tile_dict.items():
        if tile_bytes is not None:
            try:
                arr = np.frombuffer(tile_bytes, np.uint8)
                tile_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # shape (h, w, 3)
                if tile_img is not None:
                    x_offset = x * tile_size
                    y_offset = y * tile_size
                    panorama[y_offset:y_offset+tile_img.shape[0],
                             x_offset:x_offset+tile_img.shape[1], :] = tile_img
            except Exception as e:
                print(f"Error decoding/stitching tile ({x},{y}): {e}")

    return panorama

###############################################################################
# 2B) MD5 HASHING FOR FINAL PANORAMA
###############################################################################
def compute_md5_for_stitched_image(stitched_array):
    """
    stitched_array: NumPy array of shape (height, width, 3).
    Returns an MD5 hex string for the raw bytes.
    """
    raw_bytes = stitched_array.tobytes()
    md5_hash = hashlib.md5(raw_bytes).hexdigest()
    return md5_hash

async def download_and_stitch_pano_cv2(lat, lon, pano_id, pano_download_data):
    """
    Attempts multiple zoom levels. For each zoom candidate, batch-download tiles,
    stitch them, compute MD5, check duplicates, and return the saved filename if unique.
    If all attempts fail, fallback to downloading the full panorama image.
    """
    zoom_candidates = [
        (5, 26, 13),
        (4, 13, 7),
        (3, 7, 4),
    ]

    for (zoom, max_x, max_y) in zoom_candidates:
        tile_dict = await download_tiles_async(pano_id, zoom, max_x, max_y)
        if tile_dict is None:
            continue

        loop = asyncio.get_running_loop()
        def stitch_job():
            return stitch_tiles_cv2(tile_dict, max_x, max_y, tile_size=512)

        stitched_result = await loop.run_in_executor(stitch_executor, stitch_job)
        if stitched_result is None:
            continue

        # Skip if it's basically an all-black image
        if np.count_nonzero(stitched_result) == 0:
            continue

        # Compute MD5
        pano_md5 = compute_md5_for_stitched_image(stitched_result)

        # Check if we already have this MD5 in *this run or past runs*
        if any(info.get("md5") == pano_md5 for info in pano_download_data.values()):
            print(f"[Dedup] Pano {pano_id} => MD5 {pano_md5} is a duplicate. Skipping.")
            return None

        # If not a duplicate, add to dictionary
        filename = f"{lat:.5f}_{lon:.5f}_{pano_id}.jpg"
        out_path = os.path.join(STITCH_DIR, filename)
        cv2.imwrite(out_path, stitched_result)

        print(f"Stitched pano saved: {out_path}")

        # Return the filename plus the MD5
        return (filename, pano_md5)

    # Fallback to full panorama download if all zoom attempts fail
    print(f"[Fallback] Tile downloads failed for panoid {pano_id}. Attempting full panorama.")
    async with aiohttp.ClientSession() as session:
        full_image = await fetch_full_image(session, pano_id)
        if full_image:
            pano_md5 = hashlib.md5(full_image).hexdigest()
            filename = f"{lat:.5f}_{lon:.5f}_{pano_id}_full.jpg"
            out_path = os.path.join(STITCH_DIR, filename)
            with open(out_path, "wb") as f:
                f.write(full_image)
            print(f"[Full Image] Panorama saved: {out_path}")
            return filename, pano_md5

    print(f"[Failure] No panorama could be downloaded for panoid {pano_id}.")
    return None


###############################################################################
# 3) GENERATE COORD GRID
###############################################################################
def factor_rows_cols(total_points):
    rows = int(np.sqrt(total_points))
    cols = rows
    while rows * cols < total_points:
        cols += 1
    return rows, cols

def generate_coordinates(lat_min, lat_max, lon_min, lon_max, total_points):
    rows, cols = factor_rows_cols(total_points)
    lat_steps = np.linspace(lat_min, lat_max, rows)
    lon_steps = np.linspace(lon_min, lon_max, cols)

    coords = []
    for lat in lat_steps:
        for lon in lon_steps:
            coords.append((round(lat, 5), round(lon, 5)))
    return coords, rows, cols

###############################################################################
# 4) CSV LOGS: PanoLog, PanoDownload (with new "md5" column)
###############################################################################
def load_panolog(csv_path):
    """
    Returns: { (lat, lon): {'pano_id':'', 'status':'Valid'/'Invalid' } }
    """
    data = {}
    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (float(row["latitude"]), float(row["longitude"]))
                data[key] = {
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

def load_pano_downloadlog(csv_path):
    """
    Returns: { (lat, lon): {'pano_id':'', 'filename':'', 'status':'', 'md5':''} }
    Gracefully handles older CSVs missing 'md5' by setting it to ''.
    """
    data = {}
    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (float(row["latitude"]), float(row["longitude"]))
                data[key] = {
                    "pano_id": row["pano_id"],
                    "filename": row["filename"],
                    "status": row["status"],
                    # Use row.get(...) to avoid KeyError if 'md5' column is missing
                    "md5": row.get("md5", "")
                }
    return data

def save_pano_downloadlog(csv_path, data_dict):
    """
    data_dict[(lat, lon)] = {
        "pano_id": pano_id,
        "filename": stitched_file,
        "status": "Downloaded" or "Failed",
        "md5": string
    }
    """
    fieldnames = ["latitude", "longitude", "pano_id", "filename", "status", "md5"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (lat, lon), info in data_dict.items():
            writer.writerow({
                "latitude": lat,
                "longitude": lon,
                "pano_id": info["pano_id"],
                "filename": info["filename"],
                "status": info["status"],
                "md5": info.get("md5", "")
            })

###############################################################################
# 5) COVERAGE & REAL-TIME PLOT
###############################################################################
def init_coverage(rows, cols):
    return np.zeros((rows, cols), dtype=int)

def update_coverage(coverage, coords, lat, lon, rows, cols):
    idx = coords.index((lat, lon))
    r = idx // cols
    c = idx % cols
    coverage[r, c] = 1

def draw_coverage(fig, ax, coverage):
    ax.clear()
    cax = ax.imshow(coverage, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    ax.set_title("Grid Coverage Progress (0=white, 1=black)")
    ax.grid(color="black", linewidth=0.5)

    if not hasattr(draw_coverage, "cbar"):
        draw_coverage.cbar = fig.colorbar(cax, ax=ax)
    else:
        draw_coverage.cbar.update_normal(cax)

    fig.canvas.draw()
    fig.canvas.flush_events()

###############################################################################
# 6) FINAL CSV FROM PanoLog
###############################################################################
def create_final_csv_from_panolog(panolog_data, output_csv):
    """
    Build a final CSV from the coverage log.
    """
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
# 7) SCRAPER WORKER (IMMEDIATE STITCH, IMMEDIATE LOGGING)
###############################################################################
async def scraper_worker(
    worker_id,
    coords_subset,
    coverage,
    coverage_lock,
    coords,
    rows,
    cols,
    pano_log_data,
    pano_download_data,
    page,
    fig,
    ax
):
    """
    Each worker:
      1) Checks if coordinate is already valid => skip
      2) If not, scrapes for a pano_id.
      3) If valid => update coverage + PanoLog => immediately write to CSV
         => check PanoDownload => if file missing => download_and_stitch => update PanoDownload => immediately write to CSV
    """
    print(f"Worker {worker_id} started with {len(coords_subset)} coords.")
    for (lat, lon) in coords_subset:
        # Check PanoLog => skip if status=Valid
        existing_log = pano_log_data.get((lat, lon), {})
        if existing_log.get("status") == "Valid":
            dl_info = pano_download_data.get((lat, lon), {})
            if dl_info.get("status") == "Downloaded":
                print(f"Worker {worker_id}: Skipping {lat},{lon} (Already valid/downloaded)")
                continue

        # Scrape
        url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        print(f"Worker {worker_id}: Goto => {url}")
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="networkidle")
        except Exception as e:
            print(f"Worker {worker_id}: Navigation fail at {lat},{lon}: {e}")
            pano_log_data[(lat, lon)] = {"pano_id": "", "status": "Invalid"}
            save_panolog(PANOLOG, pano_log_data)
            continue

        await page.wait_for_timeout(2000)
        final_url = page.url
        pano_id = get_pano_id_from_url(final_url)

        if pano_id:
            # Mark coverage=valid
            pano_log_data[(lat, lon)] = {"pano_id": pano_id, "status": "Valid"}
            save_panolog(PANOLOG, pano_log_data)

            async with coverage_lock:
                update_coverage(coverage, coords, lat, lon, rows, cols)
                draw_coverage(fig, ax, coverage)

            # Check if we've already downloaded the file
            dl_entry = pano_download_data.get((lat, lon), {})
            if dl_entry.get("status") == "Downloaded":
                existing_file = dl_entry.get("filename", "")
                if existing_file:
                    pathcheck = os.path.join(STITCH_DIR, existing_file)
                    if os.path.exists(pathcheck):
                        print(f"Worker {worker_id}: Already physically downloaded => {pathcheck}")
                        continue
                    else:
                        print(f"Worker {worker_id}: Missing physical => re-download {existing_file}")

            # Download & stitch
            print(f"Worker {worker_id}: Downloading & Stitching => {pano_id} for {lat},{lon}")
            result = await download_and_stitch_pano_cv2(lat, lon, pano_id, pano_download_data)
            if result:
                stitched_file, pano_md5 = result
                pano_download_data[(lat, lon)] = {
                    "pano_id": pano_id,
                    "filename": stitched_file,
                    "status": "Downloaded",
                    "md5": pano_md5
                }
            else:
                pano_download_data[(lat, lon)] = {
                    "pano_id": pano_id,
                    "filename": "",
                    "status": "Failed",
                    "md5": ""
                }

            save_pano_downloadlog(PANODOWNLOAD_LOG, pano_download_data)

        else:
            pano_log_data[(lat, lon)] = {"pano_id": "", "status": "Invalid"}
            save_panolog(PANOLOG, pano_log_data)

    print(f"Worker {worker_id} completed scraping.")

###############################################################################
# MAIN
###############################################################################
async def main():
    # 1) Build coords
    all_coords, rows, cols = generate_coordinates(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_AMT)
    subsets = np.array_split(all_coords, N_WORKERS)

    # 2) Load logs
    pano_log_data = load_panolog(PANOLOG)
    pano_download_data = load_pano_downloadlog(PANODOWNLOAD_LOG)

    # 3) coverage array
    coverage = init_coverage(rows, cols)
    # fill coverage from existing valid coords
    for (lat, lon), info in pano_log_data.items():
        if info["status"] == "Valid" and (lat, lon) in all_coords:
            update_coverage(coverage, all_coords, lat, lon, rows, cols)

    # 4) Define grid layout
    total_items = N_WORKERS + 1  # Add 1 for the graph
    columns = 3  # Number of columns in the grid
    grid_rows = (total_items + columns - 1) // columns  # Calculate rows to fit all items
    window_width = 500
    window_height = 500

    # Start Playwright
    async with async_playwright() as pw:
        pages = []

        for w_id in range(total_items):
            # Calculate grid position
            row = w_id // columns
            col = w_id % columns
            x_offset = col * window_width  # Horizontal position based on column
            y_offset = row * window_height  # Vertical position based on row

            if w_id < N_WORKERS:
                # Worker browsers
                browser = await pw.chromium.launch(
                    headless=False,
                    args=[
                        f"--window-size={window_width},{window_height}",
                        f"--window-position={x_offset},{y_offset}"
                    ]
                )
                page = await browser.new_page()
                await page.set_viewport_size({"width": window_width, "height": window_height})
                pages.append(page)
            elif w_id == N_WORKERS:
                # Graph window (use matplotlib interactive mode)
                print(f"Positioning graph window at ({x_offset}, {y_offset})")
                fig, ax = plt.subplots(figsize=(6, 6))  # Graph size (in inches)
                draw_coverage(fig, ax, coverage)
                plt.get_current_fig_manager().window.setGeometry(x_offset, y_offset, window_width, window_height)
                plt.ion()  # Interactive mode for real-time updates

        # 5) Start scraping tasks
        tasks = []
        for idx, subset in enumerate(subsets, start=1):
            t = asyncio.create_task(scraper_worker(
                worker_id=idx,
                coords_subset=subset,
                coverage=coverage,
                coverage_lock=asyncio.Lock(),
                coords=all_coords,
                rows=rows,
                cols=cols,
                pano_log_data=pano_log_data,
                pano_download_data=pano_download_data,
                page=pages[idx - 1],  # Pass the corresponding page
                fig=fig,
                ax=ax
            ))
            tasks.append(t)

        await asyncio.gather(*tasks)

        # Final coverage draw
        draw_coverage(fig, ax, coverage)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    asyncio.run(main())




