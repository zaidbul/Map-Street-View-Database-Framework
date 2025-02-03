import asyncio
import csv
import re
import os
import io
import requests
import aiohttp
import hashlib
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import contextily as ctx

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from playwright.async_api import async_playwright
from pyproj import Transformer

###############################################################################
# CONFIG / CONSTANTS
###############################################################################
LAT_MAX = 38.220810
LAT_MIN = 38.212298
LON_MIN = -85.762930
LON_MAX = -85.752885

IMAGE_AMT = 7500             # total approximate points in the grid
N_WORKERS = 20               # number of parallel scraping workers
TIMEOUT = 15000
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
wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def transform_bbox(lat_min, lat_max, lon_min, lon_max):
    """Convert WGS84 coordinates to Web Mercator"""
    x_min, y_min = wgs84_to_mercator.transform(lon_min, lat_min)
    x_max, y_max = wgs84_to_mercator.transform(lon_max, lat_max)
    return x_min, x_max, y_min, y_max

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
CONCURRENT_REQUESTS = 50
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

# Global tracker for fallback attempts
fallback_attempted = set()  # Global tracker

async def fetch_tile(session, cbk0_url, panoid, fallback=False):
    """
    Fetches a tile from cbk0 or falls back to googleusercontent.com for the full image if needed.
    """
    global fallback_attempted
    async with semaphore:
        for attempt in range(RETRIES):
            try:
                async with session.get(cbk0_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 400 and not fallback and panoid not in fallback_attempted:
                        # Fall back to googleusercontent.com for the full image
                        fallback_attempted.add(panoid)
                        print(f"[Fallback] cbk0 failed with 400. Switching to googleusercontent for panoid {panoid}.")
                        full_image = await fetch_full_image(session, panoid)
                        if full_image:
                            return full_image  # Use the full image data
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
    Downloads tiles for cbk0, triggering fallback immediately if any tile fails.
    If fallback succeeds, logs and exits without continuing tile-based downloads.
    """
    tile_url_template = (
        "https://cbk0.google.com/cbk?output=tile&panoid="
        f"{panoid}&zoom={zoom}&x={{x}}&y={{y}}"
    )

    async with aiohttp.ClientSession() as session:

        # 1) Fetch the first tile synchronously
        first_tile_url = tile_url_template.format(x=0, y=0)
        print(f"[Tile Check] Trying first tile for panoid={panoid} at zoom={zoom}: {first_tile_url}")
        first_tile_data = await fetch_tile(session, first_tile_url, panoid)

        if first_tile_data is None:
            # Trigger fallback immediately
            print(f"[Zoom {zoom}] First tile failed for panoid={panoid}. Trying fallback.")
            fallback_image = await fetch_full_image(session, panoid)
            if fallback_image:
                print(f"[Fallback Success] Full image fetched for panoid={panoid}.")
                return {"full_image": fallback_image}
            else:
                print(f"[Zoom {zoom}] Fallback also failed for panoid={panoid}.")
                return None

        # 2) First tile succeeded => start downloading the rest
        tile_dict = {(0, 0): first_tile_data}
        for x in range(max_x):
            for y in range(max_y):
                # Skip (0,0) => already fetched
                if x == 0 and y == 0:
                    continue

                tile_url = tile_url_template.format(x=x, y=y)
                tile_data = await fetch_tile(session, tile_url, panoid)

                # Trigger fallback if any tile fails
                if tile_data is None:
                    print(f"[Zoom {zoom}] Tile ({x}, {y}) failed for panoid={panoid}. Triggering fallback.")
                    fallback_image = await fetch_full_image(session, panoid)
                    if fallback_image:
                        print(f"[Fallback Success] Full image fetched for panoid={panoid} after tile failure.")
                        return {"full_image": fallback_image}
                    else:
                        print(f"[Zoom {zoom}] Fallback also failed for panoid={panoid}.")
                        return None

                # Add tile to the dictionary
                tile_dict[(x, y)] = tile_data

    return tile_dict

def stitch_tiles_gpu(tile_dict, max_x, max_y, tile_size=512):
    """
    Stitches tiles on GPU using PyTorch (if available), otherwise falls back to CPU.
    """
    if torch is None or not torch.cuda.is_available():
        return stitch_tiles_cv2(tile_dict, max_x, max_y, tile_size)

    pano_height = tile_size * max_y
    pano_width = tile_size * max_x

    # Create panorama tensor on GPU
    panorama = torch.zeros((pano_height, pano_width, 3), dtype=torch.uint8, device='cuda')

    for (x, y), tile_bytes in tile_dict.items():
        if tile_bytes is not None:
            try:
                # Decode tile on CPU using OpenCV
                arr = np.frombuffer(tile_bytes, np.uint8)
                tile_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if tile_img is not None:
                    # Move tile to GPU and copy into panorama
                    tile_tensor = torch.from_numpy(tile_img).cuda()
                    x_offset = x * tile_size
                    y_offset = y * tile_size
                    panorama[y_offset:y_offset + tile_img.shape[0],
                    x_offset:x_offset + tile_img.shape[1], :] = tile_tensor
            except Exception as e:
                print(f"Error decoding/stitching tile ({x},{y}): {e}")

    # Move panorama back to CPU and convert to NumPy
    panorama_np = panorama.cpu().numpy()
    return panorama_np

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

async def download_and_stitch_pano_cv2(lat, lon, pano_id, pano_download_data, coverage, coords, rows, cols, coverage_lock):
    """
    Handles fallback gracefully and updates coverage for both tile-based and fallback downloads.
    """
    zoom_candidates = [
        (5, 26, 13),
        (4, 13, 7),
        (3, 7, 4),
    ]

    for (zoom, max_x, max_y) in zoom_candidates:
        tile_dict = await download_tiles_async(pano_id, zoom, max_x, max_y)

        if tile_dict is None:
            # First tile + fallback failed => try next zoom or mark invalid
            continue

        # Handle fallback success
        if "full_image" in tile_dict:
            fallback_image = tile_dict["full_image"]
            pano_md5 = hashlib.md5(fallback_image).hexdigest()

            # Deduplicate and save
            if any(info.get("md5") == pano_md5 for info in pano_download_data.values()):
                print(f"[Dedup] Fallback image for panoid={pano_id} => MD5 {pano_md5} is a duplicate. Skipping.")
                return None

            filename = f"{lat:.5f}_{lon:.5f}_{pano_id}_full.jpg"
            out_path = os.path.join(STITCH_DIR, filename)
            with open(out_path, "wb") as f:
                f.write(fallback_image)

            print(f"[Fallback Saved] Panorama saved as fallback for panoid={pano_id}.")

            # Update coverage for fallback
            async with coverage_lock:
                update_coverage(coverage, coords, lat, lon, rows, cols)

            return (filename, pano_md5)

        # Otherwise stitch the tiles
        loop = asyncio.get_running_loop()

        def stitch_job():
            return stitch_tiles_gpu(tile_dict, max_x, max_y, tile_size=512)

        # Use tqdm to show progress for stitching
        with tqdm(total=max_x * max_y, desc=f"Stitching {pano_id}") as pbar:
            stitched_result = await loop.run_in_executor(stitch_executor, stitch_job)
            pbar.update(max_x * max_y)  # Update progress bar to completion

        if stitched_result is None:
            continue

        if np.count_nonzero(stitched_result) == 0:
            continue

        # Compute MD5 and deduplicate
        pano_md5 = compute_md5_for_stitched_image(stitched_result)
        if any(info.get("md5") == pano_md5 for info in pano_download_data.values()):
            print(f"[Dedup] Stitched image for panoid={pano_id} => MD5 {pano_md5} is a duplicate. Skipping.")
            return None

        # Save stitched panorama
        filename = f"{lat:.5f}_{lon:.5f}_{pano_id}.jpg"
        out_path = os.path.join(STITCH_DIR, filename)
        cv2.imwrite(out_path, stitched_result)

        print(f"[Stitched Saved] Panorama saved for panoid={pano_id}.")

        # Update coverage for stitched panorama
        async with coverage_lock:
            update_coverage(coverage, coords, lat, lon, rows, cols)

        return (filename, pano_md5)

    # If all zooms + fallback fail
    print(f"[Failure] No panorama could be downloaded for panoid={pano_id}.")
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

def calculate_grid_dimensions(lat_min, lat_max, lon_min, lon_max, total_points):
    """Calculate rows/cols based on actual geographic aspect ratio."""
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # Validation for ranges
    if lat_range <= 0 or lon_range <= 0:
        raise ValueError(f"Invalid coordinate ranges: lat_range={lat_range}, lon_range={lon_range}")

    aspect_ratio = lon_range / lat_range

    # Validate aspect ratio
    if aspect_ratio <= 0 or np.isnan(aspect_ratio) or np.isinf(aspect_ratio):
        raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")

    # Calculate base rows/cols
    base = int(np.sqrt(total_points / aspect_ratio))
    rows = int(base)
    cols = int(base * aspect_ratio)

    # Ensure we have enough points
    while rows * cols < total_points:
        cols += 1
        if rows * cols < total_points:
            rows += 1

    return rows, cols


def generate_coordinates(lat_min, lat_max, lon_min, lon_max, total_points):
    rows, cols = calculate_grid_dimensions(lat_min, lat_max, lon_min, lon_max, total_points)
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


def init_map_with_coverage(lat_min, lat_max, lon_min, lon_max, rows, cols):
    """Initialize map with correct projection and aspect ratio"""
    # Convert coordinates to Web Mercator
    x_min, x_max, y_min, y_max = transform_bbox(lat_min, lat_max, lon_min, lon_max)

    # Calculate exact aspect ratio
    width = x_max - x_min
    height = y_max - y_min
    aspect_ratio = width / height

    # Create figure with dynamic size
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set bounds and lock aspect ratio
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='datalim')

    # Add basemap after setting aspect ratio
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:3857')

    # Remove padding around the plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Generate grid lines in WGS84 and transform to Mercator
    lat_steps = np.linspace(lat_min, lat_max, rows + 1)
    lon_steps = np.linspace(lon_min, lon_max, cols + 1)

    # Draw latitude lines
    for lat in lat_steps:
        x_line = np.linspace(lon_min, lon_max, 100)
        y_line = np.full_like(x_line, lat)
        x_proj, y_proj = wgs84_to_mercator.transform(x_line, y_line)
        ax.plot(x_proj, y_proj, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Draw longitude lines
    for lon in lon_steps:
        y_line = np.linspace(lat_min, lat_max, 100)
        x_line = np.full_like(y_line, lon)
        x_proj, y_proj = wgs84_to_mercator.transform(x_line, y_line)
        ax.plot(x_proj, y_proj, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_title("Live Coverage Map with Gridlines")
    return fig, ax, np.zeros((rows, cols), dtype=int)

def update_map_coverage(ax, coverage, lat_min, lat_max, lon_min, lon_max, rows, cols, r, c):
    """
    Update the map with the current coverage grid (PROPERLY PROJECTED)
    Only updates the specific cell that changed.
    """
    # Clear previous coverage for this cell
    for coll in ax.collections:
        if coll.get_label() == f'coverage_cell_{r}_{c}':
            coll.remove()

    # Create grid steps in geographic coordinates
    lon_steps = np.linspace(lon_min, lon_max, cols + 1)
    lat_steps = np.linspace(lat_min, lat_max, rows + 1)

    # Check if the cell is covered
    if coverage[r, c] == 1:
        # Get cell bounds in WGS84
        left = lon_steps[c]
        right = lon_steps[c + 1]
        bottom = lat_steps[r]
        top = lat_steps[r + 1]

        # Transform to Mercator
        x_left, y_bottom = wgs84_to_mercator.transform(left, bottom)
        x_right, y_top = wgs84_to_mercator.transform(right, top)

        # Create polygon coordinates
        poly_coords = [
            (x_left, y_bottom),
            (x_right, y_bottom),
            (x_right, y_top),
            (x_left, y_top)
        ]

        # Add the polygon as a single collection
        from matplotlib.collections import PolyCollection
        coll = PolyCollection(
            [poly_coords],
            facecolors='red',
            alpha=0.5,
            edgecolors='none',
            label=f'coverage_cell_{r}_{c}'
        )
        ax.add_collection(coll)

    # Redraw with tight layout
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

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

            # Get row and column indices for the updated cell
            idx = coords.index((lat, lon))
            r = idx // cols
            c = idx % cols

            # Update coverage for this cell
            async with coverage_lock:
                update_coverage(coverage, coords, lat, lon, rows, cols)
                update_map_coverage(ax, coverage, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, rows, cols, r, c)

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
            result = await download_and_stitch_pano_cv2(lat, lon, pano_id, pano_download_data, coverage, coords, rows, cols, coverage_lock)
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
    await page.close()

###############################################################################
# MAIN
###############################################################################
async def main():
    # Generate coordinates and initialize coverage
    all_coords, rows, cols = generate_coordinates(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_AMT)
    subsets = np.array_split(all_coords, N_WORKERS)

    # Initialize map with correct aspect ratio
    fig, ax, coverage = init_map_with_coverage(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, rows, cols)
    plt.show(block=False)

    # Load existing logs
    pano_log_data = load_panolog(PANOLOG)
    pano_download_data = load_pano_downloadlog(PANODOWNLOAD_LOG)

    # Initialize browser windows with VIEW_TABS system
    async with async_playwright() as pw:
        pages = []
        browsers = []
        grid_columns = 4
        window_width = 350
        window_height = 350
        overlap = 100  # Overlap between windows

        # Launch worker browsers
        for w_id in range(N_WORKERS):
            VIEW_TABS = False  # Set to True to see browsers

            if VIEW_TABS:
                browser = await pw.chromium.launch(
                    headless=False,
                    args=[
                        f"--window-size={window_width},{window_height}",
                        f"--window-position={w_id % grid_columns * (window_width - overlap)},"
                        f"{w_id // grid_columns * (window_height - overlap)}"
                    ]
                )
            else:
                browser = await pw.chromium.launch(headless=True)

            page = await browser.new_page()
            await page.set_viewport_size({"width": window_width, "height": window_height})
            pages.append(page)
            browsers.append(browser)

        # Position the map window
        if VIEW_TABS:
            mgr = plt.get_current_fig_manager()
            mgr.window.setGeometry(
                grid_columns * (window_width - overlap) + 50,
                50,
                600,
                int(600)
            )

        # Start scraping tasks
        tasks = []
        coverage_lock = asyncio.Lock()
        for idx, subset in enumerate(subsets, start=1):
            t = asyncio.create_task(scraper_worker(
                worker_id=idx,
                coords_subset=subset,
                coverage=coverage,
                coverage_lock=coverage_lock,
                coords=all_coords,
                rows=rows,
                cols=cols,
                pano_log_data=pano_log_data,
                pano_download_data=pano_download_data,
                page=pages[idx - 1],
                fig=fig,
                ax=ax
            ))
            tasks.append(t)

        await asyncio.gather(*tasks)

        # Close browsers
        for browser in browsers:
            await browser.close()

    # Final update and save
    print("All Done!")
    plt.ioff()
    plt.show()
    create_final_csv_from_panolog(pano_log_data, FINAL_CSV)

if __name__ == "__main__":
    asyncio.run(main())




