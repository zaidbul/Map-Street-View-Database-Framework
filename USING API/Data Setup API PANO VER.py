'''
This is for downloading images through googles API
it will return PANO images.

This code is not entirely up-to date for efficiency.
'''

import os
import csv
import re
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import hashlib
from dotenv import load_dotenv
import io

#######################################
#            CONFIG
#######################################
load_dotenv()

API_KEY = os.getenv("API_KEY")
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
# For tile download: https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}

LAT_MIN = 38.210786
LAT_MAX = 38.223611
LON_MIN = -85.764151
LON_MAX = -85.755796

IMAGE_AMT = 4  # example: ~3 coords in each dimension => total ~9 points

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "Panoramas")
os.makedirs(IMAGE_DIR, exist_ok=True)

PANOLOG = os.path.join(SCRIPT_DIR, "PanoLog.csv")
FINAL_CSV = os.path.join(SCRIPT_DIR, "PanoMetadata.csv")

# Define possible statuses
STATUS_DOWNLOADED = "Downloaded"
STATUS_MISSING = "Missing"
STATUS_DUPLICATE = "Duplicate"
STATUS_INVALID = "Invalid"

#######################################
#           HELPER FUNCTIONS
#######################################

def generate_coordinates(lat_min, lat_max, lon_min, lon_max, total_points):
    """
    Generate a grid of lat/lon. We round to 5 decimals to keep everything consistent.
    """
    area = (lat_max - lat_min) * (lon_max - lon_min)
    step = (area / total_points) ** 0.5
    latitudes = np.arange(lat_min, lat_max, step)
    longitudes = np.arange(lon_min, lon_max, step)

    coords = []
    for lat in latitudes:
        for lon in longitudes:
            lat_rounded = round(lat, 5)
            lon_rounded = round(lon, 5)
            coords.append((lat_rounded, lon_rounded))
    return coords

def fetch_pano_metadata(lat, lon):
    """
    Calls the Street View metadata endpoint to get coverage info & the pano_id.
    Returns:
      { "pano_id": "...", "status": "OK" } on success
      or { "status": "ZERO_RESULTS" or other } if no coverage.
    """
    params = {
        "location": f"{lat},{lon}",
        "key": API_KEY,
        "source": "outdoor",  # Force outdoor only
        # 'radius': 50, etc. if you want to allow some radius
    }
    r = requests.get(METADATA_URL, params=params)
    return r.json()

def download_tile(pano_id, zoom, x, y):
    """
    Download a tile for a given pano_id, zoom level, tile indices (x, y).
    Returns raw image bytes or None.
    """
    url = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.content
    return None

def stitch_pano(pano_id, zoom, max_x, max_y):
    """
    Download all tiles for the given pano_id at a specified zoom,
    and stitch them into a single equirectangular image.
    - tile_size is typically 512 for Street View.
    - max_x, max_y indicate how many tiles horizontally and vertically at that zoom.
    Returns a PIL Image or None if download fails.
    """
    tile_size = 512
    pano_width = tile_size * max_x
    pano_height = tile_size * max_y

    panorama = Image.new('RGB', (pano_width, pano_height))

    for x in range(max_x):
        for y in range(max_y):
            tile_data = download_tile(pano_id, zoom, x, y)
            if tile_data:
                tile_img = Image.open(io.BytesIO(tile_data))
                panorama.paste(tile_img, (x * tile_size, y * tile_size))
            else:
                # If any tile is missing, you can decide how to handle
                print(f"Warning: Missing tile x={x}, y={y} for pano {pano_id}")
                # You could fill with black or skip
                # For now, let's fill with black
                pass

    return panorama

#######################################
#   PANOLOG CSV HELPERS
#######################################
def load_panolog(csv_path):
    """
    Reads a CSV of columns:
        latitude, longitude, pano_id, filename, status, hash
    Returns a dict keyed by (lat, lon):
        {
          (lat, lon): {
             'pano_id': str,
             'filename': str,
             'status': str,
             'hash': str
          },
          ...
        }
    """
    data = {}
    if not os.path.isfile(csv_path):
        return data

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            key = (lat, lon)
            data[key] = {
                "pano_id": row["pano_id"],
                "filename": row["filename"],
                "status": row["status"],
                "hash": row["hash"]
            }
    return data

def save_panolog(csv_path, data_dict):
    fieldnames = ["latitude", "longitude", "pano_id", "filename", "status", "hash"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (lat, lon), info in data_dict.items():
            writer.writerow({
                "latitude": lat,
                "longitude": lon,
                "pano_id": info["pano_id"],
                "filename": info["filename"],
                "status": info["status"],
                "hash": info["hash"]
            })

def update_panolog_entry(data_dict, lat, lon, pano_id=None, filename=None, status=None, file_hash=None):
    key = (lat, lon)
    if key not in data_dict:
        data_dict[key] = {
            "pano_id": pano_id if pano_id else "",
            "filename": filename if filename else "",
            "status": status if status else "",
            "hash": file_hash if file_hash else ""
        }
    else:
        if pano_id is not None:
            data_dict[key]["pano_id"] = pano_id
        if filename is not None:
            data_dict[key]["filename"] = filename
        if status is not None:
            data_dict[key]["status"] = status
        if file_hash is not None:
            data_dict[key]["hash"] = file_hash

#######################################
#   SYNC: DISK VS. PANOLOG
#######################################
def calculate_image_hash(image_path):
    """
    Returns an MD5 of a 64x64 version of the image for dedup detection.
    """
    with Image.open(image_path) as img:
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        return hashlib.md5(img.tobytes()).hexdigest()

def sync_files_with_panolog(images_dir, panolog_data):
    """
    1) If panolog says 'Downloaded' but file doesn't exist => mark 'Missing'.
    2) If a file exists on disk but isn't in the log or is 'Missing', compute hash, see if it's duplicate, etc.
    """
    # Build a quick map of existing downloaded hashes
    downloaded_hashes = {}
    for (lat, lon), info in panolog_data.items():
        if info["status"] == STATUS_DOWNLOADED and info["hash"]:
            downloaded_hashes[info["hash"]] = (lat, lon)

    # 1) Mark missing
    for (lat, lon), info in panolog_data.items():
        if info["status"] == STATUS_DOWNLOADED:
            fp = os.path.join(images_dir, info["filename"])
            if not os.path.isfile(fp):
                print(f"File listed but missing on disk => marking Missing: {info['filename']}")
                info["status"] = STATUS_MISSING

    # 2) Check actual files in the folder
    existing_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    # You can parse the filename if you embed lat/lon/pano_id in the name
    # For this example, let's assume we do: lat_lon_panoid.jpg
    pattern = re.compile(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_([A-Za-z0-9\-_]+)\.jpg$", re.IGNORECASE)
    for filename in existing_files:
        match = pattern.match(filename)
        if not match:
            # skip files not matching pattern
            continue
        lat_str, lon_str, pid = match.groups()
        lat_f = float(lat_str)
        lon_f = float(lon_str)
        key = (lat_f, lon_f)
        filepath = os.path.join(images_dir, filename)

        # calculate hash
        new_hash = calculate_image_hash(filepath)

        # check if this hash is known
        if new_hash in downloaded_hashes:
            # it's a duplicate
            existing_key = downloaded_hashes[new_hash]
            if key not in panolog_data or panolog_data[key]["status"] != STATUS_DOWNLOADED:
                print(f"Removing duplicate file: {filename}")
                os.remove(filepath)
                update_panolog_entry(
                    panolog_data, lat_f, lon_f,
                    pano_id=pid, filename=filename, status=STATUS_DUPLICATE, file_hash=new_hash
                )
            continue

        # else it's new or recovered
        if key not in panolog_data or panolog_data[key]["status"] in [STATUS_MISSING, STATUS_INVALID]:
            print(f"Found new or recovered pano on disk: {filename}")
            update_panolog_entry(
                panolog_data, lat_f, lon_f,
                pano_id=pid,
                filename=filename,
                status=STATUS_DOWNLOADED,
                file_hash=new_hash
            )
            downloaded_hashes[new_hash] = key
        else:
            # Possibly status=Duplicate or something, but file physically exists
            current_status = panolog_data[key]["status"]
            if current_status in [STATUS_DUPLICATE, STATUS_INVALID]:
                print(f"Recovered file for a previously {current_status} entry: {filename}")
                update_panolog_entry(
                    panolog_data, lat_f, lon_f,
                    pano_id=pid,
                    filename=filename,
                    status=STATUS_DOWNLOADED,
                    file_hash=new_hash
                )
                downloaded_hashes[new_hash] = key

#######################################
#  DOWNLOADING / STITCHING PANORAMAS
#######################################
def download_and_stitch_pano(lat, lon, pano_id):
    """
    Example that tries a certain range of zoom levels & tile counts
    to fetch a big equirectangular image.
    Adjust according to your location coverage or experimentation.
    """
    # Typically for Street View, Zoom 5 might have a ~10x5 tile grid
    # But it can vary by location. Some panos won't have zoom=5. You can try decreasing if fails.
    # For demonstration, let's assume max_x=10, max_y=5 at zoom=5.
    # Or check smaller zoom for guaranteed coverage.

    zoom_candidates = [5, 4, 3]
    for zoom in zoom_candidates:
        # guessed tile layout at each zoom (this is empirical, might differ by location):
        # Zoom 5 might have max_x=26, max_y=13 for some panos.
        # For demonstration, let's guess 10x5. Feel free to refine.
        max_x = 26
        max_y = 13
        pano_img = stitch_pano(pano_id, zoom, max_x, max_y)
        if pano_img is not None:
            # Save result
            filename = f"{lat:.5f}_{lon:.5f}_{pano_id}.jpg"
            out_path = os.path.join(IMAGE_DIR, filename)
            pano_img.save(out_path)
            return filename
    # if all fails, return None
    return None

#######################################
#  CREATE FINAL CSV
#######################################
def create_final_csv(images_dir, output_csv):
    """
    For demonstration, we scan the folder for files named lat_lon_panoid.jpg
    and write them out.
    """
    pattern = re.compile(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_([A-Za-z0-9\-_]+)\.jpg$", re.IGNORECASE)
    rows = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(".jpg"):
            m = pattern.match(filename)
            if m:
                lat_str, lon_str, pid = m.groups()
                rows.append({
                    "latitude": lat_str,
                    "longitude": lon_str,
                    "pano_id": pid,
                    "filename": filename
                })
    fieldnames = ["latitude", "longitude", "pano_id", "filename"]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nFinal CSV created: {output_csv}")
    print(f"Total entries: {len(rows)}")

#######################################
#         MAIN SCRIPT
#######################################
if __name__ == "__main__":
    print("Starting panoramic download process...")

    # 1) Load existing PanoLog
    panolog_data = load_panolog(PANOLOG)

    # 2) Sync local folder with log
    print("\n=== Initial Sync ===")
    sync_files_with_panolog(IMAGE_DIR, panolog_data)
    save_panolog(PANOLOG, panolog_data)

    # 3) Generate coordinates
    coords = generate_coordinates(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_AMT)

    # 4) For each coordinate, check if we have a valid pano
    for (lat, lon) in coords:
        entry = panolog_data.get((lat, lon))
        if entry and entry["status"] == STATUS_DOWNLOADED:
            # Already downloaded, skip
            print(f"Skipping {lat},{lon} => already downloaded: {entry['pano_id']}")
            continue
        if entry and entry["status"] in [STATUS_DUPLICATE, STATUS_INVALID]:
            # We won't try again if it's marked invalid or a known duplicate
            print(f"Skipping {lat},{lon} => status={entry['status']}")
            continue

        # Not downloaded or was missing => fetch metadata
        meta = fetch_pano_metadata(lat, lon)
        if meta.get("status") != "OK":
            print(f"No valid pano for {lat},{lon}, marking INVALID.")
            update_panolog_entry(
                panolog_data, lat, lon,
                pano_id="", filename="", status=STATUS_INVALID, file_hash=""
            )
            continue

        pano_id = meta.get("pano_id", "")
        if not pano_id:
            print(f"No pano_id in metadata for {lat},{lon}, marking INVALID.")
            update_panolog_entry(
                panolog_data, lat, lon,
                pano_id="", filename="", status=STATUS_INVALID
            )
            continue

        # If the same coordinate had a different pano_id logged, overwrite or handle as you like
        print(f"Found pano_id={pano_id} for {lat},{lon}")

        # 5) Download & stitch the full pano
        pano_filename = download_and_stitch_pano(lat, lon, pano_id)
        if not pano_filename:
            print(f"Failed to stitch panorama for {lat},{lon}, marking INVALID.")
            update_panolog_entry(
                panolog_data, lat, lon,
                pano_id=pano_id, filename="", status=STATUS_INVALID
            )
            continue

        # 6) Calculate hash
        file_path = os.path.join(IMAGE_DIR, pano_filename)
        new_hash = calculate_image_hash(file_path)

        # Check for duplicates
        duplicates = [k for (k, v) in panolog_data.items()
                      if v["hash"] == new_hash and v["status"] == STATUS_DOWNLOADED]
        if duplicates:
            # It's a duplicate
            print(f"Duplicate detected for {lat},{lon}. Removing {pano_filename}")
            os.remove(file_path)
            update_panolog_entry(
                panolog_data, lat, lon,
                pano_id=pano_id, filename=pano_filename,
                status=STATUS_DUPLICATE,
                file_hash=new_hash
            )
        else:
            print(f"Downloaded full pano for {lat},{lon} => {pano_filename}")
            update_panolog_entry(
                panolog_data, lat, lon,
                pano_id=pano_id,
                filename=pano_filename,
                status=STATUS_DOWNLOADED,
                file_hash=new_hash
            )

    # 7) Save updated log
    save_panolog(PANOLOG, panolog_data)

    # 8) Final re-sync
    print("\n=== Final Sync ===")
    sync_files_with_panolog(IMAGE_DIR, panolog_data)
    save_panolog(PANOLOG, panolog_data)

    # 9) Create final CSV
    create_final_csv(IMAGE_DIR, FINAL_CSV)

    print("\nAll steps complete!")
