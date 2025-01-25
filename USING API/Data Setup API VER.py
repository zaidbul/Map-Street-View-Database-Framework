import os
import csv
import re
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import hashlib
from dotenv import load_dotenv

#######################################
#            CONFIGURATIONS
#######################################
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = "https://maps.googleapis.com/maps/api/streetview"

LAT_MIN = 38.210786
LAT_MAX = 38.223611
LON_MIN = -85.764151
LON_MAX = -85.755796


HEADINGS = [0, 90, 180, 270]
IMAGE_AMT = 3  # Example: 3 => ~3 * 8 = 24 total images (requests)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "Images")
os.makedirs(IMAGE_DIR, exist_ok=True)

DOWNLOADLOG = os.path.join(SCRIPT_DIR, "DownloadLog.csv")
FINAL_CSV = os.path.join(SCRIPT_DIR, "Metadata.csv")

# Define possible statuses
STATUS_DOWNLOADED = "Downloaded"
STATUS_MISSING = "Missing"
STATUS_DUPLICATE = "Duplicate"
STATUS_INVALID = "Invalid"
STATUS_SKIPPED = "Skipped"

#######################################
#           HELPER FUNCTIONS
#######################################

def calculate_image_hash(image_path):
    """
    Returns an MD5 hash of a resized 64x64 version of the image
    for content-based deduplication.
    """
    with Image.open(image_path) as img:
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        return hashlib.md5(img.tobytes()).hexdigest()

def parse_filename_for_coords(filename):
    """
    Given a filename like "38.21000_-85.76000_heading45.jpg",
    return (38.21000, -85.76000, 45).
    Return None if it doesn't match.
    """
    pattern = re.compile(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_heading(\d+)\.jpg$", re.IGNORECASE)
    match = pattern.match(filename)
    if not match:
        return None
    lat_str, lon_str, heading_str = match.groups()
    return float(lat_str), float(lon_str), float(heading_str)


def generate_coordinates(lat_min, lat_max, lon_min, lon_max, total_per_heading):
    """
    Generate a grid of coordinates to yield ~`total_per_heading` points
    for each heading. The step size is determined by bounding-box area
    divided by total_per_heading, then square root. It's approximate.
    """
    area = (lat_max - lat_min) * (lon_max - lon_min)
    step = (area / total_per_heading) ** 0.5
    latitudes = np.arange(lat_min, lat_max, step)
    longitudes = np.arange(lon_min, lon_max, step)

    coords = []
    for lat in latitudes:
        for lon in longitudes:
            # Round them right away to 5 decimals
            lat_rounded = round(lat, 5)
            lon_rounded = round(lon, 5)
            coords.append((lat_rounded, lon_rounded))
    return coords


#######################################
#       DOWNLOADLOG CSV HELPERS
#######################################

def load_downloadlog(log_path):
    """
    Reads DownloadLog.csv into a dict keyed by (lat, lon, heading).
    Each value is another dict with keys:
      {
        'latitude': float,
        'longitude': float,
        'heading': float,
        'filename': str,
        'status': str,
        'hash': str
      }
    """
    data = {}
    if not os.path.isfile(log_path):
        return data  # Return empty dict if no file yet

    with open(log_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            heading = float(row["heading"])
            key = (lat, lon, heading)
            data[key] = {
                "latitude": lat,
                "longitude": lon,
                "heading": heading,
                "filename": row.get("filename", ""),
                "status": row.get("status", ""),
                "hash": row.get("hash", ""),
            }
    return data

def save_downloadlog(log_path, data_dict):
    """
    Writes the log data back to CSV. `data_dict` is a dict of
    (lat, lon, heading) => { 'latitude':..., 'longitude':..., 'heading':..., 'filename':..., 'status':..., 'hash':... }
    """
    fieldnames = ["latitude", "longitude", "heading", "filename", "status", "hash"]
    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, row in data_dict.items():
            writer.writerow({
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "heading": row["heading"],
                "filename": row["filename"],
                "status": row["status"],
                "hash": row["hash"]
            })

def update_downloadlog_entry(data_dict, lat, lon, heading,
                             filename=None, status=None, file_hash=None):
    """
    Updates or creates an entry in the downloadlog dictionary.
    """
    key = (lat, lon, heading)
    if key not in data_dict:
        data_dict[key] = {
            "latitude": lat,
            "longitude": lon,
            "heading": heading,
            "filename": filename if filename else "",
            "status": status if status else "",
            "hash": file_hash if file_hash else ""
        }
    else:
        if filename is not None:
            data_dict[key]["filename"] = filename
        if status is not None:
            data_dict[key]["status"] = status
        if file_hash is not None:
            data_dict[key]["hash"] = file_hash

def sync_files_with_downloadlog(images_dir, log_data):
    """
    1) For each row in log_data with status="Downloaded", check if the file exists.
       If not, mark status="Missing".
    2) For each .jpg in images_dir, see if there's a corresponding row. If not, or if that row is "Missing",
       compute its hash, check for duplicates, then update or create row accordingly.
    """

    # 1) Mark missing downloads
    for (lat, lon, heading), row in log_data.items():
        if row["status"] == STATUS_DOWNLOADED:
            # Check if file exists
            filepath = os.path.join(images_dir, row["filename"])
            if not os.path.isfile(filepath):
                # file is missing
                print(f"File listed as downloaded but missing on disk: {row['filename']}")
                row["status"] = STATUS_MISSING

    # Build a quick map of existing (hash -> (lat,lon,heading)) for any row that is "Downloaded"
    downloaded_hashes = {}
    for (lat, lon, heading), row in log_data.items():
        if row["status"] == STATUS_DOWNLOADED and row["hash"]:
            downloaded_hashes[row["hash"]] = (lat, lon, heading)

    # 2) For each .jpg in the folder, parse lat/lon/heading
    existing_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    for filename in existing_files:
        coords = parse_filename_for_coords(filename)
        if not coords:
            # If it doesn't match the naming pattern, skip or handle as you wish
            # For simplicity, skip it
            continue

        lat, lon, heading = coords
        key = (lat, lon, heading)
        filepath = os.path.join(images_dir, filename)
        file_hash = calculate_image_hash(filepath)

        # Check if this hash is already in the log as "Downloaded" somewhere
        if file_hash in downloaded_hashes:
            # It's a duplicate of a known downloaded file
            # Remove it or handle as you like:
            if key not in log_data or log_data[key]["status"] != STATUS_DOWNLOADED:
                print(f"Removing manually added duplicate: {filename}")
                os.remove(filepath)
                # Mark in log as Duplicate if there's not an existing row
                update_downloadlog_entry(
                    log_data, lat, lon, heading,
                    filename=filename,
                    status=STATUS_DUPLICATE,
                    file_hash=file_hash
                )
            continue

        # If we reach here, it's not a known hash or it's new content
        # If key not in log or the log says 'Missing', we can fix it
        if key not in log_data or log_data[key]["status"] == STATUS_MISSING:
            print(f"Found new or recovered file on disk: {filename}")
            update_downloadlog_entry(
                log_data, lat, lon, heading,
                filename=filename,
                status=STATUS_DOWNLOADED,
                file_hash=file_hash
            )
            downloaded_hashes[file_hash] = key
        else:
            # log_data has a row but possibly different status. Let's see:
            existing_status = log_data[key]["status"]
            # If it was "Duplicate", "Invalid", or "Skipped", yet file physically exists now,
            # we could consider it "Downloaded." This depends on your preference:
            if existing_status in [STATUS_DUPLICATE, STATUS_INVALID, STATUS_SKIPPED]:
                print(f"Coordinate was {existing_status}, but file re-appeared: {filename}")
                update_downloadlog_entry(
                    log_data, lat, lon, heading,
                    filename=filename,
                    status=STATUS_DOWNLOADED,
                    file_hash=file_hash
                )
                downloaded_hashes[file_hash] = key
            # else if it's already "Downloaded", do nothing.

#######################################
#       CORE DOWNLOAD LOGIC
#######################################

def download_street_view_image(lat, lon, heading=0, fov=90):
    """
    Download a Street View image for the given coordinates.
    Returns (lat, lon, heading, filename) or None if the request fails.
    """
    params = {
        "size": "640x640",  # Resolution of the image
        "location": f"{lat},{lon}",  # Latitude and longitude
        "heading": heading,  # Camera heading
        "fov": fov,  # Field of view
        "key": API_KEY,  # Your API key
        "return_error_code": "true",  # Explicitly return error codes
        "source": "outdoor",  # Restrict imagery to outdoor only
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"Request for ({lat:.5f}, {lon:.5f}, heading={heading}) failed: {response.status_code}")
        return None

    try:
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error opening image for ({lat}, {lon}): {e}")
        return None

    filename = f"{lat:.5f}_{lon:.5f}_heading{heading}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    img.save(filepath)

    return (lat, lon, heading, filename)


    # If it’s an "empty" Street View (grey image) or “no coverage” tile, you may detect it here:
    # For simplicity, let's assume status_code=200 => valid coverage

    lat_rounded = round(lat, 5)
    lon_rounded = round(lon, 5)

    filename = f"{lat_rounded:.5f}_{lon_rounded:.5f}_heading{heading}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    img.save(filepath)
    return (lat, lon, heading, filename)

#######################################
#    BUILD FINAL METADATA CSV
#######################################

def create_final_csv(images_dir, output_csv):
    """
    Scans the directory for .jpg files named like lat_lon_headingXXX.jpg
    and writes a CSV with [latitude, longitude, heading, filename].
    """
    pattern = re.compile(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_heading(\d+)\.jpg$", re.IGNORECASE)
    rows = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(".jpg"):
            match = pattern.match(filename)
            if match:
                lat_str, lon_str, heading_str = match.groups()
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                    heading = int(heading_str)
                except ValueError:
                    continue
                rows.append({
                    "latitude": lat,
                    "longitude": lon,
                    "heading": heading,
                    "filename": filename
                })
    # Write final CSV
    fieldnames = ["latitude", "longitude", "heading", "filename"]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nFinal CSV created: {output_csv}")
    print(f"Total entries: {len(rows)}")

#######################################
#   MAIN SCRIPT / EXECUTION FLOW
#######################################

if __name__ == "__main__":

    print("\nStarting Process...")

    # 1) Load the existing DownloadLog into memory
    downloadlog_data = load_downloadlog(DOWNLOADLOG)

    # 2) Sync what's on disk with the DownloadLog
    print("\n=== Syncing DownloadLog with local Images folder ===")
    sync_files_with_downloadlog(IMAGE_DIR, downloadlog_data)
    save_downloadlog(DOWNLOADLOG, downloadlog_data)

    # 3) Generate coordinate grid
    coords = generate_coordinates(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_AMT)

    # 4) Attempt (re)downloads for coords that are either not in the log or have status="Missing"
    print("\n=== Downloading New/Missing Images ===")

    for (lat, lon) in coords:
        for heading in HEADINGS:
            key = (lat, lon, heading)
            row = downloadlog_data.get(key)

            if row:
                # If there's a row, skip if it's "Downloaded", "Duplicate", "Invalid", or "Skipped"
                if row["status"] in [STATUS_DOWNLOADED, STATUS_DUPLICATE, STATUS_INVALID, STATUS_SKIPPED]:
                    # No need to re-download
                    continue
                # If row["status"] == "Missing", we do want to attempt a new download
            else:
                # If no row, that means it's brand new => we attempt a download
                pass

            result = download_street_view_image(lat, lon, heading)
            if not result:
                # Mark invalid coverage
                print(f"No image for {lat:.5f}, {lon:.5f}, heading={heading}. Marking INVALID.")
                update_downloadlog_entry(
                    downloadlog_data, lat, lon, heading,
                    filename="NoFile", status=STATUS_INVALID, file_hash=""
                )
                continue

            # If we downloaded successfully:
            lat_dl, lon_dl, heading_dl, filename_dl = result
            filepath_dl = os.path.join(IMAGE_DIR, filename_dl)
            new_hash = calculate_image_hash(filepath_dl)

            # Check if this hash already exists as "Downloaded"
            already_downloaded = [
                k for (k, v) in downloadlog_data.items()
                if v["hash"] == new_hash and v["status"] == STATUS_DOWNLOADED
            ]
            if already_downloaded:
                # It's a duplicate
                print(f"Duplicate detected (hash match). Removing {filename_dl}")
                os.remove(filepath_dl)

                update_downloadlog_entry(
                    downloadlog_data,
                    lat_dl, lon_dl, heading_dl,
                    filename=filename_dl,
                    status=STATUS_DUPLICATE,
                    file_hash=new_hash
                )
            else:
                # It's unique => mark as downloaded
                print(f"Downloaded: {filename_dl}")
                update_downloadlog_entry(
                    downloadlog_data,
                    lat_dl, lon_dl, heading_dl,
                    filename=filename_dl,
                    status=STATUS_DOWNLOADED,
                    file_hash=new_hash
                )

    # 5) Save the updated DownloadLog after downloads
    save_downloadlog(DOWNLOADLOG, downloadlog_data)

    # 7) Re-sync after rename (filenames changed!)
    print("\n=== Re-sync after rename ===")
    # Reload log in case we changed file names
    downloadlog_data = load_downloadlog(DOWNLOADLOG)
    sync_files_with_downloadlog(IMAGE_DIR, downloadlog_data)
    save_downloadlog(DOWNLOADLOG, downloadlog_data)

    # 9) Create final CSV by scanning the Images folder
    print("\n=== Creating Final Metadata CSV ===")
    create_final_csv(IMAGE_DIR, FINAL_CSV)

    print("\nAll steps complete!")
