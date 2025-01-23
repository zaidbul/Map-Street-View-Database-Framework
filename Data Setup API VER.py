BASE_URL = "https://maps.googleapis.com/maps/api/streetview"

LAT_MIN = 38.210786
LAT_MAX = 38.223611
LON_MIN = -85.764151
LON_MAX = -85.755796

HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]
# Example: IMAGE_AMT = 50 will be (50*16= 800 images)
IMAGE_AMT = 3

import os
import csv
import re
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import hashlib
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "Images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# CSV of (lat, lon, heading, filename) to skip re-downloading same coords
DOWNLOADLOG = os.path.join(SCRIPT_DIR, "Downloadlog.csv")

# CSV that stores (hash, filename) of unique images discovered so far
HASHLOG = os.path.join(SCRIPT_DIR, "HashLog.csv")

# Final CSV after cleanup
FINAL_CSV = os.path.join(SCRIPT_DIR, "Metadata.csv")


def load_processed_coordinates(csv_file):
    """
    Load a CSV of [latitude, longitude, heading, filename]
    into a set of (lat, lon, heading).
    """
    processed = set()
    if not os.path.exists(csv_file):
        return processed

    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            heading = float(row["heading"])
            processed.add((lat, lon, heading))
    return processed

def append_to_csv(csv_file, lat, lon, heading, filename):
    """
    Append a single (lat, lon, heading, filename) row to a CSV.
    """
    file_exists = os.path.isfile(csv_file)
    fieldnames = ["latitude", "longitude", "heading", "filename"]
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "latitude": lat,
            "longitude": lon,
            "heading": heading,
            "filename": filename
        })

def load_hashes_from_csv(csv_file):
    """
    Loads a CSV of [hash, filename] into a dict or set of existing_hashes.
    """
    existing_hashes = set()
    if not os.path.exists(csv_file):
        return existing_hashes

    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row["hash"]
            existing_hashes.add(h)
    return existing_hashes

def append_hash_to_csv(csv_file, h, filename):
    """
    Append a single (hash, filename) row to a CSV.
    """
    file_exists = os.path.isfile(csv_file)
    fieldnames = ["hash", "filename"]
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({"hash": h, "filename": filename})

def generate_coordinates(lat_min, lat_max, lon_min, lon_max, total_per_heading):
    """
    Generates a grid of coordinates to yield ~`total_per_heading` points
    for each heading. The step size is determined by the area of the bounding
    box divided by total_per_heading, then taking the square root. Be aware
    that Google Street Maps has a set granularity that I didn't bother to figure
    out, so too high of precision might yield nothing. That is why for safety there is
    a part that will remove duplicate images lol.
    """
    area = (lat_max - lat_min) * (lon_max - lon_min)
    step = (area / total_per_heading) ** 0.5
    latitudes = np.arange(lat_min, lat_max, step)
    longitudes = np.arange(lon_min, lon_max, step)
    return [(lat, lon) for lat in latitudes for lon in longitudes]

def download_street_view_image(lat, lon, heading=0, fov=90):
    """
    Download a Street View image for the given coordinates.
    Returns (lat, lon, heading, filename) or None if the request fails.
    """
    params = {
        "size": "640x640",
        "location": f"{lat},{lon}",
        "heading": heading,
        "fov": fov,
        "key": API_KEY,
        "return_error_code": "true",
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

def calculate_image_hash(image_path):
    """
    Returns an MD5 hash of a resized 64x64 version of the image.
    """
    with Image.open(image_path) as img:
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        return hashlib.md5(img.tobytes()).hexdigest()

def rename_files_to_5decimals(images_dir):
    """
    Ensures all .jpg filenames have exactly 5 decimals for lat/lon. You may change this if you want more precision.
    E.g., 38.21_-85.76_heading90.jpg => 38.21000_-85.76000_heading90.jpg
    """
    pattern = re.compile(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_heading(\d+)\.jpg$")
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
                new_filename = f"{lat:.5f}_{lon:.5f}_heading{heading}.jpg"
                if new_filename != filename:
                    old_path = os.path.join(images_dir, filename)
                    new_path = os.path.join(images_dir, new_filename)
                    print(f"Renaming {filename} => {new_filename}")
                    os.rename(old_path, new_path)

def create_final_csv(images_dir, output_csv):
    """
    Scans the directory for .jpg files named like lat_lon_headingXXX.jpg
    and writes a CSV with [latitude, longitude, heading, filename].
    """
    pattern = re.compile(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_heading(\d+)\.jpg$")
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

# ADDED: Final dedup function (just for good measure)
def deduplicate_by_hash(images_dir):
    """
    Removes duplicate images based on image content (hash).
    If found, deletes them.
    """
    print("\nChecking for Possible Duplicates")

    all_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    hash_map = {}
    duplicates = []

    for f in all_files:
        path = os.path.join(images_dir, f)
        try:
            h = calculate_image_hash(path)
            if h in hash_map:
                duplicates.append(path)
            else:
                hash_map[h] = f
        except Exception as e:
            print(f"Error hashing {f}: {e}")

    for dup in duplicates:
        print(f"Removing duplicate: {os.path.basename(dup)}")
        os.remove(dup)

    print(f"Removed {len(duplicates)} duplicates.")


# Main script

if __name__ == "__main__":

    print("\nStarting Process")
    deduplicate_by_hash(IMAGE_DIR)

    # 1) Generate coords
    coords = generate_coordinates(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, IMAGE_AMT)

    # 2) Load "processed coords" from Downloadlog so we skip the same lat/lon/heading
    processed_coords = load_processed_coordinates(DOWNLOADLOG)

    # 3) Load existing image hashes from HASHLOG
    existing_hashes = load_hashes_from_csv(HASHLOG)

    print("\n=== Step 1: Downloading Images with On-The-Fly Dedup ===")

    for (lat, lon) in coords:
        # ADDED: Flag to skip remaining headings if a duplicate is found
        skip_remaining_headings = False

        for i, heading in enumerate(HEADINGS):
            if skip_remaining_headings:
                # Mark all leftover headings as processed in the download log
                print("HASH FOUND, SKIPPING remaining headings at", lat, lon)
                for leftover_heading in HEADINGS[i:]:
                    processed_coords.add((lat, lon, leftover_heading))
                    append_to_csv(DOWNLOADLOG, lat, lon, leftover_heading, "SKIPPED_DUE_TO_HASH")
                break  # Leaves the heading loop

            # skip if (lat, lon, heading) is already processed
            if (lat, lon, heading) in processed_coords:
                print(f"Skipping coords already processed: {lat:.5f}, {lon:.5f}, heading={heading}")
                continue

            result = download_street_view_image(lat, lon, heading=heading)
            if result is None:
                print(f"No image for {lat:.5f}, {lon:.5f}, heading={heading}")
                continue

            lat_dl, lon_dl, heading_dl, filename_dl = result
            print(f"Downloaded: {filename_dl}")

            new_path = os.path.join(IMAGE_DIR, filename_dl)
            try:
                new_hash = calculate_image_hash(new_path)
            except Exception as e:
                print(f"Error hashing {filename_dl}: {e}")
                continue

            if new_hash in existing_hashes:
                # DUPLICATE
                print(f"Duplicate detected (hash match). Removing {filename_dl}")
                os.remove(new_path)

                # Mark coords as processed
                processed_coords.add((lat_dl, lon_dl, heading_dl))
                append_to_csv(DOWNLOADLOG, lat_dl, lon_dl, heading_dl, filename_dl)

                skip_remaining_headings = True

            else:
                # UNIQUE
                existing_hashes.add(new_hash)
                append_hash_to_csv(HASHLOG, new_hash, filename_dl)

                processed_coords.add((lat_dl, lon_dl, heading_dl))
                append_to_csv(DOWNLOADLOG, lat_dl, lon_dl, heading_dl, filename_dl)

    print("\n=== Step 2: Renaming Files to 5 Decimals ===")
    rename_files_to_5decimals(IMAGE_DIR)

    print("\n=== Step 3: Deleting Duplicates ===")
    deduplicate_by_hash(IMAGE_DIR)

    print("\n=== Step 4: Creating Final CSV ===")
    create_final_csv(IMAGE_DIR, FINAL_CSV)


    print("\nAll steps complete!")