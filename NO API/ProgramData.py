import sys
import asyncio
import csv
import re
import os
import aiohttp
import hashlib
import torch
import numpy as np
import cv2
import contextily as ctx
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QTextEdit, QTreeView, QFileSystemModel, QLineEdit, QSpinBox, QPushButton,
    QLabel, QFormLayout, QGroupBox, QListWidget, QListWidgetItem, QSlider
)
from PyQt5.QtCore import Qt, QDir, QThread, pyqtSignal, QObject, QMutex, QSize
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pyproj import Transformer
from playwright.async_api import async_playwright

###############################################################################
# CONFIG / CONSTANTS
###############################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default bounding box (can be overridden via GUI)
LAT_MIN_DEFAULT = 38.210786
LAT_MAX_DEFAULT = 38.223611
LON_MIN_DEFAULT = -85.764151
LON_MAX_DEFAULT = -85.755796

IMAGE_AMT_DEFAULT = 100  # Adjust this number as needed
STEP_SIZE_DEFAULT = 0.001  # degrees
N_WORKERS_DEFAULT = 8
TIMEOUT = 5000  # milliseconds
RETRIES = 3

# Log files
PANOLOG = os.path.join(SCRIPT_DIR, "PanoLog.csv")
PANODOWNLOAD_LOG = os.path.join(SCRIPT_DIR, "PanoDownload.csv")
FINAL_CSV = os.path.join(SCRIPT_DIR, "FinalMetadata.csv")

# Directory to store stitched panoramas
STITCH_DIR = os.path.join(SCRIPT_DIR, "StitchedPanos")
os.makedirs(STITCH_DIR, exist_ok=True)

# ThreadPool for CPU-bound stitching
stitch_executor = ThreadPoolExecutor(max_workers=4)

# Coordinate transformers
wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def transform_bbox(lat_min, lat_max, lon_min, lon_max):
    """Convert WGS84 coordinates to Web Mercator bounding box."""
    x_min, y_min = wgs84_to_mercator.transform(lon_min, lat_min)
    x_max, y_max = wgs84_to_mercator.transform(lon_max, lat_max)
    return x_min, x_max, y_min, y_max

def get_pano_id_from_url(url):
    """
    Extract the pano_id from a Google Street View URL.
    """
    match = re.search(r'!1s([^!]+)!2', url)
    return match.group(1) if match else None

def calculate_grid_dimensions(lat_min, lat_max, lon_min, lon_max, total_points):
    """Calculate rows/cols based on actual geographic aspect ratio."""
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    aspect_ratio = lon_range / lat_range

    # Start with square-like grid
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
    """Generates (lat, lon) pairs across a bounding box."""
    rows, cols = calculate_grid_dimensions(lat_min, lat_max, lon_min, lon_max, total_points)
    lat_steps = np.linspace(lat_min, lat_max, rows)
    lon_steps = np.linspace(lon_min, lon_max, cols)

    coords = []
    for lat in lat_steps:
        for lon in lon_steps:
            coords.append((round(lat, 5), round(lon, 5)))
    return coords, rows, cols


def compute_md5_for_stitched_image(stitched_array):
    """
    Compute MD5 hash for a stitched panorama image.
    """
    raw_bytes = stitched_array.tobytes()
    md5_hash = hashlib.md5(raw_bytes).hexdigest()
    return md5_hash

###############################################################################
# CSV LOGS HANDLING
###############################################################################
def load_panolog(csv_path):
    """
    Load PanoLog CSV.
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
    """Save PanoLog CSV."""
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
    Load PanoDownload CSV.
    Returns: { (lat, lon): {'pano_id':'', 'filename':'', 'status':'', 'md5':''} }
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
                    "md5": row.get("md5", "")
                }
    return data

def save_pano_downloadlog(csv_path, data_dict):
    """Save PanoDownload CSV."""
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

def create_final_csv_from_panolog(panolog_data, output_csv):
    """
    Create a final CSV from PanoLog data.
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
# TILE-DOWNLOAD AND STITCHING
###############################################################################
class TileDownloader:
    def __init__(self, semaphore, fallback_attempted):
        self.semaphore = semaphore
        self.fallback_attempted = fallback_attempted

    async def fetch_full_image(self, session, panoid):
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

    async def fetch_tile(self, session, cbk0_url, panoid, fallback=False):
        """
        Fetches a tile from cbk0 or falls back to googleusercontent.com for the full image if needed.
        """
        async with self.semaphore:
            for attempt in range(RETRIES):
                try:
                    async with session.get(cbk0_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            return await response.read()
                        elif response.status == 400 and not fallback and panoid not in self.fallback_attempted:
                            # Fall back to googleusercontent.com for the full image
                            self.fallback_attempted.add(panoid)
                            print(f"[Fallback] cbk0 failed with 400. Switching to googleusercontent for panoid {panoid}.")
                            full_image = await self.fetch_full_image(session, panoid)
                            if full_image:
                                return full_image  # Use the full image data
                except Exception as e:
                    print(f"Error fetching tile {cbk0_url} on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2)
            print(f"Failed to fetch tile {cbk0_url} after {RETRIES} attempts.")
            return None

    async def download_tiles_async(self, panoid, zoom, max_x, max_y):
        """
        Downloads tiles for cbk0, triggering fallback if any tile fails.
        If fallback succeeds, returns {"full_image": data}.
        If tile-based approach succeeds, returns dict {(x,y): tile_bytes}.
        """
        tile_url_template = (
            "https://cbk0.google.com/cbk?output=tile&panoid="
            f"{panoid}&zoom={zoom}&x={{x}}&y={{y}}"
        )

        async with aiohttp.ClientSession() as session:

            # 1) Fetch the first tile to check if it fails
            first_tile_url = tile_url_template.format(x=0, y=0)
            print(f"[Tile Check] Trying first tile for panoid={panoid} at zoom={zoom}: {first_tile_url}")
            first_tile_data = await self.fetch_tile(session, first_tile_url, panoid)

            if first_tile_data is None:
                # Trigger fallback immediately
                print(f"[Zoom {zoom}] First tile failed for panoid={panoid}. Trying fallback.")
                fallback_image = await self.fetch_full_image(session, panoid)
                if fallback_image:
                    print(f"[Fallback Success] Full image fetched for panoid={panoid}.")
                    return {"full_image": fallback_image}
                else:
                    print(f"[Zoom {zoom}] Fallback also failed for panoid={panoid}.")
                    return None

            # 2) If the first tile succeeded => start downloading the rest
            tile_dict = {(0, 0): first_tile_data}
            for x in range(max_x):
                for y in range(max_y):
                    # Skip (0,0) => already fetched
                    if x == 0 and y == 0:
                        continue

                    tile_url = tile_url_template.format(x=x, y=y)
                    tile_data = await self.fetch_tile(session, tile_url, panoid)

                    # Trigger fallback if any tile fails
                    if tile_data is None:
                        print(f"[Zoom {zoom}] Tile ({x}, {y}) failed for panoid={panoid}. Triggering fallback.")
                        fallback_image = await self.fetch_full_image(session, panoid)
                        if fallback_image:
                            print(f"[Fallback Success] Full image fetched for panoid={panoid} after tile failure.")
                            return {"full_image": fallback_image}
                        else:
                            print(f"[Zoom {zoom}] Fallback also failed for panoid={panoid}.")
                            return None

                    # Add tile to the dictionary
                    tile_dict[(x, y)] = tile_data

        return tile_dict

###############################################################################
# STITCHING FUNCTIONS
###############################################################################
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
                    panorama[y_offset:y_offset + tile_img.shape[0],
                             x_offset:x_offset + tile_img.shape[1], :] = tile_img
            except Exception as e:
                print(f"Error decoding/stitching tile ({x},{y}): {e}")

    return panorama

def stitch_tiles_gpu(tile_dict, max_x, max_y, tile_size=512):
    """
    Attempts to stitch tiles on GPU using PyTorch. If no GPU, fallback to stitch_tiles_cv2.
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

###############################################################################
# SCRAPER WORKER
###############################################################################
class MapCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.coverage = None
        self.rows = 0
        self.cols = 0
        self.coords = []
        self.lat_min = self.lat_max = self.lon_min = self.lon_max = 0.0

    def init_map(self, lat_min, lat_max, lon_min, lon_max, rows, cols, coords):
        """Initialize map with correct projection and no UI elements."""
        self.rows = rows
        self.cols = cols
        self.coords = coords
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        # Convert bounds to Web Mercator
        x_min, x_max, y_min, y_max = transform_bbox(lat_min, lat_max, lon_min, lon_max)

        # Configure axes
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('auto')  # Allow aspect ratio to adjust dynamically

        # Add basemap
        ctx.add_basemap(self.ax, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:3857')

        # Hide all UI elements
        self.ax.axis('off')

        # Remove padding
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Generate latitude and longitude grid lines
        lat_steps = np.linspace(lat_min, lat_max, rows + 1)
        lon_steps = np.linspace(lon_min, lon_max, cols + 1)

        # Plot latitude grid lines
        for lat in lat_steps:
            x_line = np.linspace(lon_min, lon_max, 100)
            y_line = np.full_like(x_line, lat)
            x_proj, y_proj = wgs84_to_mercator.transform(x_line, y_line)
            self.ax.plot(x_proj, y_proj, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=3)

        # Plot longitude grid lines
        for lon in lon_steps:
            y_line = np.linspace(lat_min, lat_max, 100)
            x_line = np.full_like(y_line, lon)
            x_proj, y_proj = wgs84_to_mercator.transform(x_line, y_line)
            self.ax.plot(x_proj, y_proj, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=3)

        # Initialize coverage grid
        self.coverage = np.zeros((rows, cols), dtype=int)

        self.draw()

    def update_coverage(self, lat, lon):
        """Update coverage based on WGS84 grid, then project to Mercator."""
        if not self.coords or self.coverage is None:
            return

        try:
            idx = self.coords.index((lat, lon))
            row = idx // self.cols
            col = idx % self.cols
        except ValueError:
            return

        if self.coverage[row, col] == 1:
            return  # Already covered

        self.coverage[row, col] = 1

        # Calculate cell bounds in WGS84
        lat_step = (self.lat_max - self.lat_min) / self.rows
        lon_step = (self.lon_max - self.lon_min) / self.cols

        cell_lat_min = self.lat_min + row * lat_step
        cell_lat_max = cell_lat_min + lat_step
        cell_lon_min = self.lon_min + col * lon_step
        cell_lon_max = cell_lon_min + lon_step

        # Convert bounds to Mercator
        x_min, y_min = wgs84_to_mercator.transform(cell_lon_min, cell_lat_min)
        x_max, y_max = wgs84_to_mercator.transform(cell_lon_max, cell_lat_max)

        # Draw the cell
        rect = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            facecolor='red', alpha=0.5, edgecolor='black'
        )
        self.ax.add_patch(rect)
        self.draw_idle()

###############################################################################
# MAIN WINDOW CLASS
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker_thread = None
        self.worker = None

    def init_ui(self):
        self.setWindowTitle('Street View Scraper')
        self.setGeometry(100, 100, 1400, 800)

        # Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)

        # Left Panel (Map Visualization)
        self.map_canvas = MapCanvas(self)
        self.main_layout.addWidget(self.map_canvas, 50)

        # Right Panel (Splitter)
        self.right_splitter = QSplitter(Qt.Vertical)

        # Log Display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.right_splitter.addWidget(self.log_display)

        # File Browser
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(QDir.currentPath())
        self.file_view = QTreeView()
        self.file_view.setModel(self.file_model)
        # Set the root to STITCH_DIR
        stitched_index = self.file_model.index(STITCH_DIR)
        if stitched_index.isValid():
            self.file_view.setRootIndex(stitched_index)
        self.right_splitter.addWidget(self.file_view)

        # Add right splitter to the main layout
        self.main_layout.addWidget(self.right_splitter, 50)

        # Control Panel
        control_group = QGroupBox("Parameters")
        form_layout = QFormLayout()

        # Coordinate Inputs
        self.lat_min_edit = QLineEdit(str(LAT_MIN_DEFAULT))
        self.lat_max_edit = QLineEdit(str(LAT_MAX_DEFAULT))
        self.lon_min_edit = QLineEdit(str(LON_MIN_DEFAULT))
        self.lon_max_edit = QLineEdit(str(LON_MAX_DEFAULT))

        # Numeric Inputs
        self.image_amt_spin = QSpinBox()
        self.image_amt_spin.setRange(2, 99999)
        self.image_amt_spin.setValue(IMAGE_AMT_DEFAULT)

        self.n_workers_spin = QSpinBox()
        self.n_workers_spin.setRange(1, 16)
        self.n_workers_spin.setValue(N_WORKERS_DEFAULT)

        # Buttons
        self.start_btn = QPushButton("Start Scraping")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        # Form layout
        form_layout.addRow(QLabel("Min Latitude:"), self.lat_min_edit)
        form_layout.addRow(QLabel("Max Latitude:"), self.lat_max_edit)
        form_layout.addRow(QLabel("Min Longitude:"), self.lon_min_edit)
        form_layout.addRow(QLabel("Max Longitude:"), self.lon_max_edit)
        form_layout.addRow(QLabel("Sample Density:"), self.image_amt_spin)
        form_layout.addRow(QLabel("Parallel Workers:"), self.n_workers_spin)
        form_layout.addRow(self.start_btn, self.stop_btn)

        control_group.setLayout(form_layout)
        self.main_layout.addWidget(control_group)

        # Connect signals
        self.start_btn.clicked.connect(self.start_scraping)
        self.stop_btn.clicked.connect(self.stop_scraping)

    def start_scraping(self):
        # Validate inputs
        try:
            params = {
                'LAT_MIN': float(self.lat_min_edit.text()),
                'LAT_MAX': float(self.lat_max_edit.text()),
                'LON_MIN': float(self.lon_min_edit.text()),
                'LON_MAX': float(self.lon_max_edit.text()),
                'IMAGE_AMT': self.image_amt_spin.value(),
                'N_WORKERS': self.n_workers_spin.value()
            }
        except ValueError:
            self.log_display.append("Invalid input for parameters. Please check and try again.")
            return

        # Disable start button and enable stop button
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Generate grid dimensions based on image_amt and aspect ratio
        coords, rows, cols = generate_coordinates(
            params['LAT_MIN'], params['LAT_MAX'],
            params['LON_MIN'], params['LON_MAX'],
            params['IMAGE_AMT']
        )

        # Initialize Map with calculated rows, cols, and coords
        self.map_canvas.init_map(
            params['LAT_MIN'],
            params['LAT_MAX'],
            params['LON_MIN'],
            params['LON_MAX'],
            rows=rows,
            cols=cols,
            coords=coords  # Pass coords here
        )

        # Adjust the side panel size based on the grid's aspect ratio
        self.adjust_side_panel_size(rows, cols)

        # Initialize worker and thread
        self.worker_thread = QThread()
        self.worker = StreetViewWorker(params)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.update_log.connect(self.log_display.append)
        self.worker.update_coverage.connect(self.map_canvas.update_coverage)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.on_finished)
        self.worker_thread.started.connect(self.worker.run)

        # Start the thread
        self.worker_thread.start()

        self.log_display.append("Scraping started...")

    def adjust_side_panel_size(self, rows, cols):
        """
        Adjust the side panel's size based on the grid's aspect ratio.
        """
        aspect_ratio = cols / rows

        # Calculate the maximum width for the side panel (half of the window width)
        max_side_panel_width = self.width() // 2

        # Calculate the desired width based on the aspect ratio
        desired_side_panel_width = int(self.width() * 0.3)  # Default to 30% of window width
        if aspect_ratio > 1:
            # If the grid is wider than tall, reduce the side panel width
            desired_side_panel_width = min(desired_side_panel_width, max_side_panel_width)

        # Set the size of the right splitter
        self.main_layout.setStretch(0, 50)  # Map canvas stretch factor
        self.main_layout.setStretch(1, 50)  # Side panel stretch factor

    def stop_scraping(self):
        if self.worker:
            self.worker.running = False
            self.log_display.append("Stopping scraping process...")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

    def on_finished(self):
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_display.append("Scraping process has finished.")

###############################################################################
# STREETS VIEW WORKER RUN METHOD FIX
###############################################################################
class StreetViewWorker(QObject):
    update_log = pyqtSignal(str)
    update_coverage = pyqtSignal(float, float)  # (lat, lon)
    finished = pyqtSignal()

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.running = True
        self.mutex = QMutex()

        # Initialize TileDownloader
        self.tile_downloader = TileDownloader(asyncio.Semaphore(10), set())

    def run(self):
        """Synchronous method to run the async run_async coroutine."""
        asyncio.run(self.run_async())

    async def run_async(self):
        try:
            await self.main_scraping()
        except Exception as e:
            self.update_log.emit(f"Critical error: {str(e)}")
        finally:
            self.finished.emit()

    async def main_scraping(self):
        params = self.params
        lat_min = params['LAT_MIN']
        lat_max = params['LAT_MAX']
        lon_min = params['LON_MIN']
        lon_max = params['LON_MAX']
        image_amt = params['IMAGE_AMT']
        n_workers = params['N_WORKERS']

        # Generate coordinate grid
        coords, rows, cols = generate_coordinates(lat_min, lat_max, lon_min, lon_max, image_amt)
        total_points = len(coords)
        coverage = np.zeros((rows, cols), dtype=int)
        coverage_lock = asyncio.Lock()

        # Load existing data
        pano_log_data = load_panolog(PANOLOG)
        pano_download_data = load_pano_downloadlog(PANODOWNLOAD_LOG)

        # Initialize Playwright
        async with async_playwright() as pw:
            browsers = []
            pages = []
            for _ in range(n_workers):
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                pages.append(page)
                browsers.append(browser)

            # Split coordinates among workers
            subsets = np.array_split(coords, n_workers)

            # Start scraping tasks
            tasks = []
            for idx, subset in enumerate(subsets, start=1):
                task = asyncio.create_task(
                    self.scraper_worker(
                        worker_id=idx,
                        coords_subset=subset,
                        coverage=coverage,
                        coverage_lock=coverage_lock,
                        coords=coords,
                        rows=rows,
                        cols=cols,
                        pano_log_data=pano_log_data,
                        pano_download_data=pano_download_data,
                        page=pages[idx - 1]
                    )
                )
                tasks.append(task)

            # Wait for all tasks to complete
            await asyncio.gather(*tasks)

            # Close the browsers
            for browser in browsers:
                await browser.close()

        # Final coverage update and CSV generation
        create_final_csv_from_panolog(pano_log_data, FINAL_CSV)
        self.update_log.emit("Scraping completed successfully!\n")

    async def scraper_worker(
            self, worker_id, coords_subset, coverage, coverage_lock,
            coords, rows, cols, pano_log_data, pano_download_data, page):
        """
        Each worker:
          1) Checks if coordinate is already valid => skip
          2) If not, scrapes for a pano_id.
          3) If valid => update coverage + PanoLog => immediately write to CSV
             => check PanoDownload => if file missing => download_and_stitch => update PanoDownload => immediately write to CSV
        """
        for lat, lon in coords_subset:
            if not self.running:
                self.update_log.emit("Scraping stopped by user.")
                return

            try:
                # Check PanoLog => skip if status=Valid and downloaded
                existing_log = pano_log_data.get((lat, lon), {})
                if existing_log.get("status") == "Valid":
                    dl_info = pano_download_data.get((lat, lon), {})
                    if dl_info.get("status") == "Downloaded":
                        continue

                # Scrape
                pano_id = await self.get_pano_id(page, lat, lon)
                if not pano_id:
                    pano_log_data[(lat, lon)] = {"pano_id": "", "status": "Invalid"}
                    save_panolog(PANOLOG, pano_log_data)
                    continue

                # Update PanoLog
                pano_log_data[(lat, lon)] = {"pano_id": pano_id, "status": "Valid"}
                save_panolog(PANOLOG, pano_log_data)

                # Download and stitch
                result = await self.process_panorama(lat, lon, pano_id, pano_download_data)
                if result:
                    filename, pano_md5 = result
                    pano_download_data[(lat, lon)] = {
                        "pano_id": pano_id,
                        "filename": filename,
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

                # Update coverage
                async with coverage_lock:
                    idx = coords.index((lat, lon))
                    r = idx // cols
                    c = idx % cols
                    coverage[r, c] = 1
                    # Emit coverage update signal
                    self.update_coverage.emit(lat, lon)

            except Exception as e:
                self.update_log.emit(f"Worker {worker_id} error at ({lat}, {lon}): {str(e)}")

        self.update_log.emit(f"Worker {worker_id} completed scraping.")

    async def get_pano_id(self, page, lat, lon):
        """
        Navigate to the Street View URL and extract the pano_id.
        """
        url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="networkidle")
            await page.wait_for_timeout(3000)  # Wait for 3 seconds
            final_url = page.url
            pano_id = get_pano_id_from_url(final_url)
            if pano_id:
                self.update_log.emit(f"Found pano_id {pano_id} at ({lat}, {lon})")
            else:
                self.update_log.emit(f"No pano_id found at ({lat}, {lon})")
            return pano_id
        except Exception as e:
            self.update_log.emit(f"Error navigating to {url}: {str(e)}")
            return None

    async def process_panorama(self, lat, lon, pano_id, pano_download_data):
        """
        Downloads and stitches the panorama, handling deduplication.
        """
        zoom_candidates = [
            (5, 26, 13),
            (4, 13, 7),
            (3, 7, 4),
        ]

        for zoom, max_x, max_y in zoom_candidates:
            tile_dict = await self.tile_downloader.download_tiles_async(pano_id, zoom, max_x, max_y)
            if tile_dict is None:
                continue

            # Handle fallback image
            if "full_image" in tile_dict:
                fallback_image = tile_dict["full_image"]
                pano_md5 = hashlib.md5(fallback_image).hexdigest()

                # Deduplication
                if any(info.get("md5") == pano_md5 for info in pano_download_data.values()):
                    self.update_log.emit(f"[Dedup] Fallback image for panoid={pano_id} => MD5 {pano_md5} is a duplicate. Skipping.")
                    return None

                filename = f"{lat:.5f}_{lon:.5f}_{pano_id}_full.jpg"
                out_path = os.path.join(STITCH_DIR, filename)
                with open(out_path, "wb") as f:
                    f.write(fallback_image)

                self.update_log.emit(f"[Fallback Saved] Panorama saved as fallback for panoid={pano_id}.")

                return (filename, pano_md5)

            # Otherwise, stitch the tiles
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
                self.update_log.emit(f"[Dedup] Stitched image for panoid={pano_id} => MD5 {pano_md5} is a duplicate. Skipping.")
                return None

            # Save stitched panorama
            filename = f"{lat:.5f}_{lon:.5f}_{pano_id}.jpg"
            out_path = os.path.join(STITCH_DIR, filename)
            cv2.imwrite(out_path, stitched_result)

            self.update_log.emit(f"[Stitched Saved] Panorama saved for panoid={pano_id}.")

            return (filename, pano_md5)

        # If all zooms + fallback fail
        self.update_log.emit(f"[Failure] No panorama could be downloaded for panoid={pano_id}.")
        return None

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()