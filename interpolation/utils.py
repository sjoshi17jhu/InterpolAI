import os
import re

import imageio
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)


def load_image(img_path: str) -> np.ndarray:
    image = imageio.imread(img_path)
    return image.astype(np.float32) / _UINT8_MAX_F


def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1]


def pad_and_tile_image(image: np.ndarray, tile_size: tuple):
    """Pad image to tile boundaries and reshape into a grid of tiles.

    Returns:
        tiled_array: shape (rows, cols, tile_h, tile_w, channels)
        (pad_h, pad_w): padding added, needed to crop after stitching
    """
    h, w, _ = image.shape
    tile_height, tile_width = tile_size
    pad_h = (tile_height - h % tile_height) % tile_height
    pad_w = (tile_width - w % tile_width) % tile_width
    padded = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode="constant", constant_values=0)
    ph, pw, _ = padded.shape
    tiled = padded.reshape(ph // tile_height, tile_height, pw // tile_width, tile_width, -1)
    return tiled.swapaxes(1, 2), (pad_h, pad_w)


def stitch_tiles(tiles: np.ndarray, pad_h: int, pad_w: int, tile_size: tuple) -> np.ndarray:
    """Reconstruct a full image from a grid of tiles, removing any padding."""
    tile_height, tile_width = tile_size
    tile_rows, tile_cols, _, _, _ = tiles.shape
    stitched = tiles.swapaxes(1, 2).reshape(tile_rows * tile_height, tile_cols * tile_width, -1)
    if pad_h > 0 or pad_w > 0:
        stitched = stitched[: -pad_h or None, : -pad_w or None]
    return stitched


def extract_number_from_filename(filename: str):
    """Extract the last integer from a filename stem, or None if absent."""
    match = re.search(r"(\d+)(?=\D*$)", filename)
    return int(match.group(1)) if match else None


def make_output_filename(source_filename: str, idx: int, file_extension: str) -> str:
    """Generate a sequentially numbered output filename from a source filename.

    The trailing number in source_filename is incremented by (idx + 1),
    preserving zero-padding.  Falls back to a ``_int<n>`` suffix when no
    trailing number is found.

    Examples:
        make_output_filename("img_001.tif", 0, ".tif") -> "img_002.tif"
        make_output_filename("img_001.tif", 1, ".tif") -> "img_003.tif"
        make_output_filename("image.tif",   0, ".tif") -> "image_int1.tif"
    """
    base_name = os.path.splitext(source_filename)[0]
    match = re.search(r"(\d+)$", base_name)
    if match:
        number_part = match.group(1)
        new_number = str(int(number_part) + (idx + 1)).zfill(len(number_part))
        return f"{base_name[: -len(number_part)]}{new_number}{file_extension}"
    return f"{base_name}_int{idx + 1}{file_extension}"
