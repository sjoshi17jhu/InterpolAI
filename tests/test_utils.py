"""Tests for model-free utility functions.

These tests do not require TensorFlow or the model weights — they cover the
pure-Python / NumPy helpers that can be validated in CI without a GPU.
"""

import numpy as np
import pytest

from interpolation.utils import (
    extract_number_from_filename,
    make_output_filename,
    pad_and_tile_image,
    stitch_tiles,
)
from interpolation.interpolation_function_auto import list_skip_images


# ---------------------------------------------------------------------------
# extract_number_from_filename
# ---------------------------------------------------------------------------

class TestExtractNumber:
    def test_simple_trailing_number(self):
        assert extract_number_from_filename("img_001.tif") == 1

    def test_number_at_end_of_stem(self):
        assert extract_number_from_filename("frame42.png") == 42

    def test_no_number_returns_none(self):
        assert extract_number_from_filename("image.tif") is None

    def test_multiple_numbers_returns_last(self):
        # "series2_frame010" — last number is 010 → 10
        assert extract_number_from_filename("series2_frame010.tif") == 10

    def test_zero_padded(self):
        assert extract_number_from_filename("z_0007.tif") == 7


# ---------------------------------------------------------------------------
# make_output_filename
# ---------------------------------------------------------------------------

class TestMakeOutputFilename:
    def test_numbered_first_interpolation(self):
        assert make_output_filename("img_001.tif", 0, ".tif") == "img_002.tif"

    def test_numbered_second_interpolation(self):
        assert make_output_filename("img_001.tif", 1, ".tif") == "img_003.tif"

    def test_preserves_zero_padding(self):
        assert make_output_filename("frame_010.png", 0, ".png") == "frame_011.png"

    def test_fallback_suffix_when_no_trailing_number(self):
        assert make_output_filename("image.tif", 0, ".tif") == "image_int1.tif"

    def test_fallback_second_interpolation(self):
        assert make_output_filename("image.tif", 2, ".tif") == "image_int3.tif"


# ---------------------------------------------------------------------------
# pad_and_tile_image
# ---------------------------------------------------------------------------

class TestPadAndTileImage:
    def test_exact_fit_no_padding(self):
        image = np.zeros((256, 256, 3), dtype=np.float32)
        tiles, (pad_h, pad_w) = pad_and_tile_image(image, (256, 256))
        assert pad_h == 0
        assert pad_w == 0
        assert tiles.shape == (1, 1, 256, 256, 3)

    def test_needs_padding_produces_correct_shape(self):
        image = np.zeros((300, 400, 3), dtype=np.float32)
        tiles, (pad_h, pad_w) = pad_and_tile_image(image, (256, 256))
        # 300 -> padded to 512 (2 tiles), 400 -> padded to 512 (2 tiles)
        assert tiles.shape == (2, 2, 256, 256, 3)
        assert pad_h == 256 - (300 % 256)  # 212
        assert pad_w == 256 - (400 % 256)  # 112

    def test_pixel_values_preserved_in_first_tile(self):
        image = np.random.rand(256, 256, 3).astype(np.float32)
        tiles, (pad_h, pad_w) = pad_and_tile_image(image, (256, 256))
        assert pad_h == 0 and pad_w == 0
        np.testing.assert_array_equal(tiles[0, 0], image)

    def test_padding_region_is_zero(self):
        image = np.ones((100, 100, 3), dtype=np.float32)
        tiles, (pad_h, pad_w) = pad_and_tile_image(image, (256, 256))
        assert pad_h == 156 and pad_w == 156
        # Bottom-right corner of the single tile should be zero (padding)
        assert tiles[0, 0, 200, 200, 0] == 0.0


# ---------------------------------------------------------------------------
# stitch_tiles (roundtrip with pad_and_tile_image)
# ---------------------------------------------------------------------------

class TestStitchTiles:
    def test_roundtrip_exact_fit(self):
        image = np.random.rand(512, 512, 3).astype(np.float32)
        tiles, (pad_h, pad_w) = pad_and_tile_image(image, (256, 256))
        reconstructed = stitch_tiles(tiles, pad_h, pad_w, (256, 256))
        np.testing.assert_array_almost_equal(reconstructed, image)

    def test_roundtrip_with_padding(self):
        """Tile → stitch should recover the original shape and values exactly."""
        image = np.random.rand(300, 400, 3).astype(np.float32)
        tiles, (pad_h, pad_w) = pad_and_tile_image(image, (256, 256))
        reconstructed = stitch_tiles(tiles, pad_h, pad_w, (256, 256))
        assert reconstructed.shape == image.shape
        np.testing.assert_array_almost_equal(reconstructed, image)

    def test_roundtrip_single_channel(self):
        image = np.random.rand(128, 128, 1).astype(np.float32)
        tiles, (pad_h, pad_w) = pad_and_tile_image(image, (64, 64))
        reconstructed = stitch_tiles(tiles, pad_h, pad_w, (64, 64))
        assert reconstructed.shape == image.shape
        np.testing.assert_array_almost_equal(reconstructed, image)


# ---------------------------------------------------------------------------
# list_skip_images
# ---------------------------------------------------------------------------

class TestListSkipImages:
    def test_detects_single_missing_image(self, tmp_path):
        # z_001, z_002, z_004, z_005 — z_003 is missing
        for n in [1, 2, 4, 5]:
            (tmp_path / f"z_{n:03d}.tif").touch()
        result = list_skip_images(str(tmp_path))
        # gap between 2 and 4 → skip_index = 4 - 2 - 1 = 1
        assert 1 in result
        assert [2, 4] in result[1]

    def test_no_missing_images_returns_empty(self, tmp_path):
        for n in [1, 2, 3]:
            (tmp_path / f"z_{n:03d}.tif").touch()
        result = list_skip_images(str(tmp_path))
        assert result == {}

    def test_multiple_gaps_different_sizes(self, tmp_path):
        # z_001, z_003, z_006 → gap of 1 between 1–3, gap of 2 between 3–6
        for n in [1, 3, 6]:
            (tmp_path / f"z_{n:03d}.tif").touch()
        result = list_skip_images(str(tmp_path))
        assert 1 in result   # one image missing between 1 and 3
        assert 2 in result   # two images missing between 3 and 6

    def test_ignores_non_image_files(self, tmp_path):
        for n in [1, 2, 3]:
            (tmp_path / f"z_{n:03d}.tif").touch()
        (tmp_path / "notes.txt").touch()
        result = list_skip_images(str(tmp_path))
        assert result == {}
