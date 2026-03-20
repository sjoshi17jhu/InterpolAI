import os
import time

import imageio
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from .utils import (
    _UINT8_MAX_F,
    get_file_extension,
    load_image,
    pad_and_tile_image,
    stitch_tiles,
)


def interpolate_from_image_stack_no_skip(pthims, skips, TILE_SIZE, model, image_files=None, output_dir=None):
    """Interpolate n frames between every consecutive image pair.

    For each skip value n, generates n frames between image[i] and image[i+1]
    for every consecutive pair in the folder.

    Args:
        pthims: path to source image folder.
        skips: list of interpolation counts (frames to insert between each pair).
        TILE_SIZE: (height, width) tile size for large images.
        model: loaded TensorFlow saved model.
        image_files: unused; kept for API compatibility.
        output_dir: root folder for output subfolders.  Defaults to pthims.
    """
    base_output = output_dir or pthims
    image_files = natsorted([f for f in os.listdir(pthims) if f.endswith(("tif", "png", "jpg"))])
    file_extension = get_file_extension(image_files[0])
    image2 = None

    for skip_num in skips:
        print(f"Interpolating int {skip_num}:")
        output_folder = os.path.join(base_output, f"int_{skip_num}_no_skip")
        os.makedirs(output_folder, exist_ok=True)
        times = np.linspace(0, 1, skip_num + 2)[1:-1]

        for i in range(len(image_files) - 1):
            image1_path = os.path.join(pthims, image_files[i])
            image2_path = os.path.join(pthims, image_files[i + 1])

            stem1 = os.path.splitext(image_files[i])[0]
            output_checkpath = os.path.join(output_folder, f"{stem1}_int{skip_num}{file_extension}")
            if os.path.exists(output_checkpath):
                print(f"Already interpolated from {image_files[i]} and {image_files[i + 1]}")
                continue

            if image2 is not None:
                image1 = image2
            else:
                image1 = load_image(image1_path)
            image2 = load_image(image2_path)

            needs_tiling = max(image1.shape[:2]) > TILE_SIZE[0] or max(image2.shape[:2]) > TILE_SIZE[1]

            if needs_tiling:
                print("Stitching needed")
                tiles1, (pad_h, pad_w) = pad_and_tile_image(image1, TILE_SIZE)
                tiles2, _ = pad_and_tile_image(image2, TILE_SIZE)
                tile_height, tile_width = TILE_SIZE
                num_channels = image1.shape[-1]
                tile_rows, tile_cols, _, _, _ = tiles1.shape

                with tqdm(total=len(times), desc=f"Interpolating {image_files[i]}", unit="frame") as pbar:
                    for idx, time_value in enumerate(times):
                        start_time = time.time()
                        tiles_out = []
                        for tile_row in range(tile_rows):
                            for tile_col in range(tile_cols):
                                input_data = {
                                    "time": np.array([[time_value]], dtype=np.float32),
                                    "x0": np.expand_dims(tiles1[tile_row, tile_col], axis=0),
                                    "x1": np.expand_dims(tiles2[tile_row, tile_col], axis=0),
                                }
                                generated = model(input_data)["image"][0].numpy()
                                tile_uint8 = (np.clip(generated * _UINT8_MAX_F, 0, _UINT8_MAX_F) + 0.5).astype(np.uint8)
                                tiles_out.append(tile_uint8)

                        filename = f"{stem1}_int{idx + 1}{file_extension}"
                        tiles_np = np.array(tiles_out).reshape(tile_rows, tile_cols, tile_height, tile_width, num_channels)
                        stitched = stitch_tiles(tiles_np, pad_h, pad_w, TILE_SIZE)
                        imageio.imwrite(os.path.join(output_folder, filename), stitched, format=file_extension.lstrip("."))
                        elapsed = time.time() - start_time
                        print(f"Time to generate {filename}: {elapsed:.2f} seconds")
                        pbar.update(1)
            else:
                print("No stitching needed")
                with tqdm(total=len(times), desc=f"Interpolating {image_files[i]}", unit="frame") as pbar:
                    for idx, time_value in enumerate(times):
                        start_time = time.time()
                        filename = f"{stem1}_int{idx + 1}{file_extension}"
                        output_path = os.path.join(output_folder, filename)
                        if os.path.exists(output_path):
                            print(f"Already interpolated from {image_files[i]} and {image_files[i + 1]}")
                            pbar.update(1)
                            continue

                        input_data = {
                            "time": np.array([[time_value]], dtype=np.float32),
                            "x0": np.expand_dims(image1, axis=0),
                            "x1": np.expand_dims(image2, axis=0),
                        }
                        generated = model(input_data)["image"][0].numpy()
                        image_uint8 = (np.clip(generated * _UINT8_MAX_F, 0, _UINT8_MAX_F) + 0.5).astype(np.uint8)
                        imageio.imwrite(output_path, image_uint8, format=file_extension.lstrip("."))
                        elapsed = time.time() - start_time
                        print(f"Time to generate {filename}: {elapsed:.2f} seconds")
                        pbar.update(1)
