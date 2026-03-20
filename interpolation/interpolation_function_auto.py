import os

import imageio
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from .utils import (
    _UINT8_MAX_F,
    extract_number_from_filename,
    get_file_extension,
    load_image,
    make_output_filename,
    pad_and_tile_image,
    stitch_tiles,
)


def list_skip_images(pthims: str) -> dict:
    """Detect missing images in a numbered sequence.

    Returns a dict mapping skip_index -> list of [prev_num, next_num] pairs
    that bracket each gap.
    """
    image_files = [f for f in os.listdir(pthims) if f.endswith(("tif", "png", "jpg"))]
    list_z_nums = sorted(filter(None, [extract_number_from_filename(f) for f in image_files]))

    full_range = set(range(min(list_z_nums), max(list_z_nums) + 1))
    missing_numbers = sorted(full_range - set(list_z_nums))

    skip_inputs: dict = {}
    added_pairs: set = set()

    for missing in missing_numbers:
        prev_image = max([n for n in list_z_nums if n < missing], default=None)
        next_image = min([n for n in list_z_nums if n > missing], default=None)

        if prev_image is not None and next_image is not None:
            skip_index = next_image - prev_image - 1
            pair = (prev_image, next_image)
            if skip_index not in skip_inputs:
                skip_inputs[skip_index] = []
            if pair not in added_pairs:
                skip_inputs[skip_index].append([prev_image, next_image])
                added_pairs.add(pair)

    return skip_inputs


def interpolate_from_image_list(pthims, skip_images, TILE_SIZE, model, image_files, output_dir=None):
    """Interpolate missing frames detected automatically by list_skip_images.

    Args:
        pthims: path to the folder containing source images.
        skip_images: dict returned by list_skip_images.
        TILE_SIZE: (height, width) tile size for large images.
        model: loaded TensorFlow saved model.
        image_files: list of image filenames in pthims.
        output_dir: root folder for output subfolders.  Defaults to pthims.
    """
    base_output = output_dir or pthims
    image_files = natsorted(image_files)
    file_extension = get_file_extension(image_files[0])
    image_dict = {extract_number_from_filename(f): f for f in image_files}

    for skip_num, image_pairs in skip_images.items():
        print(f"Interpolating for skip {skip_num}:")
        output_folder = os.path.join(base_output, f"int_{skip_num}")
        os.makedirs(output_folder, exist_ok=True)
        times = np.linspace(0, 1, skip_num + 2)[1:-1]

        for img1_num, img2_num in image_pairs:
            image1_path = os.path.join(pthims, image_dict[img1_num])
            image2_path = os.path.join(pthims, image_dict[img2_num])

            all_exist = all(
                os.path.exists(
                    os.path.join(output_folder, make_output_filename(image_dict[img1_num], idx, file_extension))
                )
                for idx in range(len(times))
            )
            if all_exist:
                print(
                    f"All interpolated images from {image_dict[img1_num]} and "
                    f"{image_dict[img2_num]} already exist. Skipping..."
                )
                continue

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

                with tqdm(total=len(times), desc=f"Interpolating {image_dict[img1_num]}", unit="frame") as pbar:
                    for idx, time_value in enumerate(times):
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

                        filename = make_output_filename(image_dict[img1_num], idx, file_extension)
                        tiles_np = np.array(tiles_out).reshape(tile_rows, tile_cols, tile_height, tile_width, num_channels)
                        stitched = stitch_tiles(tiles_np, pad_h, pad_w, TILE_SIZE)
                        imageio.imwrite(os.path.join(output_folder, filename), stitched, format=file_extension.lstrip("."))
                        pbar.update(1)
            else:
                print("No stitching needed")
                with tqdm(total=len(times), desc=f"Interpolating {image_dict[img1_num]}", unit="frame") as pbar:
                    for idx, time_value in enumerate(times):
                        filename = make_output_filename(image_dict[img1_num], idx, file_extension)
                        output_path = os.path.join(output_folder, filename)
                        if os.path.exists(output_path):
                            print(f"Already interpolated from {image_dict[img1_num]} and {image_dict[img2_num]}")
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
                        pbar.update(1)
