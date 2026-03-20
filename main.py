import argparse
import os

import tensorflow as tf

from interpolation import (
    interpolate_from_image_list,
    interpolate_from_image_stack_no_skip,
    interpolate_from_image_stack_skip,
    list_skip_images,
)


def load_model():
    model_path = os.path.join("interpolation", "model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Download the model folder from the link in the README and place it at interpolation/model."
        )
    return tf.saved_model.load(model_path)


def run_auto(tile_size, pth, model, output_dir=None):
    image_files = [f for f in os.listdir(pth) if f.endswith(("tif", "png", "jpg"))]
    if not image_files:
        raise ValueError(f"No .tif/.png/.jpg images found in '{pth}'")
    skip_images = list_skip_images(pth)
    if not skip_images:
        print("No missing images detected in sequence — nothing to interpolate.")
        return
    interpolate_from_image_list(pth, skip_images, tile_size, model, image_files, output_dir=output_dir)


def run_no_skip(tile_size, pth, skip, model, output_dir=None):
    image_files = [f for f in os.listdir(pth) if f.endswith(("tif", "png", "jpg"))]
    if not image_files:
        raise ValueError(f"No .tif/.png/.jpg images found in '{pth}'")
    interpolate_from_image_stack_no_skip(pth, skip, tile_size, model, image_files, output_dir=output_dir)


def run_skip(tile_size, pth, skip, model, output_dir=None):
    image_files = [f for f in os.listdir(pth) if f.endswith(("tif", "png", "jpg"))]
    if not image_files:
        raise ValueError(f"No .tif/.png/.jpg images found in '{pth}'")
    interpolate_from_image_stack_skip(pth, skip, tile_size, model, image_files, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InterpolAI — AI-based image sequence interpolation")
    parser.add_argument("--mode", choices=["auto", "no_skip", "skip"], required=True, help="Interpolation mode")
    parser.add_argument("--tile_size", type=int, nargs=2, required=True, metavar=("H", "W"), help="Tile size for large images, e.g. --tile_size 1024 1024")
    parser.add_argument("--pth", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--output", type=str, default=None, help="Root folder for output subfolders (default: same as --pth)")
    parser.add_argument("--skip", nargs="+", type=int, default=[1], help="Skip values for no_skip/skip modes (default: 1)")

    args = parser.parse_args()
    model = load_model()

    if args.mode == "auto":
        run_auto(tuple(args.tile_size), args.pth, model, output_dir=args.output)
    elif args.mode == "no_skip":
        run_no_skip(tuple(args.tile_size), args.pth, args.skip, model, output_dir=args.output)
    elif args.mode == "skip":
        run_skip(tuple(args.tile_size), args.pth, args.skip, model, output_dir=args.output)
