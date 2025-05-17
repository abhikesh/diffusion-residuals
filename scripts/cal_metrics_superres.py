#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Calculate metrics for super-resolution images: PSNR, SSIM, LPIPS, CLIPIQA, and MUSIQ.

Usage:
    uv run scripts/cal_metrics_superres.py --gt_dir [path_to_gt] --sr_dir [path_to_sr]
    # Calculate only PSNR and SSIM
    uv run scripts/cal_metrics_superres.py --gt_dir [path_to_gt] --sr_dir [path_to_sr] --metrics psnr ssim
    # For no-reference mode (without ground truth, only calculates specified NR metrics):
    uv run scripts/cal_metrics_superres.py --sr_dir [path_to_sr] --no_reference --metrics clipiqa
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from loguru import logger as base_logger
import csv
import torch
from tqdm import tqdm
import cv2

# Conditionally import metric libraries only if needed (or always import and conditionally use)
try:
    import lpips
except ImportError:
    lpips = None
try:
    import pyiqa
except ImportError:
    pyiqa = None


sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import util_image


def load_im_tensor(im_path):
    """
    Load image and normalize to [-1, 1] for LPIPS
    """
    im = util_image.imread(im_path, chn="rgb", dtype="float32")
    if im is None:
        return None
    im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).cuda()
    im = (im - 0.5) / 0.5
    return im


def load_im_tensor_01(im_path):
    """
    Load image and normalize to [0, 1] for CLIPIQA and MUSIQ
    """
    im = util_image.imread(im_path, chn="rgb", dtype="float32")
    if im is None:
        return None
    im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).cuda()
    return im


def load_im_numpy(im_path):
    """
    Load image as uint8 numpy array for PSNR and SSIM
    """
    im = util_image.imread(im_path, chn="rgb", dtype="uint8")
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, help="Path to the ground truth images")
    parser.add_argument(
        "--sr_dir",
        type=str,
        required=True,
        help="Path to the super-resolution/test images",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=0,
        help="Border pixels to exclude for PSNR and SSIM calculation",
    )
    parser.add_argument(
        "--use_ycbcr",
        action="store_true",
        help="Use Y channel of YCbCr for PSNR and SSIM",
    )
    parser.add_argument(
        "--no_reference",
        action="store_true",
        help="Only calculate no-reference metrics (CLIPIQA, MUSIQ) without ground truth",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=["psnr", "ssim", "lpips_vgg", "lpips_alex", "clipiqa", "musiq"],
        help="Specify which metrics to calculate (default: all applicable).",
    )
    args = parser.parse_args()

    # Validate metric choices based on reference mode
    requested_metrics = set(args.metrics) if args.metrics else None
    supported_metrics = set(
        ["psnr", "ssim", "lpips_vgg", "lpips_alex", "clipiqa", "musiq"]
    )
    reference_metrics = set(["psnr", "ssim", "lpips_vgg", "lpips_alex"])
    no_reference_metrics = set(["clipiqa", "musiq"])

    if requested_metrics:
        if args.no_reference:
            # Filter out reference metrics if in no-reference mode
            metrics_to_run = requested_metrics.intersection(no_reference_metrics)
            if not metrics_to_run:
                parser.error(
                    f"Requested metrics {list(requested_metrics)} are not applicable in --no_reference mode. Choose from {list(no_reference_metrics)}."
                )
        else:
            # In full reference mode, allow all requested metrics
            metrics_to_run = requested_metrics
    else:
        # Default: run all applicable metrics
        if args.no_reference:
            metrics_to_run = no_reference_metrics
        else:
            metrics_to_run = supported_metrics

    # Check library availability for requested metrics
    if (
        "lpips_vgg" in metrics_to_run or "lpips_alex" in metrics_to_run
    ) and lpips is None:
        parser.error("LPIPS metric requested but 'lpips' library is not installed.")
    if ("clipiqa" in metrics_to_run or "musiq" in metrics_to_run) and pyiqa is None:
        parser.error(
            "CLIPIQA or MUSIQ metric requested but 'pyiqa' library is not installed."
        )

    # Check arguments
    if args.no_reference:
        if args.gt_dir:
            base_logger.warning(
                "--gt_dir is provided but will be ignored in no-reference mode"
            )
    else:
        if not args.gt_dir:
            parser.error("--gt_dir is required when not in no-reference mode")
        # Cannot run only NR metrics in reference mode if they were not explicitly requested
        if requested_metrics and not requested_metrics.intersection(reference_metrics):
            parser.error(
                f"Running in reference mode but only non-reference metrics {list(requested_metrics)} were requested."
            )

    # Setting logger
    log_path = str(
        Path(args.sr_dir).parent / f"metrics_{Path(args.sr_dir).name}.log"
    )  # Log in parent dir, named after SR dir
    logger = base_logger
    logger.remove()
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD(HH:mm:ss)}: {message}",
        mode="w",
        level="INFO",
    )
    logger.add(sys.stderr, format="{message}", level="INFO")

    logger.info("--- Configuration ---")
    for key in vars(args):
        value = getattr(args, key)
        logger.info(f"{key}: {value}")
    logger.info(f"Metrics to calculate: {list(metrics_to_run)}")
    logger.info("--- End Configuration ---")

    # Define temporary CSV file paths within the SR directory
    tmp_per_image_csv_path = Path(args.sr_dir) / "_tmp_metrics_per_image.csv"
    tmp_mean_csv_path = Path(args.sr_dir) / "_tmp_metrics_mean.csv"
    logger.info(f"Saving per-image metrics to: {tmp_per_image_csv_path}")
    logger.info(f"Saving mean metrics to: {tmp_mean_csv_path}")

    # Define headers based on metrics_to_run
    all_headers = [
        "image_name",
        "psnr",
        "ssim",
        "lpips_vgg",
        "lpips_alex",
        "clipiqa",
        "musiq",
    ]
    tmp_per_image_header = ["image_name"] + [
        m for m in all_headers[1:] if m in metrics_to_run
    ]
    tmp_mean_header = ["metric_name", "value"]

    # Initialize metric models conditionally
    lpips_metric_vgg = None
    lpips_metric_alex = None
    clipiqa_metric = None
    musiq_metric = None
    metrics_initialized = set()

    try:
        lpips_available = lpips is not None
        pyiqa_available = pyiqa is not None

        if "lpips_vgg" in metrics_to_run and lpips_available:
            lpips_metric_vgg = lpips.LPIPS(net="vgg").cuda()
            metrics_initialized.add("lpips_vgg")
        if "lpips_alex" in metrics_to_run and lpips_available:
            lpips_metric_alex = lpips.LPIPS(net="alex").cuda()
            metrics_initialized.add("lpips_alex")
        if "clipiqa" in metrics_to_run and pyiqa_available:
            clipiqa_metric = pyiqa.create_metric("clipiqa")
            metrics_initialized.add("clipiqa")
        if "musiq" in metrics_to_run and pyiqa_available:
            musiq_metric = pyiqa.create_metric("musiq")
            metrics_initialized.add("musiq")
        # Add psnr and ssim as they don't need external models
        if "psnr" in metrics_to_run:
            metrics_initialized.add("psnr")
        if "ssim" in metrics_to_run:
            metrics_initialized.add("ssim")

        logger.info(
            f"Successfully initialized metric models for: {list(metrics_initialized)}"
        )
        uninitialized_requested = metrics_to_run - metrics_initialized
        if uninitialized_requested:
            logger.warning(
                f"Could not initialize requested metrics: {list(uninitialized_requested)}"
            )
            metrics_to_run = metrics_initialized
            if not metrics_to_run:
                logger.error(
                    "No requested metric models could be initialized. Exiting."
                )
                return
            tmp_per_image_header = ["image_name"] + [
                m for m in all_headers[1:] if m in metrics_to_run
            ]

    except Exception as e:
        logger.error(
            f"Failed during metric model initialization: {e}. Exiting.", exc_info=True
        )
        return

    # Open per-image CSV file and write header
    try:
        tmp_per_image_file = open(tmp_per_image_csv_path, "w", newline="")
        tmp_per_image_writer = csv.writer(tmp_per_image_file)
        tmp_per_image_writer.writerow(tmp_per_image_header)
    except IOError as e:
        logger.error(f"Failed to open {tmp_per_image_csv_path} for writing: {e}")
        return

    # Get all image files from the SR directory
    sr_files = sorted(
        [
            f
            for f in Path(args.sr_dir).glob("*")
            if f.is_file()
            and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
        ]
    )
    logger.info(f"Found {len(sr_files)} images in SR directory: {args.sr_dir}")

    # --- Metrics Accumulators ---
    metrics_totals = {metric: 0 for metric in metrics_to_run}
    valid_count = 0

    # --- Process Images ---
    if args.no_reference:
        # --- No-Reference Mode ---
        logger.info("Processing images in no-reference mode...")
        for sr_path in tqdm(
            sr_files, desc="Processing NR images", file=sys.stderr, dynamic_ncols=True
        ):
            current_metrics = {"image_name": sr_path.name}
            try:
                # Load image for CLIPIQA and MUSIQ (normalized to [0, 1])
                sr_tensor_01 = None
                if "clipiqa" in metrics_to_run or "musiq" in metrics_to_run:
                    sr_tensor_01 = load_im_tensor_01(sr_path)
                    if sr_tensor_01 is None:
                        logger.warning(
                            f"Could not load SR image {sr_path.name}, skipping."
                        )
                        continue

                # Calculate requested metrics
                with torch.no_grad():
                    if "clipiqa" in metrics_to_run and clipiqa_metric:
                        score = clipiqa_metric(sr_tensor_01).item()
                        current_metrics["clipiqa"] = score
                        metrics_totals["clipiqa"] += score
                    if "musiq" in metrics_to_run and musiq_metric:
                        score = musiq_metric(sr_tensor_01).item()
                        current_metrics["musiq"] = score
                        metrics_totals["musiq"] += score

                # Write per-image CSV row
                tmp_per_image_writer.writerow(
                    [current_metrics.get(h) for h in tmp_per_image_header]
                )
                valid_count += 1

            except Exception as e:
                logger.error(
                    f"Error processing {sr_path.name}: {e}", exc_info=False
                )  # Don't print full traceback for each image error

    else:
        # --- Full Reference Mode ---
        logger.info("Processing images in full-reference mode...")
        gt_dir = Path(args.gt_dir)
        logger.info(f"Comparing against GT directory: {gt_dir}")

        for sr_path in tqdm(
            sr_files, desc="Processing FR images", file=sys.stderr, dynamic_ncols=True
        ):
            gt_path = gt_dir / sr_path.name
            current_metrics = {"image_name": sr_path.name}

            if not gt_path.is_file():
                logger.warning(f"  Corresponding GT image not found: {gt_path}")
                continue

            try:
                # Load images for PSNR/SSIM if needed
                gt_img, sr_img = None, None
                if "psnr" in metrics_to_run or "ssim" in metrics_to_run:
                    gt_img = load_im_numpy(gt_path)
                    sr_img = load_im_numpy(sr_path)
                    if gt_img is None:
                        logger.warning(
                            f"Could not load GT image {gt_path.name}, skipping pair."
                        )
                        continue
                    if sr_img is None:
                        logger.warning(
                            f"Could not load SR image {sr_path.name}, skipping pair."
                        )
                        continue

                    if gt_img.shape != sr_img.shape:
                        logger.warning(
                            f"Shape mismatch for {sr_path.name}: GT={gt_img.shape}, SR={sr_img.shape}. Resizing SR image."
                        )
                        sr_img = cv2.resize(
                            sr_img,
                            (gt_img.shape[1], gt_img.shape[0]),
                            interpolation=cv2.INTER_CUBIC,
                        )

                # Load images for LPIPS if needed
                gt_tensor, sr_tensor = None, None
                if (
                    "lpips_vgg" in metrics_to_run or "lpips_alex" in metrics_to_run
                ) and lpips:
                    gt_tensor = load_im_tensor(gt_path)
                    sr_tensor = load_im_tensor(sr_path)
                    if gt_tensor is None or sr_tensor is None:
                        logger.warning(
                            f"Could not load tensor images for LPIPS for {sr_path.name}, skipping LPIPS."
                        )

                # Load images for NR metrics if needed
                sr_tensor_01 = None
                if ("clipiqa" in metrics_to_run or "musiq" in metrics_to_run) and pyiqa:
                    sr_tensor_01 = load_im_tensor_01(sr_path)
                    if sr_tensor_01 is None:
                        logger.warning(
                            f"Could not load tensor image for NR metrics for {sr_path.name}, skipping NR."
                        )

                # Calculate requested metrics
                if (
                    "psnr" in metrics_to_run
                    and gt_img is not None
                    and sr_img is not None
                ):
                    score = util_image.calculate_psnr(
                        gt_img, sr_img, border=args.border, ycbcr=args.use_ycbcr
                    )
                    current_metrics["psnr"] = score
                    metrics_totals["psnr"] += score
                if (
                    "ssim" in metrics_to_run
                    and gt_img is not None
                    and sr_img is not None
                ):
                    score = util_image.calculate_ssim(
                        gt_img, sr_img, border=args.border, ycbcr=args.use_ycbcr
                    )
                    current_metrics["ssim"] = score
                    metrics_totals["ssim"] += score

                with torch.no_grad():
                    if (
                        "lpips_vgg" in metrics_to_run
                        and lpips_metric_vgg
                        and gt_tensor is not None
                        and sr_tensor is not None
                    ):
                        score = lpips_metric_vgg(gt_tensor, sr_tensor).item()
                        current_metrics["lpips_vgg"] = score
                        metrics_totals["lpips_vgg"] += score
                    if (
                        "lpips_alex" in metrics_to_run
                        and lpips_metric_alex
                        and gt_tensor is not None
                        and sr_tensor is not None
                    ):
                        score = lpips_metric_alex(gt_tensor, sr_tensor).item()
                        current_metrics["lpips_alex"] = score
                        metrics_totals["lpips_alex"] += score
                    if (
                        "clipiqa" in metrics_to_run
                        and clipiqa_metric
                        and sr_tensor_01 is not None
                    ):
                        score = clipiqa_metric(sr_tensor_01).item()
                        current_metrics["clipiqa"] = score
                        metrics_totals["clipiqa"] += score
                    if (
                        "musiq" in metrics_to_run
                        and musiq_metric
                        and sr_tensor_01 is not None
                    ):
                        score = musiq_metric(sr_tensor_01).item()
                        current_metrics["musiq"] = score
                        metrics_totals["musiq"] += score

                # Write per-image CSV row
                if gt_img is not None and sr_img is not None:
                    tmp_per_image_writer.writerow(
                        [current_metrics.get(h) for h in tmp_per_image_header]
                    )
                    valid_count += 1

            except Exception as e:
                logger.error(
                    f"Error processing pair GT: {gt_path.name}, SR: {sr_path.name}: {e}",
                    exc_info=False,
                )

    # --- Finalize and Report ---
    tmp_per_image_file.close()  # Close the per-image file

    if valid_count == 0:
        logger.error("No valid images or image pairs processed for evaluation.")
        # Clean up empty CSV file
        try:
            # Check if only header exists (approximate check)
            header_size = len(",".join(tmp_per_image_header).encode("utf-8")) + len(
                os.linesep.encode("utf-8")
            )
            if (
                tmp_per_image_csv_path.is_file()
                and tmp_per_image_csv_path.stat().st_size <= header_size
            ):
                tmp_per_image_csv_path.unlink()
                logger.info(
                    f"Removed empty or header-only per-image CSV: {tmp_per_image_csv_path}"
                )
        except Exception as e:
            logger.warning(
                f"Could not check/remove potentially empty CSV: {e}"
            )  # Ignore errors during cleanup
        return

    logger.info(f"\n--- Average Metrics ({valid_count} images) ---")
    avg_metrics = {}
    for metric, total in metrics_totals.items():
        if metric in metrics_to_run:
            if valid_count > 0:
                avg = total / valid_count
                avg_metrics[f"mean_{metric}"] = avg
                format_str = ".4f" if metric not in ["musiq"] else ".2f"
                logger.info(f"MEAN {metric.upper()}: {avg:{format_str}}")
            else:
                logger.info(f"MEAN {metric.upper()}: N/A (no valid images)")

    # --- Write Mean CSV File ---
    try:
        with open(tmp_mean_csv_path, "w", newline="") as mean_file:
            mean_writer = csv.writer(mean_file)
            mean_writer.writerow(tmp_mean_header)
            for metric_name_full in sorted(avg_metrics.keys()):
                metric_name_base = metric_name_full.replace("mean_", "")
                format_str = ".4f" if metric_name_base not in ["musiq"] else ".2f"
                mean_writer.writerow(
                    [metric_name_full, f"{avg_metrics[metric_name_full]:{format_str}}"]
                )
            all_possible_mean_metrics = [f"mean_{m}" for m in metrics_to_run]
            for m_name in all_possible_mean_metrics:
                if m_name not in avg_metrics:
                    mean_writer.writerow([m_name, None])

        logger.info(f"Mean metrics saved to: {tmp_mean_csv_path}")
    except IOError as e:
        logger.error(f"Failed to write mean metrics CSV {tmp_mean_csv_path}: {e}")
    # --- End Mean CSV Write ---

    logger.info("--- Metrics Calculation Complete ---")


if __name__ == "__main__":
    main()
