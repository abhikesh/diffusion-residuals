#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Calculate metrics for super-resolution images: PSNR, SSIM, LPIPS, CLIPIQA, and MUSIQ.

Usage:
    uv run scripts/cal_metrics_superres.py --gt_dir [path_to_gt] --sr_dir [path_to_sr]

    # For no-reference mode (without ground truth):
    uv run scripts/cal_metrics_superres.py --sr_dir [path_to_sr] --no_reference
"""

import os
import sys
import lpips
import pyiqa
import argparse
import numpy as np
from pathlib import Path
from loguru import logger as base_logger

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import util_image


def load_im_tensor(im_path):
    """
    Load image and normalize to [-1, 1] for LPIPS
    """
    im = util_image.imread(im_path, chn="rgb", dtype="float32")
    im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).cuda()
    im = (im - 0.5) / 0.5

    return im


def load_im_tensor_01(im_path):
    """
    Load image and normalize to [0, 1] for CLIPIQA and MUSIQ
    """
    im = util_image.imread(im_path, chn="rgb", dtype="float32")
    im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).cuda()
    # No normalization needed as imread already returns [0, 1]
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
    args = parser.parse_args()

    # Check arguments
    if args.no_reference:
        if args.gt_dir:
            base_logger.warning(
                "--gt_dir is provided but will be ignored in no-reference mode"
            )
    else:
        if not args.gt_dir:
            parser.error("--gt_dir is required when not in no-reference mode")

    # Setting logger
    log_path = str(Path(args.sr_dir).parent / "metrics_superres.log")
    logger = base_logger
    logger.remove()
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD(HH:mm:ss)}: {message}",
        mode="w",
        level="INFO",
    )
    logger.add(sys.stderr, format="{message}", level="INFO")

    for key in vars(args):
        value = getattr(args, key)
        logger.info(f"{key}: {value}")

    # Get all image files from the SR directory
    sr_dir = Path(args.sr_dir)
    sr_files = [
        f
        for f in sr_dir.glob("*")
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    ]
    logger.info(f"Found {len(sr_files)} images in SR directory")

    if args.no_reference:
        # No-reference mode: Only calculate CLIPIQA and MUSIQ
        logger.info("Running in no-reference mode (CLIPIQA and MUSIQ only)")

        # Initialize no-reference metrics
        clipiqa_metric = pyiqa.create_metric("clipiqa")
        musiq_metric = pyiqa.create_metric("musiq")

        # Metrics accumulators
        total_clipiqa = 0
        total_musiq = 0
        valid_count = 0

        for sr_path in sr_files:
            # Load image for CLIPIQA and MUSIQ (normalized to [0, 1])
            sr_tensor_01 = load_im_tensor_01(sr_path)

            # Calculate metrics
            with torch.no_grad():
                clipiqa_score = clipiqa_metric(sr_tensor_01).item()
                musiq_score = musiq_metric(sr_tensor_01).item()

            # Log individual image metrics
            logger.info(f"Image: {sr_path.name}")
            logger.info(f"  CLIPIQA: {clipiqa_score:.4f}")
            logger.info(f"  MUSIQ: {musiq_score:.2f}")

            # Accumulate metrics
            total_clipiqa += clipiqa_score
            total_musiq += musiq_score
            valid_count += 1

        if valid_count == 0:
            logger.error("No valid images found for evaluation")
            return

        # Calculate and log average metrics
        avg_clipiqa = total_clipiqa / valid_count
        avg_musiq = total_musiq / valid_count

        logger.info(f"\nAverage metrics over {valid_count} images:")
        logger.info(f"MEAN CLIPIQA: {avg_clipiqa:.4f}")
        logger.info(f"MEAN MUSIQ: {avg_musiq:.2f}")

    else:
        # Full reference mode: Calculate all metrics
        logger.info("Running in full-reference mode (all metrics)")

        # Initialize metrics
        lpips_metric_vgg = lpips.LPIPS(net="vgg").cuda()
        lpips_metric_alex = lpips.LPIPS(net="alex").cuda()
        clipiqa_metric = pyiqa.create_metric("clipiqa")
        musiq_metric = pyiqa.create_metric("musiq")

        # Get GT files
        gt_dir = Path(args.gt_dir)
        gt_files = [
            f
            for f in gt_dir.glob("*")
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ]
        logger.info(f"Found {len(gt_files)} images in GT directory")

        # Metrics accumulators
        total_psnr = 0
        total_ssim = 0
        total_lpips_vgg = 0
        total_lpips_alex = 0
        total_clipiqa = 0
        total_musiq = 0
        valid_count = 0

        for gt_path in gt_files:
            sr_path = sr_dir / gt_path.name

            if not sr_path.exists():
                logger.warning(f"SR image not found: {sr_path}")
                continue

            # Load images for PSNR and SSIM (uint8)
            gt_img = load_im_numpy(gt_path)
            sr_img = load_im_numpy(sr_path)

            # Ensure images have the same shape (required for metrics)
            if gt_img.shape != sr_img.shape:
                logger.warning(
                    f"Shape mismatch: {gt_path.name}, GT: {gt_img.shape}, SR: {sr_img.shape}"
                )
                continue

            # Calculate PSNR and SSIM
            psnr = util_image.calculate_psnr(
                gt_img, sr_img, border=args.border, ycbcr=args.use_ycbcr
            )
            ssim = util_image.calculate_ssim(
                gt_img, sr_img, border=args.border, ycbcr=args.use_ycbcr
            )

            # Load images for LPIPS (normalized to [-1, 1])
            gt_tensor = load_im_tensor(gt_path)
            sr_tensor = load_im_tensor(sr_path)

            # Load images for CLIPIQA and MUSIQ (normalized to [0, 1])
            gt_tensor_01 = load_im_tensor_01(gt_path)
            sr_tensor_01 = load_im_tensor_01(sr_path)

            # Calculate metrics
            with torch.no_grad():
                lpips_vgg = lpips_metric_vgg(gt_tensor, sr_tensor).item()
                lpips_alex = lpips_metric_alex(gt_tensor, sr_tensor).item()
                clipiqa_score = clipiqa_metric(sr_tensor_01).item()
                musiq_score = musiq_metric(sr_tensor_01).item()

            # Log individual image metrics
            logger.info(f"Image: {gt_path.name}")
            logger.info(f"  PSNR: {psnr:.4f}")
            logger.info(f"  SSIM: {ssim:.4f}")
            logger.info(f"  LPIPS-VGG: {lpips_vgg:.4f}")
            logger.info(f"  LPIPS-Alex: {lpips_alex:.4f}")
            logger.info(f"  CLIPIQA: {clipiqa_score:.4f}")
            logger.info(f"  MUSIQ: {musiq_score:.2f}")

            # Accumulate metrics
            total_psnr += psnr
            total_ssim += ssim
            total_lpips_vgg += lpips_vgg
            total_lpips_alex += lpips_alex
            total_clipiqa += clipiqa_score
            total_musiq += musiq_score

            valid_count += 1

        if valid_count == 0:
            logger.error("No valid image pairs found for evaluation")
            return

        # Calculate and log average metrics
        avg_psnr = total_psnr / valid_count
        avg_ssim = total_ssim / valid_count
        avg_lpips_vgg = total_lpips_vgg / valid_count
        avg_lpips_alex = total_lpips_alex / valid_count
        avg_clipiqa = total_clipiqa / valid_count
        avg_musiq = total_musiq / valid_count

        logger.info(f"\nAverage metrics over {valid_count} images:")
        logger.info(f"MEAN PSNR: {avg_psnr:.4f}")
        logger.info(f"MEAN SSIM: {avg_ssim:.4f}")
        logger.info(f"MEAN LPIPS-VGG: {avg_lpips_vgg:.4f}")
        logger.info(f"MEAN LPIPS-Alex: {avg_lpips_alex:.4f}")
        logger.info(f"MEAN CLIPIQA: {avg_clipiqa:.4f}")
        logger.info(f"MEAN MUSIQ: {avg_musiq:.2f}")


if __name__ == "__main__":
    main()
