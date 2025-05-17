import argparse
import cv2
import numpy as np
import os
import math
import random
from glob import glob
from tqdm import tqdm  # For progress bar


def get_motion_blur_kernel(kernel_size: int, angle_deg: float) -> np.ndarray:
    """Generates a motion blur kernel."""
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    # Convert angle to radians
    angle_rad = math.radians(angle_deg)

    # Calculate endpoint based on angle and center
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # Bresenham's line algorithm adaptation for kernel generation
    x0, y0 = center, center
    # Calculate end point relative to center, ensuring it's within bounds for line drawing
    # We scale by center distance to draw line across the kernel
    x1 = int(round(center + center * dx))
    y1 = int(round(center + center * dy))

    # Clamp coordinates to kernel boundaries
    x1 = max(0, min(kernel_size - 1, x1))
    y1 = max(0, min(kernel_size - 1, y1))

    # Draw the line on the kernel using cv2.line for simplicity
    cv2.line(
        kernel, (y0, x0), (y1, x1), 1.0, thickness=1
    )  # Note: cv2 uses (x,y) but numpy uses (row,col) -> (y,x)

    # Normalize the kernel
    kernel_sum = kernel.sum()
    if (
        kernel_sum == 0
    ):  # Avoid division by zero if line has zero length (e.g., angle edge cases)
        kernel[center, center] = 1.0
    else:
        kernel /= kernel_sum

    return kernel


def process_images(
    hr_dir: str,
    output_dir: str,
    degradation_type: str,
    kernel_size: int,
    noise_sigma: float,
    jpeg_quality: int,
    blur_sigma: float,
    low_light_gamma: float,
    haze_alpha: float,
    haze_add_noise: bool,
    scale: int = 4,
):
    """
    Processes HR images to create HQ and LQ datasets based on the specified degradation type.
    Optionally creates HQ_Blur for motion_blur degradation.
    """
    # Create base output directories
    hq_out_dir = os.path.join(output_dir, "hq")
    lq_out_dir = os.path.join(output_dir, "lq")
    os.makedirs(hq_out_dir, exist_ok=True)
    os.makedirs(lq_out_dir, exist_ok=True)

    # Create hq_blur directory only if needed
    hq_blur_out_dir = None
    if degradation_type == "motion_blur":
        hq_blur_out_dir = os.path.join(output_dir, "hq_blur")
        os.makedirs(hq_blur_out_dir, exist_ok=True)

    # Find images (assuming common extensions)
    hr_image_paths = (
        sorted(glob(os.path.join(hr_dir, "*.png")))
        + sorted(glob(os.path.join(hr_dir, "*.jpg")))
        + sorted(glob(os.path.join(hr_dir, "*.bmp")))
    )

    if not hr_image_paths:
        print(f"Error: No images found in {hr_dir}")
        return

    print(f"Found {len(hr_image_paths)} images in {hr_dir}. Processing...")

    for img_path in tqdm(hr_image_paths):
        try:
            base_name = os.path.basename(img_path)
            img_hr = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img_hr is None:
                print(f"Warning: Could not read {img_path}. Skipping.")
                continue

            # 1. Save original HR image (common to all types)
            cv2.imwrite(os.path.join(hq_out_dir, base_name), img_hr)

            # --- Apply specific degradation ---
            img_lr_final = None

            if degradation_type == "motion_blur":
                # 2a. Apply Motion Blur
                random_angle = random.uniform(0, 360)
                motion_kernel = get_motion_blur_kernel(kernel_size, random_angle)
                img_hr_blurred = cv2.filter2D(
                    img_hr, -1, motion_kernel
                )  # -1 means same depth as source
                img_hr_blurred = np.clip(img_hr_blurred, 0, 255).astype(
                    np.uint8
                )  # Ensure valid range

                # 3a. Save blurred HR image (specific to this type)
                if hq_blur_out_dir:  # Check if dir was created
                    cv2.imwrite(
                        os.path.join(hq_blur_out_dir, base_name), img_hr_blurred
                    )

                # 4a. Downsample blurred HR image
                h, w, _ = img_hr_blurred.shape
                img_lr = cv2.resize(
                    img_hr_blurred,
                    (w // scale, h // scale),
                    interpolation=cv2.INTER_CUBIC,
                )

                # 5a. Add Gaussian Noise
                noise = np.random.normal(0, noise_sigma, img_lr.shape).astype(
                    np.float32
                )
                img_lr_noisy = img_lr.astype(np.float32) + noise
                img_lr_noisy = np.clip(img_lr_noisy, 0, 255).astype(
                    np.uint8
                )  # Clip to valid range
                img_lr_final = img_lr_noisy

            elif degradation_type == "bicubic":
                # 2b. Downsample original HR image directly
                h, w, _ = img_hr.shape
                img_lr = cv2.resize(
                    img_hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC
                )
                img_lr_final = img_lr.astype(np.uint8)  # Ensure type

            elif degradation_type == "jpeg":
                # 2c. Downsample original HR image
                h, w, _ = img_hr.shape
                img_lr = cv2.resize(
                    img_hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC
                )
                # 3c. Apply severe JPEG compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                result, encoded_img = cv2.imencode(".jpg", img_lr, encode_param)
                if not result:
                    print(
                        f"Warning: Failed to encode image {img_path} as JPEG. Skipping."
                    )
                    continue
                # 4c. Decode back to introduce artifacts
                img_lr_artifacts = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                img_lr_final = img_lr_artifacts.astype(np.uint8)

            elif degradation_type == "combined":
                # 1d. Apply Gaussian Blur to HR
                # Determine kernel size based on sigma (common practice)
                ksize = int(math.ceil(blur_sigma * 3)) * 2 + 1
                img_hr_blurred = cv2.GaussianBlur(img_hr, (ksize, ksize), blur_sigma)
                img_hr_blurred = np.clip(img_hr_blurred, 0, 255).astype(np.uint8)

                # 2d. Downsample blurred HR image
                h, w, _ = img_hr_blurred.shape
                img_lr = cv2.resize(
                    img_hr_blurred,
                    (w // scale, h // scale),
                    interpolation=cv2.INTER_CUBIC,
                )

                # 3d. Add Gaussian Noise
                noise = np.random.normal(0, noise_sigma, img_lr.shape).astype(
                    np.float32
                )
                img_lr_noisy = img_lr.astype(np.float32) + noise
                img_lr_noisy = np.clip(img_lr_noisy, 0, 255).astype(np.uint8)

                # 4d. Apply JPEG compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                result, encoded_img = cv2.imencode(".jpg", img_lr_noisy, encode_param)
                if not result:
                    print(
                        f"Warning: Failed to encode image {img_path} with JPEG for combined degradation. Skipping."
                    )
                    continue
                img_lr_artifacts = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                img_lr_final = img_lr_artifacts.astype(np.uint8)

            elif degradation_type == "low_light":
                # 1e. Downsample original HR image
                h, w, _ = img_hr.shape
                img_lr = cv2.resize(
                    img_hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC
                )
                # 2e. Reduce Brightness
                img_lr_dark = img_lr.astype(np.float32) * low_light_gamma
                # 3e. Add Noise (will be relatively high compared to dark signal)
                noise = np.random.normal(0, noise_sigma, img_lr_dark.shape).astype(
                    np.float32
                )
                img_lr_noisy = img_lr_dark + noise
                # 4e. Clip to valid range
                img_lr_final = np.clip(img_lr_noisy, 0, 255).astype(np.uint8)

            elif degradation_type == "haze":
                # 1f. Downsample original HR image
                h, w, _ = img_hr.shape
                img_lr = cv2.resize(
                    img_hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC
                )
                # 2f. Apply Haze blending
                # Ensure float32 for calculation
                img_lr_float = img_lr.astype(np.float32)
                # Blend with white (255)
                img_hazy = img_lr_float * haze_alpha + (1.0 - haze_alpha) * 255.0
                # 3f. Optional Noise
                if haze_add_noise:
                    noise = np.random.normal(0, noise_sigma, img_hazy.shape).astype(
                        np.float32
                    )
                    img_hazy = img_hazy + noise
                # 4f. Clip
                img_lr_final = np.clip(img_hazy, 0, 255).astype(np.uint8)

            # Add more degradation types with elif degradation_type == '...' here
            # elif degradation_type == 'other_type':

            else:
                print(
                    f"Error: Unknown degradation type '{degradation_type}' for image {img_path}. Skipping."
                )
                continue

            # 6. Save final LR image (common to all types)
            if img_lr_final is not None:
                cv2.imwrite(os.path.join(lq_out_dir, base_name), img_lr_final)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Processing complete. Output saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare degraded datasets (LQ) and corresponding HQ images from a source HR directory using various degradation pipelines."
    )
    parser.add_argument(
        "--hr_dir",
        type=str,
        required=True,
        help="Directory containing the original High-Resolution images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory to save the output (hq/, lq/, and optionally hq_blur/ subdirs will be created).",
    )
    parser.add_argument(
        "--degradation_type",
        type=str,
        required=True,
        choices=[
            "motion_blur",
            "bicubic",
            "jpeg",
            "combined",
            "low_light",
            "haze",
        ],  # Added haze type
        help="Type of degradation pipeline to apply.",
    )
    # --- Degradation-specific arguments ---
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=15,
        help="Size of the motion blur kernel (will be forced odd). Used only if degradation_type='motion_blur'. Default: 15.",
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=10.0,
        help="Standard deviation of the Gaussian noise. Used by degradation_type='motion_blur', 'combined', 'low_light', and optionally 'haze'. Default: 10.0.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=20,
        help="JPEG quality factor (0-100, lower is more compressed). Used by degradation_type='jpeg' and 'combined'. Default: 20 (jpeg), maybe adjust for combined?",
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=1.0,
        help="Standard deviation of the Gaussian blur applied to HR image. Used only if degradation_type='combined'. Default: 1.0.",
    )
    parser.add_argument(
        "--low_light_gamma",
        type=float,
        default=0.2,
        help="Brightness reduction factor (0.0-1.0). Used only if degradation_type='low_light'. Default: 0.2",
    )
    parser.add_argument(
        "--haze_alpha",
        type=float,
        default=0.7,
        help="Alpha blending factor for haze (0.0=max haze, 1.0=no haze). Used only if degradation_type='haze'. Default: 0.7",
    )
    parser.add_argument(
        "--haze_add_noise",
        action="store_true",
        help="Add Gaussian noise after haze effect. Used only if degradation_type='haze'. Uses --noise_sigma.",
    )
    # parser.add_argument('--scale', type=int, default=4, help='Downscaling factor. Default: 4.') # Kept fixed at 4 for now

    args = parser.parse_args()

    # Optional: Add validation here if needed, e.g., check if kernel_size/noise_sigma make sense for the chosen type

    process_images(
        args.hr_dir,
        args.output_dir,
        args.degradation_type,
        args.kernel_size,
        args.noise_sigma,
        args.jpeg_quality,
        args.blur_sigma,
        args.low_light_gamma,
        args.haze_alpha,
        args.haze_add_noise,
    )  # Using fixed scale=4
