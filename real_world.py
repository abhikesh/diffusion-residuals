import argparse
import subprocess
import sys
from pathlib import Path
from loguru import logger
import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import cv2
from models.network_rrdbnet import RRDBNet
from models.network_swinir import SwinIR
from utils.bsrgan_utils_image import (
    imread_uint,
    single2tensor4,
    tensor2single,
    imsave,
)
from tqdm import tqdm

# Add project root to sys.path to allow importing from other directories if needed
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logger
logger.remove()
logger.add(
    sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO"
)

# --- Configuration ---
# Adjusted paths and arguments for the new standalone scripts
RESRSHIFT_SCRIPT = "inference_resshift.py"
RESRSHIFT_TASK = "realsr"
RESRSHIFT_VERSION = "v3"

REALESRGAN_SCRIPT = "inference_realesrgan.py"
REALESRGAN_MODEL_NAME = "RealESRGAN_x4plus"

BSRGAN_SCRIPT = "inference_bsrgan.py"
BSRGAN_MODEL_NAME = "BSRGAN"

METRICS_SCRIPT = "scripts/cal_metrics_superres.py"

# Models to evaluate - Updated structure
MODELS = {
    "ResShift": {
        "script": RESRSHIFT_SCRIPT,
        "args": [
            "--task",
            RESRSHIFT_TASK,
            "--version",
            RESRSHIFT_VERSION,
            # Input/Output added dynamically below
        ],
    },
    # "RealESRGAN": { ... }, # If you want to re-enable it
    "BSRGAN": {
        "script": BSRGAN_SCRIPT,  # Script not used, but kept for consistency
        "args": [
            "-n",
            BSRGAN_MODEL_NAME,
            # Input/Output added dynamically below
        ],
    },
    "SwinIR": {  # Added SwinIR entry
        "script": None,  # Indicates in-process execution
        "args": [],  # No external script args needed
    },
}


def run_command(command, log_prefix=""):
    """
    Helper function to run a shell command, stream stderr (for tqdm),
    and capture stdout.
    """
    logger.info(f"{log_prefix} Running command: {' '.join(command)}")
    stdout_lines = []
    stderr_lines = []  # Keep stderr lines for logging warnings later if needed

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            encoding="utf-8",
            errors="replace",  # Handle potential decoding errors
        )

        # Stream stderr directly to the console (for tqdm)
        if process.stderr:
            # Ensure stderr is not None before iterating
            for line in iter(process.stderr.readline, ""):
                sys.stderr.write(line)
                sys.stderr.flush()
                stderr_lines.append(line)  # Optionally collect stderr lines too

        # Capture stdout
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                stdout_lines.append(line)
                # Optional: print stdout live as well, if desired
                # sys.stdout.write(line)
                # sys.stdout.flush()

        # Wait for the process to finish and get return code
        process.wait()
        return_code = process.returncode

        # Log captured stdout for debugging
        if stdout_lines:
            logger.debug(f"{log_prefix} STDOUT:\n{''.join(stdout_lines)}")
        # Log captured stderr as warning if needed (optional, as it was already printed)
        # if stderr_lines:
        #      logger.warning(f"{log_prefix} STDERR (already streamed):\n{''.join(stderr_lines)}")

        if return_code != 0:
            # Include captured output in the exception message
            stderr_output = "".join(stderr_lines)
            stdout_output = "".join(stdout_lines)
            error_message = (
                f"Command '{' '.join(command)}' returned non-zero exit status {return_code}.\n"
                f"Stderr:\n{stderr_output}\n"
                f"Stdout:\n{stdout_output}"
            )
            raise subprocess.CalledProcessError(
                return_code, command, output=stdout_output, stderr=stderr_output
            )

        logger.info(f"{log_prefix} Command finished successfully.")
        return "".join(stdout_lines)  # Return collected stdout

    except FileNotFoundError:
        logger.error(f"{log_prefix} Command not found: {command[0]}")
        raise  # Re-raise the exception
    except Exception as e:
        logger.error(f"{log_prefix} Error running command {' '.join(command)}: {e}")
        raise  # Re-raise other exceptions


# --- SwinIR Specific Tiling Function ---
def run_swinir_tiled_inference(
    model, img_lq_tensor, tile_size=256, tile_overlap=32, scale=4, window_size=8
):
    """Runs SwinIR inference with tiling."""
    if tile_size is None:
        # test the image as a whole
        output = model(img_lq_tensor)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq_tensor.size()
        tile = min(tile_size, h, w)
        if tile % window_size != 0:
            logger.warning(
                f"Tile size {tile} is not a multiple of window_size {window_size}. Adjusting tile size."
            )
            tile = tile - (tile % window_size)
            logger.info(f"Adjusted tile size to {tile}")

        if tile <= 0:  # Check if tile size is valid after adjustment
            logger.error(
                f"Invalid tile size {tile} after adjustment based on window size {window_size}. Cannot proceed with tiling."
            )
            raise ValueError("Invalid tile size for SwinIR tiling.")

        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(
            b, c, h * sf, w * sf, dtype=img_lq_tensor.dtype, device=img_lq_tensor.device
        )
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq_tensor[
                    ..., h_idx : h_idx + tile, w_idx : w_idx + tile
                ]
                with torch.no_grad():
                    out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ].add_(out_patch)
                W[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ].add_(out_patch_mask)
        output = E.div_(W)

    return output


# --- End SwinIR Specific ---


def main():
    parser = argparse.ArgumentParser(
        description="Run real-world SR evaluation for multiple models."
    )
    # --- Command Line Arguments ---
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument("--lr_dir", type=str, required=True, help="Path to LR images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to GT images.")
    parser.add_argument(
        "--results_base_dir",
        type=str,
        default="./results_real_world",
        help="Base directory for results.",
    )
    # SwinIR specific arguments
    parser.add_argument(
        "--swinir_tile_size",
        type=int,
        default=256,
        help="Tile size for SwinIR inference (default: 256, 0 or None for no tiling).",
    )
    parser.add_argument(
        "--swinir_tile_overlap",
        type=int,
        default=32,
        help="Tile overlap for SwinIR inference (default: 32).",
    )
    # Control arguments
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip the inference step and only calculate/consolidate metrics.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",  # Accepts one or more arguments
        default=None,  # Default is to calculate all metrics supported by the script
        help="Specify which metrics to calculate (e.g., psnr ssim lpips). Assumes metrics script supports a '--metrics' flag.",
    )
    args = parser.parse_args()

    # --- Setup Paths and Device ---
    lr_dir = Path(args.lr_dir).resolve()
    # LR dir validation only needed if not skipping inference
    if not args.skip_inference and not lr_dir.is_dir():
        logger.error(f"LR directory not found: {lr_dir}")
        sys.exit(1)

    gt_dir = Path(args.gt_dir).resolve()
    if not gt_dir.is_dir():
        # GT dir is always needed for metrics
        logger.error(f"Ground Truth directory not found: {gt_dir}")
        sys.exit(1)

    results_dir = Path(args.results_base_dir).resolve() / args.dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be stored in: {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Run Inference (conditional) ---
    if not args.skip_inference:
        logger.info("--- Starting Inference Phase ---")
        img_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]

        # Add tqdm progress bar for models during inference
        model_inference_loop = tqdm(MODELS.items(), desc="Running Inference")
        for model_name, config in model_inference_loop:
            model_inference_loop.set_postfix_str(
                f"Model: {model_name}"
            )  # Update tqdm description
            # logger.info(f"Running inference for: {model_name}") # Can be removed if tqdm is enough
            output_path = results_dir / model_name
            output_path.mkdir(exist_ok=True)

            # Get list of images for the current dataset once
            img_paths = []
            for ext in img_extensions:
                img_paths.extend(list(lr_dir.glob(ext)))
            img_paths = sorted(img_paths)

            if not img_paths:
                logger.warning(
                    f"No images found in LR directory: {lr_dir} with extensions {img_extensions}. Skipping inference for all models."
                )
                break  # Exit the model loop if no images are found

            # --- BSRGAN Specific Inference Logic (In-Process) ---
            if model_name == "BSRGAN":
                model_path = project_root / "weights" / "BSRGAN.pth"
                sf = 4

                if not model_path.is_file():
                    logger.error(f"BSRGAN model not found at: {model_path}. Skipping.")
                    continue

                try:
                    # Load model
                    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
                    model.load_state_dict(
                        torch.load(model_path, map_location=device), strict=True
                    )
                    model.eval()
                    model = model.to(device)
                    logger.info(f"BSRGAN model loaded successfully from {model_path}")

                    # Add tqdm for image loop within BSRGAN inference
                    image_loop = tqdm(
                        img_paths, desc=f"{model_name} Images", leave=False
                    )
                    for img_path in image_loop:
                        # logger.debug(f"Processing image: {img_path.name}") # Can be removed
                        # 1. Read image (HWC, uint8, BGR)
                        img_lq = imread_uint(str(img_path), n_channels=3)
                        if img_lq is None:
                            logger.warning(
                                f"Failed to read image {img_path} for BSRGAN. Skipping."
                            )
                            continue
                        # 2. Preprocess (HWC, float32, [0,1]) -> (BCHW, float32, [0,1])
                        img_lq_tensor = single2tensor4(img_lq / 255.0).to(device)

                        # 3. Inference
                        with torch.no_grad():
                            output_tensor = model(img_lq_tensor)

                        # 4. Postprocess (BCHW, float32, [0,1]) -> (HWC, uint8, [0,255])
                        output_img = tensor2single(output_tensor)  # HWC, float32, [0,1]
                        output_img = (
                            (output_img.clip(0, 1) * 255.0).round().astype(np.uint8)
                        )  # HWC, uint8

                        # 5. Save image (imsave expects HWC BGR uint8)
                        save_path = output_path / img_path.name
                        imsave(output_img, str(save_path))
                        # logger.debug(f"Saved SR image to: {save_path}") # Can be removed

                    logger.info(
                        f"BSRGAN inference completed for {len(img_paths)} images."
                    )

                except Exception as e:
                    logger.error(f"Error during BSRGAN inference: {e}", exc_info=True)

            # --- SwinIR Specific Inference Logic (In-Process) ---
            elif model_name == "SwinIR":
                model_params = {
                    "upscale": 4,
                    "in_chans": 3,
                    "img_size": 48,
                    "window_size": 8,
                    "img_range": 1.0,
                    "depths": [6, 6, 6, 6, 6, 6],
                    "embed_dim": 180,
                    "num_heads": [6, 6, 6, 6, 6, 6],
                    "mlp_ratio": 2,
                    "upsampler": "pixelshuffle",
                    "resi_connection": "1conv",
                }
                weights_path = (
                    project_root
                    / "weights"
                    / "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
                )
                scale = model_params["upscale"]
                window_size = model_params["window_size"]
                tile_size = (
                    args.swinir_tile_size
                    if args.swinir_tile_size and args.swinir_tile_size > 0
                    else None
                )
                tile_overlap = args.swinir_tile_overlap

                if not weights_path.is_file():
                    logger.error(
                        f"SwinIR model not found at: {weights_path}. Skipping."
                    )
                    continue

                try:
                    # Load model
                    model = SwinIR(**model_params)
                    pretrained_model = torch.load(weights_path, map_location=device)

                    # Reverted loading logic
                    param_key = "params"
                    model.load_state_dict(
                        (
                            pretrained_model[param_key]
                            if param_key in pretrained_model
                            else pretrained_model
                        ),
                        strict=True,
                    )

                    model.eval()
                    model = model.to(device)
                    logger.info(f"SwinIR model loaded successfully from {weights_path}")
                    logger.info(
                        f"Using tile_size={tile_size}, tile_overlap={tile_overlap}"
                    )

                    # Add tqdm for image loop within SwinIR inference
                    image_loop = tqdm(
                        img_paths, desc=f"{model_name} Images", leave=False
                    )
                    for img_path in image_loop:
                        logger.debug(
                            f"--- Processing SwinIR for image: {img_path.name} ---"
                        )

                        # 1. Read image (HWC, BGR, uint8)
                        img_lq = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                        if img_lq is None:
                            logger.warning(
                                f"Failed to read image {img_path} for SwinIR. Skipping."
                            )
                            continue
                        logger.debug(
                            f"  Loaded LR image shape (HWC BGR): {img_lq.shape}"
                        )

                        # 2. Preprocess (HWC BGR uint8 -> CHW RGB float32 [0,1] -> BCHW)
                        img_lq = img_lq.astype(np.float32) / 255.0
                        img_lq = img_lq[:, :, [2, 1, 0]]  # BGR to RGB
                        img_lq = np.transpose(img_lq, (2, 0, 1))  # HWC to CHW
                        img_lq_tensor = (
                            torch.from_numpy(img_lq).float().unsqueeze(0).to(device)
                        )  # CHW to BCHW

                        # Remember original size
                        _, _, h_old, w_old = img_lq_tensor.size()
                        logger.debug(
                            f"  Original LR tensor shape (BCHW): {img_lq_tensor.size()} (h_old={h_old}, w_old={w_old})"
                        )

                        # 3. Pad input tensor to be divisible by window_size
                        h_pad = (window_size - h_old % window_size) % window_size
                        w_pad = (window_size - w_old % window_size) % window_size
                        img_lq_padded = F.pad(
                            img_lq_tensor, (0, w_pad, 0, h_pad), mode="reflect"
                        )
                        logger.debug(
                            f"  Padded LR tensor shape (BCHW): {img_lq_padded.size()}"
                        )

                        # 4. Inference (Tiled)
                        output_tensor_padded = run_swinir_tiled_inference(
                            model,
                            img_lq_padded,
                            tile_size,
                            tile_overlap,
                            scale,
                            window_size,
                        )
                        logger.debug(
                            f"  Raw tiled output shape (BCHW): {output_tensor_padded.size()}"
                        )

                        # 5. Crop output to original size
                        output_tensor = output_tensor_padded[
                            ..., : h_old * scale, : w_old * scale
                        ]
                        logger.debug(
                            f"  Cropped final output shape (BCHW): {output_tensor.size()}"
                        )

                        # 6. Postprocess (BCHW RGB float32 [0,1] -> HWC RGB uint8 [0,255])
                        output_img = (
                            output_tensor.data.squeeze()
                            .float()
                            .cpu()
                            .clamp_(0, 1)
                            .numpy()
                        )
                        if output_img.ndim == 3:
                            # Transpose CHW RGB -> HWC RGB directly
                            output_img = np.transpose(output_img, (1, 2, 0))
                        # ELSE: Handle grayscale case if needed, though SwinIR model is likely color
                        output_img = (output_img * 255.0).round().astype(np.uint8)
                        logger.debug(
                            f"  Final output image shape (HWC RGB): {output_img.shape}"
                        )

                        # 7. Save image
                        save_path = output_path / img_path.name
                        # Pass HWC RGB to imsave, which will convert it to BGR for cv2.imwrite
                        imsave(output_img, str(save_path))
                        # logger.debug(f"Saved SR image to: {save_path}") # Can be removed

                    logger.info(
                        f"SwinIR inference completed for {len(img_paths)} images."
                    )

                except Exception as e:
                    logger.error(f"Error during SwinIR inference: {e}", exc_info=True)

            # --- Original Subprocess Logic for other models ---
            elif config["script"] is not None:  # Check if a script path is provided
                # NOTE: Assumes python executable is in PATH
                cmd = [sys.executable, config["script"]]
                cmd.extend(config["args"])
                # Add input/output paths dynamically using consistent arguments
                # Assumes all scripts now use -i/--input and -o/--output
                cmd.extend(["-i", str(lr_dir), "-o", str(output_path)])
                cmd.extend(["--bs", "1"])  # Example: Add batch size if needed by script
                cmd.extend(["--chop_size", "256"])  # Example: Add chop size if needed

                try:
                    run_command(cmd, log_prefix=f"[{model_name} Inference]")
                except FileNotFoundError:
                    logger.error(
                        f"Script not found for {model_name}: {config['script']}. Skipping."
                    )
                    logger.error(f"Ensure {config['script']} exists and is executable.")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Inference failed for {model_name}. Error: {e}")
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred during {model_name} inference: {e}",
                        exc_info=True,
                    )
            # --- End of conditional logic ---
        logger.info("--- Inference Phase Complete ---")
    else:
        logger.info("--- Skipping Inference Phase (due to --skip_inference flag) ---")

    # --- Calculate Metrics ---
    logger.info("--- Starting Metrics Calculation Phase ---")

    # Add tqdm progress bar for models during metrics calculation
    model_metrics_loop = tqdm(MODELS.keys(), desc="Calculating Metrics")
    for model_name in model_metrics_loop:
        model_metrics_loop.set_postfix_str(
            f"Model: {model_name}"
        )  # Update tqdm description
        sr_output_dir = results_dir / model_name
        # Check if the SR output directory exists, even if inference was skipped
        if not sr_output_dir.is_dir():
            logger.warning(
                f"Output directory for {model_name} not found ({sr_output_dir}). Skipping metrics calculation."
            )
            continue
        # Optional: Check if directory is empty only if inference was skipped
        if args.skip_inference and not any(sr_output_dir.iterdir()):
            logger.warning(
                f"Output directory for {model_name} is empty ({sr_output_dir}). Skipping metrics."
            )
            continue

        # logger.info(f"Calculating metrics for: {model_name}") # Can be removed if tqdm is enough
        # Base command
        cmd = [
            sys.executable,
            METRICS_SCRIPT,
            "--sr_dir",
            str(sr_output_dir),
            "--gt_dir",
            str(gt_dir),
        ]
        # Add specific metrics if requested
        if args.metrics:
            cmd.append("--metrics")
            cmd.extend(args.metrics)
            # logger.info(f"Calculating only specified metrics: {args.metrics}") # Can be removed
        # else:
        # logger.info("Calculating all metrics supported by the metrics script.") # Can be removed

        try:
            # Run metrics script using the modified run_command to stream progress
            logger.info(
                f"Running metrics calculation for {model_name}..."
            )  # Log before starting the subprocess
            stdout = run_command(cmd, log_prefix=f"[{model_name} Metrics]")
            logger.info(
                f"Metrics calculation finished for {model_name}."
            )  # Log after completion

        except FileNotFoundError:
            logger.error(
                f"Metrics script not found: {METRICS_SCRIPT}. Skipping metrics for {model_name}."
            )
        except subprocess.CalledProcessError as e:
            # Error details are now included in the exception raised by run_command
            logger.error(
                f"Metrics calculation failed for {model_name}. See details above or in log file."
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during {model_name} metrics calculation: {e}",
                exc_info=True,
            )

    # --- Consolidate Metrics from Temporary Files ---
    logger.info("--- Starting Metrics Consolidation Phase ---")

    per_image_dfs = []
    mean_metrics_data = {}
    processed_models = list(MODELS.keys())  # Get models that were intended to run
    dataset_name = results_dir.name  # Get dataset name from results dir

    for model_name in processed_models:
        model_results_dir = results_dir / model_name
        # Define paths to temporary files potentially created by the metrics script
        tmp_per_image_csv = model_results_dir / "_tmp_metrics_per_image.csv"
        tmp_mean_csv = model_results_dir / "_tmp_metrics_mean.csv"

        # Check if these temp files exist before trying to read/delete
        # This handles cases where metrics script failed or didn't run for a model
        delete_tmp_files = False  # Set to True to delete temp files after consolidation

        # Consolidate per-image metrics
        if tmp_per_image_csv.is_file():
            try:
                df_image = pd.read_csv(tmp_per_image_csv)
                # Rename columns to include model name, except for image_name
                df_image = df_image.rename(
                    columns=lambda c: f"{model_name}_{c}" if c != "image_name" else c
                )
                per_image_dfs.append(df_image)
                logger.info(
                    f"Read per-image metrics for {model_name} from {tmp_per_image_csv}"
                )
                if delete_tmp_files:
                    try:
                        tmp_per_image_csv.unlink()
                        logger.debug(f"Deleted temporary file: {tmp_per_image_csv}")
                    except OSError as e:
                        logger.warning(
                            f"Could not delete temporary file {tmp_per_image_csv}: {e}"
                        )
            except Exception as e:
                logger.error(f"Failed to read or process {tmp_per_image_csv}: {e}")
        else:
            # Log only if the SR dir exists but the temp file doesn't (implies metrics step might have failed)
            if (results_dir / model_name).is_dir():
                logger.warning(
                    f"Temporary per-image metrics file not found for {model_name}: {tmp_per_image_csv}"
                )

        # Consolidate mean metrics
        if tmp_mean_csv.is_file():
            try:
                df_mean = pd.read_csv(tmp_mean_csv)
                mean_metrics_dict = df_mean.set_index("metric_name")["value"].to_dict()
                mean_metrics_data[model_name] = mean_metrics_dict
                logger.info(f"Read mean metrics for {model_name} from {tmp_mean_csv}")
                if delete_tmp_files:
                    try:
                        tmp_mean_csv.unlink()
                        logger.debug(f"Deleted temporary file: {tmp_mean_csv}")
                    except OSError as e:
                        logger.warning(
                            f"Could not delete temporary file {tmp_mean_csv}: {e}"
                        )
            except Exception as e:
                logger.error(f"Failed to read or process {tmp_mean_csv}: {e}")
        else:
            if (results_dir / model_name).is_dir():
                logger.warning(
                    f"Temporary mean metrics file not found for {model_name}: {tmp_mean_csv}"
                )

    # Merge per-image DataFrames and save
    if per_image_dfs:
        try:
            # Start with the first DataFrame
            consolidated_per_image_df = per_image_dfs[0]
            # Merge subsequent DataFrames one by one
            for i in range(1, len(per_image_dfs)):
                consolidated_per_image_df = pd.merge(
                    consolidated_per_image_df,
                    per_image_dfs[i],
                    on="image_name",
                    how="outer",  # Use outer merge to keep all images even if one model failed
                )

            # Add dataset name column
            consolidated_per_image_df.insert(0, "dataset_name", dataset_name)

            # Define output path and save
            consolidated_per_image_csv = (
                results_dir / "consolidated_metrics_per_image.csv"
            )
            consolidated_per_image_df.to_csv(
                consolidated_per_image_csv, index=False, float_format="%.4f"
            )
            logger.info(
                f"Consolidated per-image metrics saved to: {consolidated_per_image_csv}"
            )
        except Exception as e:
            logger.error(f"Failed to save consolidated per-image metrics: {e}")
    else:
        logger.warning("No per-image metrics data found to consolidate.")

    # Create mean metrics DataFrame and save
    if mean_metrics_data:
        try:
            consolidated_mean_df = pd.DataFrame(mean_metrics_data)
            consolidated_mean_df.index.name = "metric_name"
            consolidated_mean_df = consolidated_mean_df.reset_index()

            # Add dataset name column
            consolidated_mean_df.insert(0, "dataset_name", dataset_name)

            # Define output path and save
            consolidated_mean_csv = results_dir / "consolidated_metrics_mean.csv"
            consolidated_mean_df.to_csv(
                consolidated_mean_csv, index=False, float_format="%.4f"
            )
            logger.info(f"Consolidated mean metrics saved to: {consolidated_mean_csv}")
        except Exception as e:
            logger.error(f"Failed to save consolidated mean metrics: {e}")
    else:
        logger.warning("No mean metrics data found to consolidate.")

    logger.info(f"--- Evaluation Complete for dataset {args.dataset_name} ---")
    logger.info(f"Results stored in: {results_dir}")
    logger.info(
        "Check consolidated CSV files for metrics and visual results in model subdirectories (if generated)."
    )


if __name__ == "__main__":
    main()

"""
Example Usage:

# Run full pipeline (Inference + All Metrics)
uv run python real_world.py \\
    --dataset_name YourDatasetName \\
    --lr_dir /path/to/YourDatasetName/lq \\
    --gt_dir /path/to/YourDatasetName/gt \\
    --results_base_dir ./results_eval

# Run only PSNR and SSIM metrics (assuming SR images already exist)
uv run python real_world.py \\
    --dataset_name YourDatasetName \\
    --lr_dir /path/to/YourDatasetName/lq \\
    --gt_dir /path/to/YourDatasetName/gt \\
    --results_base_dir ./results_eval \\
    --skip_inference \\
    --metrics psnr ssim

Note:
- The '--metrics' flag assumes your script 'scripts/cal_metrics_superres.py'
  accepts a '--metrics' argument followed by metric names (e.g., psnr ssim).
  Adjust the command construction in the 'Calculate Metrics' section if needed.
- Ensure model weights are present if running inference.
- Ensure SR results folders exist if using '--skip_inference'.
- Progress bars from cal_metrics_superres.py should now be visible.
"""
