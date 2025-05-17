# inference_bsrgan.py
import argparse
import cv2
import glob
import os
import torch
import numpy as np
from pathlib import Path
from loguru import logger
import sys

# Try to import basicsr components
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from basicsr.utils import (
        img2tensor,
        tensor2img,
    )  # Assuming these utils exist and are suitable
except ImportError:
    logger.error("*" * 60)
    logger.error("Could not import necessary components from basicsr (RRDBNet, utils).")
    logger.error("Please ensure 'basicsr' is installed correctly.")
    logger.error("You can try: pip install basicsr")
    logger.error("Alternatively, if using as a submodule, ensure it's initialized:")
    logger.error("  git submodule update --init --recursive")
    logger.error("*" * 60)
    sys.exit(1)

# Default model URLs for common BSRGAN models
MODEL_URLS = {
    "BSRGAN": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth",
    "BSRGANx2": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGANx2.pth",
}
DEFAULT_MODEL_NAME = "BSRGAN"


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(description="BSRGAN inference", **parser_kwargs)
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input image or folder"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results/BSRGAN", help="Output folder"
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name. Options: {list(MODEL_URLS.keys())}. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Explicit path to the model file. Overrides --model_name if provided.",
    )
    # BSRGAN typically doesn't need outscale separate from model scale
    # parser.add_argument(
    #     '-s',
    #     '--outscale',
    #     type=float,
    #     default=None, # Default is derived from model name/scale
    #     help='The final upsampling scale of the image. Default: model scale')
    parser.add_argument(
        "--suffix",
        type=str,
        default="out",
        help="Suffix of the restored image. Default: out",
    )
    # BSRGAN standard inference often doesn't use tiling/padding like RealESRGANer
    # parser.add_argument(
    #     '-t',
    #     '--tile',
    #     type=int,
    #     default=0,
    #     help='Tile size, 0 for no tile during testing. Default: 0')
    # parser.add_argument(
    #     '--tile_pad',
    #     type=int,
    #     default=10,
    #     help='Tile padding. Default: 10')
    # parser.add_argument(
    #     '--pre_pad',
    #     type=int,
    #     default=0,
    #     help='Pre padding size at each border. Default: 0')
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    parser.add_argument(
        "--ext",
        type=str,
        default="auto",
        help="Image extension. Options: auto | jpg | png. auto means using the same extension as inputs. Default: auto",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=None,
        help="GPU device ID to use (default=None uses cuda:0 if available, otherwise cpu)",
    )

    args = parser.parse_args()

    # Determine model path and scale
    if args.model_path is None:
        if args.model_name not in MODEL_URLS:
            logger.error(
                f"Model name '{args.model_name}' not recognized. Available: {list(MODEL_URLS.keys())}"
            )
            sys.exit(1)
        args.model_path = MODEL_URLS[args.model_name]  # Use the URL
    else:
        # If model_path is provided, try to infer model_name for scale detection
        for name, url in MODEL_URLS.items():
            if Path(url).name == Path(args.model_path).name:
                args.model_name = name
                logger.info(f"Inferred model name '{args.model_name}' from path.")
                break
        if (
            args.model_name == DEFAULT_MODEL_NAME
            and Path(args.model_path).name != Path(MODEL_URLS[DEFAULT_MODEL_NAME]).name
        ):
            logger.warning(
                f"Could not infer model name from path '{args.model_path}'. Assuming default '{DEFAULT_MODEL_NAME}'. Scale detection might be incorrect."
            )

    # Determine scale from model name
    if "x2" in args.model_name.lower():  # Use lower() for robustness
        args.scale = 2
    elif (
        "x4" in args.model_name.lower() or args.model_name == "BSRGAN"
    ):  # BSRGAN default is x4
        args.scale = 4
    else:
        logger.warning(
            f"Could not determine scale from model name '{args.model_name}'. Assuming scale 4."
        )
        args.scale = 4  # Default assumption

    logger.info(f"Using model scale: {args.scale}")
    args.outscale = args.scale  # BSRGAN output scale matches model scale

    return args


def main():
    args = get_parser()
    os.makedirs(args.output, exist_ok=True)

    # Determine device
    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, running on CPU. This will be slow.")

    # Download model if path is a URL
    model_path = args.model_path
    if model_path.startswith("https://"):
        model_basename = Path(model_path).name
        model_path = load_file_from_url(
            url=model_path,
            model_dir=os.path.join("weights"),
            progress=True,
            file_name=model_basename,
        )
        logger.info(f"Downloaded model to: {model_path}")

    # --- Set up BSRGAN model ---
    try:
        # Instantiate the RRDBNet model (commonly used for BSRGAN)
        # BSRGAN default uses num_block=23
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=args.scale,
        )

        # Load state dict
        loadnet = torch.load(model_path, map_location=torch.device("cpu"))
        # prefer to use params_ema
        if "params_ema" in loadnet:
            keyname = "params_ema"
        elif "params" in loadnet:
            keyname = "params"
        else:
            keyname = None

        if keyname:
            logger.info(f"Loading state dict using key: {keyname}")
            model.load_state_dict(loadnet[keyname], strict=True)
        else:
            logger.info(
                "Loading state dict directly (no 'params' or 'params_ema' key found)."
            )
            model.load_state_dict(loadnet, strict=True)

        model.eval()
        model = model.to(device)
        if args.half:
            logger.info("Using half precision (FP16).")
            model = model.half()

    except ImportError as e:
        logger.error(f"ImportError: {e}. Could not import RRDBNet.")
        logger.error("Ensure 'basicsr' is installed correctly.")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load BSRGAN model: {e}")
        sys.exit(1)

    # --- Inference ---
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, "*")))
        if not paths:
            logger.error(f"No image files found in directory: {args.input}")
            sys.exit(1)

    logger.info(f"Processing {len(paths)} images...")
    for idx, img_path in enumerate(paths):
        img_name = Path(img_path).stem
        logger.info(f"[{idx+1}/{len(paths)}] Processing: {img_name}")

        # Read image HWC, BGR, uint8
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Could not read image: {img_path}. Skipping.")
            continue

        # Prepare image: HWC -> CHW, BGR -> RGB, uint8 -> float32 (0-1), add batch dim
        if img.dtype != np.uint8:
            logger.warning(f"Image {img_name} is not uint8 ({img.dtype}), converting.")
            # Basic handling for other types, might need adjustment
            if np.max(img) > 256 and img.dtype == np.uint16:
                img = (img / 65535.0).astype(np.float32)
            else:  # Assume float range 0-1 or normalize 0-255
                img = img.astype(np.float32)
                if np.max(img) > 1.0:
                    img = img / 255.0
            img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        img_lq = img.astype(np.float32) / 255.0
        if len(img_lq.shape) == 2:  # Handle grayscale
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_GRAY2RGB)
            logger.info(f"Converted grayscale image {img_name} to RGB.")

        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 3 else img_lq[:, :, :3], (2, 0, 1)
        )  # HWC => CHW RGB? (cv2 reads BGR)
        img_lq = np.ascontiguousarray(
            img_lq[::-1, :, :]
        )  # BGR to RGB -> Reverse channel order
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)
        if args.half:
            img_lq = img_lq.half()

        # Inference
        try:
            with torch.no_grad():
                output_tensor = model(img_lq)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                logger.error(f"CUDA out of memory for {img_name}: {error}")
                logger.warning(
                    "BSRGAN script does not support tiling by default. Try on a GPU with more memory."
                )
            else:
                logger.error(
                    f"Runtime error during enhancement for {img_name}: {error}"
                )
            continue  # Skip this image on error
        except Exception as error:
            logger.error(f"An unexpected error occurred for {img_name}: {error}")
            continue  # Skip this image on error

        # Post-process: tensor (0-1) -> numpy (0-255), CHW -> HWC, RGB -> BGR
        output = tensor2img(
            output_tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)
        )

        # Determine save path and extension
        extension = args.ext
        if extension == "auto":
            extension = Path(img_path).suffix[1:]
            if not extension:
                extension = "png"
                logger.warning(f"No extension found for {img_name}, saving as png.")
        else:
            extension = extension.lower()
            if extension not in ["png", "jpg", "jpeg"]:
                logger.warning(f"Unsupported extension {extension}, defaulting to png.")
                extension = "png"

        save_path = os.path.join(args.output, f"{img_name}_{args.suffix}.{extension}")

        # Save image
        try:
            cv2.imwrite(save_path, output)
            logger.info(f"Saved: {save_path}")
        except Exception as error:
            logger.error(f"Could not save image {save_path}: {error}")

    logger.info(f"--- BSRGAN Inference Complete for {args.input} ---")


if __name__ == "__main__":
    # Setup logger for standalone execution
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
