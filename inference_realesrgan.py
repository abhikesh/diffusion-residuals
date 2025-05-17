# inference_realesrgan.py
import argparse
import cv2
import glob
import os
import torch
from pathlib import Path
from loguru import logger
import sys
import numpy as np

# Try to import basicsr components
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer
    from basicsr.utils.download_util import load_file_from_url
except ImportError:
    logger.error("*" * 60)
    logger.error("Could not import RRDBNet or RealESRGANer from basicsr.")
    logger.error("Please ensure 'basicsr' is installed correctly.")
    logger.error("You can try: pip install basicsr")
    logger.error("Alternatively, if using as a submodule, ensure it's initialized:")
    logger.error("  git submodule update --init --recursive")
    logger.error("*" * 60)
    sys.exit(1)

# Default model URLs for common Real-ESRGAN models
MODEL_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}
DEFAULT_MODEL_NAME = "RealESRGAN_x4plus"


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(
        description="Real-ESRGAN inference", **parser_kwargs
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input image or folder"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results/RealESRGAN", help="Output folder"
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
    parser.add_argument(
        "-s",
        "--outscale",
        type=float,
        default=None,  # Default is derived from model name/scale
        help="The final upsampling scale of the image. Default: model scale",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="out",
        help="Suffix of the restored image. Default: out",
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=0,
        help="Tile size, 0 for no tile during testing. Default: 0",
    )
    parser.add_argument(
        "--tile_pad", type=int, default=10, help="Tile padding. Default: 10"
    )
    parser.add_argument(
        "--pre_pad",
        type=int,
        default=0,
        help="Pre padding size at each border. Default: 0",
    )
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    parser.add_argument(
        "--alpha_upsampler",
        type=str,
        default="realesrgan",
        help="The upsampler for the alpha channels. Options: realesrgan | bicubic. Default: realesrgan",
    )
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
    if "x2" in args.model_name:
        args.scale = 2
    elif "x4" in args.model_name:
        args.scale = 4
    else:
        logger.warning(
            f"Could not determine scale from model name '{args.model_name}'. Assuming scale 4."
        )
        args.scale = 4  # Default assumption

    # Set default outscale if not provided
    if args.outscale is None:
        args.outscale = args.scale
        logger.info(f"Setting output scale to model scale: {args.outscale}")
    elif args.outscale != args.scale:
        logger.warning(
            f"Output scale ({args.outscale}) differs from model scale ({args.scale})."
        )

    return args


def main():
    args = get_parser()
    os.makedirs(args.output, exist_ok=True)

    # Determine device
    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info(f"Using GPU: {args.gpu_id}")
    else:
        # Default to cuda:0 if available, else cpu
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

    # Set up the RealESRGANer
    # Note: RealESRGANer handles model loading internally if given model=None,
    # but requires architecture definition to be available for state_dict loading.
    # It's cleaner to instantiate the model first based on expected architecture.
    try:
        # Instantiate the correct RRDBNet architecture based on model name
        if args.model_name == "RealESRGAN_x4plus_anime_6B":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            )
        elif (
            args.model_name == "RealESRGAN_x4plus"
            or args.model_name == "RealESRNet_x4plus"
        ):
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
        elif args.model_name == "RealESRGAN_x2plus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
        else:
            # Fallback or error if model name doesn't match known configs
            logger.warning(
                f"Using default RRDBNet config (num_block=23, scale={args.scale}) for unknown model '{args.model_name}'."
            )
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=args.scale,
            )

        # Initialize RealESRGANer
        upsampler = RealESRGANer(
            scale=args.scale,
            model_path=model_path,  # Provide path to downloaded/local weights
            model=model,  # Provide instantiated model structure
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=args.half,
            device=device,
            gpu_id=args.gpu_id,  # Pass gpu_id if specified
        )
    except Exception as e:
        logger.error(f"Failed to initialize RealESRGANer: {e}")
        logger.error("Ensure 'basicsr' is installed and the model path is correct.")
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
        img_name = Path(img_path).stem  # Use stem to remove extension
        logger.info(f"[{idx+1}/{len(paths)}] Processing: {img_name}")

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Could not read image: {img_path}. Skipping.")
            continue

        # Enhance image
        try:
            output, img_mode = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                logger.error(f"CUDA out of memory for {img_name}: {error}")
                logger.warning(
                    "Try reducing tile size with --tile or disabling tiling with --tile 0."
                )
            else:
                logger.error(
                    f"Runtime error during enhancement for {img_name}: {error}"
                )
            continue  # Skip this image on error
        except Exception as error:
            logger.error(f"An unexpected error occurred for {img_name}: {error}")
            continue  # Skip this image on error

        # Determine save path and extension
        extension = args.ext
        if extension == "auto":
            extension = Path(img_path).suffix[1:]  # Get extension without dot
            if not extension:  # Handle cases with no extension
                extension = "png"
                logger.warning(f"No extension found for {img_name}, saving as png.")
        else:
            extension = extension.lower()
            if extension not in ["png", "jpg", "jpeg"]:
                logger.warning(f"Unsupported extension {extension}, defaulting to png.")
                extension = "png"

        if (
            img_mode == "RGBA" and extension != "png"
        ):  # RGBA images should be saved in png format
            logger.warning(
                f"Input image {img_name} has alpha channel, forcing output extension to png."
            )
            extension = "png"

        save_path = os.path.join(args.output, f"{img_name}_{args.suffix}.{extension}")

        # Save image
        try:
            cv2.imwrite(save_path, output)
            logger.info(f"Saved: {save_path}")
        except Exception as error:
            logger.error(f"Could not save image {save_path}: {error}")

    logger.info(f"--- Real-ESRGAN Inference Complete for {args.input} ---")


if __name__ == "__main__":
    # Setup logger for standalone execution
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
