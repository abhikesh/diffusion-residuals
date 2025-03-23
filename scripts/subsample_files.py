#!/usr/bin/env python3
"""
Subsample Files

This script randomly selects and copies a specified number of files from
an input directory to an output directory. It specifically handles a structure with
'gt', 'hq', 'lq', and 'mask' subdirectories, ensuring that corresponding files
with the same name are copied together.

Usage:
    uv run scripts/subsample_files.py [input_path] [output_path] [num_files]
"""

import argparse
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Randomly subsample files from one directory to another"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the directory containing the files to sample from"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the directory where sampled files will be copied to"
    )
    parser.add_argument(
        "num_files",
        type=int,
        help="Number of files to randomly select and copy"
    )
    return parser.parse_args()

def get_subdirectory_files(input_path: str) -> Dict[str, Dict[str, str]]:
    """
    Get files from all subdirectories, organized by filename.

    Args:
        input_path: Path to the directory containing the subdirectories

    Returns:
        Dictionary mapping filenames to subdirectory paths
    """
    input_dir = Path(input_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_path} does not exist")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_path} is not a directory")

    # Expected subdirectories
    subdirs = ['gt', 'hq', 'lq', 'mask']

    # Maps filename -> {subdir -> full_path}
    file_map = defaultdict(dict)

    # Track all unique files across all subdirectories
    all_filenames = set()

    # Check which subdirectories exist and get their files
    for subdir in subdirs:
        subdir_path = input_dir / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            logger.info(f"Found subdirectory: {subdir}")
            for file_path in subdir_path.iterdir():
                if file_path.is_file():
                    file_map[file_path.name][subdir] = str(file_path)
                    all_filenames.add(file_path.name)

    logger.info(f"Found {len(all_filenames)} unique filenames across all subdirectories")
    return file_map

def get_unique_filenames(file_map: Dict[str, Dict[str, str]]) -> Set[str]:
    """
    Get the set of unique filenames across all subdirectories.

    Args:
        file_map: Mapping of filenames to subdirectory paths

    Returns:
        Set of unique filenames
    """
    return set(file_map.keys())

def subsample_files(
    input_path: str,
    output_path: str,
    num_files: int
) -> Optional[List[str]]:
    """
    Randomly select files and copy them while maintaining the relationship
    between files with the same name across all subdirectories.

    Args:
        input_path: Directory containing the subdirectories
        output_path: Directory to copy selected files to
        num_files: Number of files to select

    Returns:
        List of copied file paths, or None if operation failed
    """
    try:
        # Ensure output directory exists
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        # Get files from subdirectories
        file_map = get_subdirectory_files(input_path)

        if not file_map:
            logger.warning(f"No files found in subdirectories of {input_path}")
            return None

        # Get unique filenames
        unique_filenames = list(get_unique_filenames(file_map))
        logger.info(f"Found {len(unique_filenames)} unique files across subdirectories")

        # Check if we're requesting more files than available
        if num_files > len(unique_filenames):
            logger.warning(
                f"Requested {num_files} files but only {len(unique_filenames)} unique files available. "
                f"Using all available files."
            )
            selected_filenames = unique_filenames
        else:
            # Randomly select filenames
            selected_filenames = random.sample(unique_filenames, num_files)

        logger.info(f"Selected {len(selected_filenames)} filenames for copying")

        # Create necessary subdirectories in output path
        subdirs = ['gt', 'hq', 'lq', 'mask']
        for subdir in subdirs:
            subdir_path = output_dir / subdir
            subdir_path.mkdir(exist_ok=True)

        # Copy selected files
        copied_files = []
        for filename in selected_filenames:
            logger.info(f"Processing file: {filename}")

            # For each filename, check all possible subdirectories
            for subdir in subdirs:
                # Check if this file exists in this subdirectory
                if subdir in file_map[filename]:
                    src_path = file_map[filename][subdir]
                    dst_path = str(output_dir / subdir / filename)

                    try:
                        # Copy the file
                        shutil.copy2(src_path, dst_path)
                        copied_files.append(dst_path)
                        logger.info(f"Copied: {src_path} -> {dst_path}")
                    except (IOError, OSError) as e:
                        logger.error(f"Error copying {src_path} to {dst_path}: {str(e)}")
                else:
                    logger.info(f"File {filename} does not exist in {subdir} directory")

        return copied_files

    except Exception as e:
        logger.error(f"Error during subsampling: {str(e)}", exc_info=True)
        return None

def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()

    logger.info(f"Starting subsample from {args.input_path} to {args.output_path}")
    logger.info(f"Will select {args.num_files} files maintaining the relationship between subdirectories")

    copied_files = subsample_files(
        args.input_path,
        args.output_path,
        args.num_files
    )

    if copied_files:
        logger.info(f"Successfully copied {len(copied_files)} files")
        unique_files = len(set([os.path.basename(f) for f in copied_files]))
        logger.info(f"Representing {unique_files} unique files across subdirectories")
    else:
        logger.error("Failed to copy files")

if __name__ == "__main__":
    main()
