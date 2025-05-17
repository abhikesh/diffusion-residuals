# prepare_data_realsr.py
import os
import shutil
import glob

# --- Configuration ---
INPUT_BASE_DIR = "testdata/RealSR_V3"
OUTPUT_BASE_DIR = "testdata/RealSR_V3_processed_x4"  # New directory for processed data
SOURCES = ["Canon", "Nikon"]
TARGET_SUBDIR = "Test/4"  # Specific subdir containing x4 scale images
# --- End Configuration ---


def process_realsr_data(input_base, output_base, sources, target_subdir):
    """
    Processes RealSR dataset: finds LR/HR pairs in the specified target_subdir
    (e.g., Test/4/), renames, and copies them into gt/ and lq/ subdirectories.
    """
    output_gt_dir = os.path.join(output_base, "gt")
    output_lq_dir = os.path.join(output_base, "lq")

    print(f"Creating output directories...")
    os.makedirs(output_gt_dir, exist_ok=True)
    os.makedirs(output_lq_dir, exist_ok=True)
    print(f" -> GT directory: {output_gt_dir}")
    print(f" -> LQ directory: {output_lq_dir}")

    copied_count = 0
    skipped_count = 0
    error_count = 0

    for source_name in sources:
        # Construct the path to the specific subdirectory (e.g., testdata/RealSR_V3/Canon/Test/4)
        source_target_dir = os.path.join(input_base, source_name, target_subdir)

        if not os.path.isdir(source_target_dir):
            print(
                f"\nWarning: Target directory not found: {source_target_dir}. Skipping source '{source_name}'."
            )
            continue

        print(f"\nProcessing target directory: {source_target_dir}...")

        # Find all LR files in the target subdirectory
        lr_pattern = os.path.join(source_target_dir, f"*_LR4.png")  # Look for _LR.png
        lr_files = sorted(glob.glob(lr_pattern))

        if not lr_files:
            print(f"  No '_LR4.png' files found in {source_target_dir}.")
            continue

        print(f"  Found {len(lr_files)} potential '_LR4.png' files.")

        for lr_filepath in lr_files:
            lr_filename = os.path.basename(lr_filepath)

            # Extract base name (e.g., '001', '080') by removing _LR.png
            try:
                base_name = lr_filename.replace("_LR4.png", "")
                if not base_name or base_name == lr_filename:
                    print(
                        f"  Warning: Could not extract base name from {lr_filename}. Skipping."
                    )
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"  Error parsing base name from {lr_filename}: {e}. Skipping.")
                skipped_count += 1
                continue

            # Construct expected HR filename and path (removing scale suffix from pattern)
            hr_filename = f"{base_name}_HR.png"
            hr_filepath = os.path.join(
                source_target_dir, hr_filename
            )  # Look in the same subdir

            # Check if the corresponding HR file exists
            if not os.path.isfile(hr_filepath):
                print(
                    f"  Warning: Missing HR pair for {lr_filename} (expected {hr_filename}). Skipping pair."
                )
                skipped_count += 1
                continue

            # Construct the new output filename (prefixing with source)
            output_filename = f"{source_name}_{base_name}.png"
            output_lq_path = os.path.join(output_lq_dir, output_filename)
            output_gt_path = os.path.join(output_gt_dir, output_filename)

            # Copy the files
            try:
                shutil.copy2(lr_filepath, output_lq_path)
                shutil.copy2(hr_filepath, output_gt_path)
                copied_count += 1
            except Exception as e:
                print(
                    f"  Error copying pair for base '{base_name}' from {source_name}: {e}"
                )
                error_count += 1
                if os.path.exists(output_lq_path):
                    os.remove(output_lq_path)
                if os.path.exists(output_gt_path):
                    os.remove(output_gt_path)

    print("\n--- Processing Summary ---")
    print(f"Successfully copied {copied_count} LR/HR pairs.")
    print(
        f"Skipped {skipped_count} files/pairs (missing counterparts or parsing errors)."
    )
    print(f"Encountered {error_count} errors during file copying.")
    print(f"Processed data located in: {output_base}")
    print("--------------------------")


if __name__ == "__main__":
    process_realsr_data(INPUT_BASE_DIR, OUTPUT_BASE_DIR, SOURCES, TARGET_SUBDIR)
