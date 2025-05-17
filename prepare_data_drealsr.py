import os
import shutil

# Define base directory relative to the script location
base_dir = "testdata/DRealSR"
src_hr_dir = os.path.join(base_dir, "Test_x4/test_HR")
src_lr_dir = os.path.join(base_dir, "Test_x4/test_LR")
tgt_hr_dir = os.path.join(base_dir, "gt")
tgt_lr_dir = os.path.join(base_dir, "lq")

# Create target directories if they don't exist
os.makedirs(tgt_hr_dir, exist_ok=True)
os.makedirs(tgt_lr_dir, exist_ok=True)

print(f"Source HR directory: {src_hr_dir}")
print(f"Source LR directory: {src_lr_dir}")
print(f"Target HR (gt) directory: {tgt_hr_dir}")
print(f"Target LR (lq) directory: {tgt_lr_dir}")

# Process files
copied_count = 0
skipped_count = 0

if not os.path.isdir(src_hr_dir):
    print(f"Error: Source HR directory not found: {src_hr_dir}")
    exit(1)
if not os.path.isdir(src_lr_dir):
    print(f"Error: Source LR directory not found: {src_lr_dir}")
    exit(1)

print("\nProcessing files...")

for hr_filename in os.listdir(src_hr_dir):
    if hr_filename.endswith("_x4.png") and os.path.isfile(
        os.path.join(src_hr_dir, hr_filename)
    ):
        base_name = hr_filename[: -len("_x4.png")]
        lr_filename = f"{base_name}_x1.png"
        tgt_filename = f"{base_name}.png"

        src_hr_path = os.path.join(src_hr_dir, hr_filename)
        src_lr_path = os.path.join(src_lr_dir, lr_filename)
        tgt_hr_path = os.path.join(tgt_hr_dir, tgt_filename)
        tgt_lr_path = os.path.join(tgt_lr_dir, tgt_filename)

        # Check if the corresponding LR file exists
        if os.path.isfile(src_lr_path):
            try:
                # Copy HR file
                shutil.copy2(src_hr_path, tgt_hr_path)
                # Copy LR file
                shutil.copy2(src_lr_path, tgt_lr_path)
                print(
                    f"  Copied: {hr_filename} -> gt/{tgt_filename} and {lr_filename} -> lq/{tgt_filename}"
                )
                copied_count += 1
            except Exception as e:
                print(f"  Error copying {hr_filename} or {lr_filename}: {e}")
                skipped_count += 1
        else:
            print(
                f"  Skipped: Corresponding LR file not found for {hr_filename} (Expected: {lr_filename})"
            )
            skipped_count += 1
    elif os.path.isfile(os.path.join(src_hr_dir, hr_filename)):
        print(
            f"  Skipped: File does not match expected HR pattern (*_x4.png): {hr_filename}"
        )
        skipped_count += 1


print(f"\nProcessing complete.")
print(f"  Files copied: {copied_count}")
print(f"  Files skipped: {skipped_count}")
