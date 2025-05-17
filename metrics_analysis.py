# metrics_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import os
import re

# --- Configuration ---
CSV_FILE_PATH = "results/RealSR_V3/consolidated_metrics_per_image.csv"
GT_DIR = "testdata/RealSR_V3_processed_x4/gt"
BSRGAN_DIR = "results/RealSR_V3/BSRGAN"
RESHIFT_DIR = "results/RealSR_V3/ResShift"
# PDF Filenames will be generated dynamically based on comparison
TOP_N = 20
MODEL1_PREFIX = "BSRGAN"
MODEL2_PREFIX = "ResShift"
# --- End Configuration ---


def get_metric_pairs(df, prefix1, prefix2):
    """Identifies pairs of metric columns for two model prefixes."""
    metric_pairs = []
    prefix1_cols = sorted([col for col in df.columns if col.startswith(f"{prefix1}_")])
    prefix2_cols_set = set(col for col in df.columns if col.startswith(f"{prefix2}_"))

    for p1_col in prefix1_cols:
        try:
            metric_name = p1_col.split(f"{prefix1}_", 1)[1]
        except IndexError:
            continue  # Skip if column name doesn't match expected format
        p2_col = f"{prefix2}_{metric_name}"
        if p2_col in prefix2_cols_set:
            if pd.api.types.is_numeric_dtype(
                df[p1_col]
            ) and pd.api.types.is_numeric_dtype(df[p2_col]):
                metric_pairs.append(
                    {"prefix1_col": p1_col, "prefix2_col": p2_col, "name": metric_name}
                )
            else:
                print(f"Warning: Skipping non-numeric metric pair: {p1_col}, {p2_col}")
    return metric_pairs


def find_top_performing_images(
    df, metric_pairs, top_n_per_metric, primary_prefix, secondary_prefix
):
    """
    Finds top N images across metrics where primary_prefix outperforms secondary_prefix.

    Handles lower-is-better metrics (containing 'lpips').

    Returns:
        dict: Dictionary mapping image_name to list of selection reasons.
              Returns empty dict if no images found or on error.
    """
    image_selection_reasons = {}
    top_images_set = set()
    print(
        f"\nFinding top {top_n_per_metric} images per metric where {primary_prefix} outperforms {secondary_prefix}..."
    )

    for metric in metric_pairs:
        # Ensure the columns match the primary/secondary roles for this run
        if metric["prefix1_col"].startswith(primary_prefix):
            primary_col, secondary_col = metric["prefix1_col"], metric["prefix2_col"]
        elif metric["prefix2_col"].startswith(primary_prefix):
            primary_col, secondary_col = metric["prefix2_col"], metric["prefix1_col"]
        else:
            # This shouldn't happen if get_metric_pairs worked correctly
            print(
                f"Warning: Could not assign primary/secondary roles for metric {metric['name']}"
            )
            continue

        metric_name = metric["name"]
        is_lower_better = "lpips" in metric_name.lower()

        df_analysis = df[[primary_col, secondary_col, "image_name"]].copy()

        if is_lower_better:
            # Primary outperforms if its score is lower
            df_filtered = df_analysis[
                df_analysis[primary_col] < df_analysis[secondary_col]
            ].copy()
            if df_filtered.empty:
                print(
                    f" - No images where {primary_prefix} < {secondary_prefix} for metric '{metric_name}'."
                )
                continue
            # Difference = improvement (Secondary - Primary), positive means Primary is lower
            df_filtered["difference"] = (
                df_filtered[secondary_col] - df_filtered[primary_col]
            )
            print(
                f" - Analyzing '{metric_name}' ({primary_prefix} < {secondary_prefix})..."
            )
        else:
            # Primary outperforms if its score is higher
            df_filtered = df_analysis[
                df_analysis[primary_col] > df_analysis[secondary_col]
            ].copy()
            if df_filtered.empty:
                print(
                    f" - No images where {primary_prefix} > {secondary_prefix} for metric '{metric_name}'."
                )
                continue
            # Difference = improvement (Primary - Secondary), positive means Primary is higher
            df_filtered["difference"] = (
                df_filtered[primary_col] - df_filtered[secondary_col]
            )
            print(
                f" - Analyzing '{metric_name}' ({primary_prefix} > {secondary_prefix})..."
            )

        # Sort by difference (descending - higher difference means primary model is better)
        df_sorted = df_filtered.sort_values(by="difference", ascending=False)
        df_top_for_metric = df_sorted.head(top_n_per_metric)

        count_before = len(top_images_set)
        current_metric_images = set()
        for index, row in df_top_for_metric.iterrows():
            img_name = row["image_name"]
            # Fetch original values using the correct column names from the original df
            # Need to be careful here, the column names depend on which prefix is primary
            primary_val = df.loc[df["image_name"] == img_name, primary_col].iloc[0]
            secondary_val = df.loc[df["image_name"] == img_name, secondary_col].iloc[0]

            reason = {
                "metric_name": metric_name,
                "difference": row[
                    "difference"
                ],  # Represents how much primary outperformed secondary
                f"{primary_prefix}_val": primary_val,
                f"{secondary_prefix}_val": secondary_val,
                # Store which model was considered primary for this reason
                "primary_model": primary_prefix,
            }
            if img_name not in image_selection_reasons:
                image_selection_reasons[img_name] = []
            if not any(
                r["metric_name"] == metric_name and r["primary_model"] == primary_prefix
                for r in image_selection_reasons[img_name]
            ):
                image_selection_reasons[img_name].append(reason)
            current_metric_images.add(img_name)

        top_images_set.update(current_metric_images)
        count_after = len(top_images_set)
        print(
            f"   -> Found {len(df_top_for_metric)} images, added {count_after - count_before} unique images."
        )

    if not top_images_set:
        print(
            f"\nNo images found across any metric where {primary_prefix} outperformed {secondary_prefix}."
        )

    return image_selection_reasons


def create_detailed_comparison_pdf(
    df,
    reasons,
    gt_dir,
    model1_dir,
    model2_dir,
    model1_prefix,
    model2_prefix,
    output_pdf_path,
):
    """
    Creates a detailed PDF comparing images. Titles above model images show
    only the metric values for which the image was selected in the 'reasons' dict.
    """
    # Get the set of images relevant to this PDF from the reasons dictionary keys
    images_to_plot = set(reasons.keys())
    if not images_to_plot:
        print(f"No images with selection reasons found for PDF: {output_pdf_path}")
        return

    # Filter the main DataFrame to only include rows for images we need to plot
    df_plot = df[df["image_name"].isin(images_to_plot)].copy()
    df_plot = df_plot.sort_values(by="image_name").reset_index(drop=True)

    if df_plot.empty:
        print(f"Filtered DataFrame is empty, cannot generate PDF: {output_pdf_path}")
        return

    print(f"\nGenerating detailed PDF: {output_pdf_path}...")
    try:
        with PdfPages(output_pdf_path) as pdf:
            for (
                index,
                row,
            ) in df_plot.iterrows():  # Iterate through the filtered df_plot
                img_name = row["image_name"]
                # Reasons specific to this image (already filtered by the calling context)
                img_reasons = reasons.get(img_name, [])
                if not img_reasons:
                    continue  # Skip if somehow no reasons are found

                img_reasons.sort(key=lambda x: x["metric_name"])

                gt_path = os.path.join(gt_dir, img_name)
                model1_path = os.path.join(model1_dir, img_name)  # e.g., BSRGAN path
                model2_path = os.path.join(model2_dir, img_name)  # e.g., ResShift path

                # --- Check images exist ---
                image_paths = {
                    "GT": gt_path,
                    model1_prefix: model1_path,
                    model2_prefix: model2_path,
                }
                missing = False
                for key, path in image_paths.items():
                    if not os.path.exists(path):
                        print(
                            f"Warning: {key} image not found, skipping PDF page for: {path}"
                        )
                        missing = True
                        break
                if missing:
                    continue
                # --- Load images ---
                try:
                    img_gt = Image.open(gt_path)
                    img_model1 = Image.open(model1_path)
                    img_model2 = Image.open(model2_path)
                except Exception as e:
                    print(
                        f"Warning: Error loading images for {img_name}: {e}. Skipping."
                    )
                    continue

                # --- Prepare text elements ---
                # Titles use the values stored *in the reason* dict for the selecting metrics
                model1_title_metrics = (
                    ", ".join(
                        [
                            f"{r['metric_name']} = {r[f'{model1_prefix}_val']:.4f}"
                            # Check if the reason includes value for model1
                            for r in img_reasons
                            if f"{model1_prefix}_val" in r
                        ]
                    )
                    if img_reasons
                    else "N/A"
                )
                model1_title = f"{model1_prefix} ({model1_title_metrics})"

                model2_title_metrics = (
                    ", ".join(
                        [
                            f"{r['metric_name']} = {r[f'{model2_prefix}_val']:.4f}"
                            # Check if the reason includes value for model2
                            for r in img_reasons
                            if f"{model2_prefix}_val" in r
                        ]
                    )
                    if img_reasons
                    else "N/A"
                )
                model2_title = f"{model2_prefix} ({model2_title_metrics})"

                # Reason text for top of page (indicates primary model for this reason)
                selection_reason_text = (
                    f"Selected for {img_reasons[0]['primary_model']} outperforming:\n"
                )
                if img_reasons:
                    selection_reason_text += "\n".join(
                        [
                            f" - {r['metric_name']} (Diff: {r['difference']:.4f})"
                            for r in img_reasons
                        ]
                    )
                else:
                    selection_reason_text += " (N/A)"

                # --- Create figure ---
                fig = plt.figure(figsize=(15, 8))
                gs = fig.add_gridspec(
                    2, 3, height_ratios=[0.15, 0.85], width_ratios=[1, 1, 1]
                )

                ax_top_text = fig.add_subplot(gs[0, :])
                ax_top_text.text(
                    0.05, 0.95, f"Image: {img_name}", fontsize=14, va="top"
                )
                ax_top_text.text(
                    0.05,
                    0.75,
                    selection_reason_text,
                    fontsize=9,
                    va="top",
                    linespacing=1.5,
                )
                ax_top_text.axis("off")

                # Images order: GT, Model1, Model2
                ax_gt = fig.add_subplot(gs[1, 0])
                ax_gt.imshow(img_gt)
                ax_gt.set_title("Ground Truth", fontsize=10)
                ax_gt.axis("off")

                ax_model1 = fig.add_subplot(gs[1, 1])
                ax_model1.imshow(img_model1)
                ax_model1.set_title(model1_title, fontsize=10)
                ax_model1.axis("off")

                ax_model2 = fig.add_subplot(gs[1, 2])
                ax_model2.imshow(img_model2)
                ax_model2.set_title(model2_title, fontsize=10)
                ax_model2.axis("off")

                plt.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.0, rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Successfully generated detailed PDF: {output_pdf_path}")

    except Exception as e:
        print(
            f"An unexpected error occurred during PDF creation for {output_pdf_path}: {e}"
        )


if __name__ == "__main__":
    print("--- Starting Analysis ---")
    try:
        df_full = pd.read_csv(CSV_FILE_PATH)
        if "image_name" not in df_full.columns:
            raise ValueError(f"'image_name' column missing in {CSV_FILE_PATH}")

        metric_pairs = get_metric_pairs(df_full, MODEL1_PREFIX, MODEL2_PREFIX)
        if not metric_pairs:
            raise ValueError(
                "No valid metric pairs found between specified model prefixes."
            )

        # --- Analysis 1: Model 1 outperforms Model 2 ---
        print(
            f"\n=== Analyzing cases where {MODEL1_PREFIX} outperforms {MODEL2_PREFIX} ==="
        )
        model1_better_reasons = find_top_performing_images(
            df_full, metric_pairs, TOP_N, MODEL1_PREFIX, MODEL2_PREFIX
        )
        if model1_better_reasons:
            pdf1_name = f"{MODEL1_PREFIX}_vs_{MODEL2_PREFIX}_top_performing.pdf"
            create_detailed_comparison_pdf(
                df_full,  # Pass the full df, filtering happens inside pdf func
                model1_better_reasons,
                GT_DIR,
                (
                    BSRGAN_DIR if MODEL1_PREFIX == "BSRGAN" else RESHIFT_DIR
                ),  # Assign correct dir
                (
                    RESHIFT_DIR if MODEL2_PREFIX == "ResShift" else BSRGAN_DIR
                ),  # Assign correct dir
                MODEL1_PREFIX,
                MODEL2_PREFIX,
                pdf1_name,
            )
        else:
            print(
                f"No images found where {MODEL1_PREFIX} outperformed {MODEL2_PREFIX}."
            )

        # --- Analysis 2: Model 2 outperforms Model 1 ---
        print(
            f"\n=== Analyzing cases where {MODEL2_PREFIX} outperforms {MODEL1_PREFIX} ==="
        )
        model2_better_reasons = find_top_performing_images(
            df_full, metric_pairs, TOP_N, MODEL2_PREFIX, MODEL1_PREFIX  # Swapped roles
        )
        if model2_better_reasons:
            pdf2_name = f"{MODEL2_PREFIX}_vs_{MODEL1_PREFIX}_top_performing.pdf"
            create_detailed_comparison_pdf(
                df_full,  # Pass the full df
                model2_better_reasons,
                GT_DIR,
                BSRGAN_DIR if MODEL1_PREFIX == "BSRGAN" else RESHIFT_DIR,  # Model1 dir
                (
                    RESHIFT_DIR if MODEL2_PREFIX == "ResShift" else BSRGAN_DIR
                ),  # Model2 dir
                MODEL1_PREFIX,  # Keep original prefix order for consistency in PDF display
                MODEL2_PREFIX,
                pdf2_name,
            )
        else:
            print(
                f"No images found where {MODEL2_PREFIX} outperformed {MODEL1_PREFIX}."
            )

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {CSV_FILE_PATH}")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main block: {e}")

    print("\n--- Analysis Complete ---")
