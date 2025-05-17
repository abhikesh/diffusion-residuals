
Okay, let's explore the experimental design for the second idea: **"Robustness to Degradations Beyond Training"** from `docs/conf_version_ideas.md`.

The core hypothesis is that the 15-step ResShift model, trained on a specific synthetic degradation pipeline (RealESRGAN-style + bicubic), might struggle when faced with real-world images containing different or more complex degradations it wasn't trained on.

We need to design experiments to test this *without* retraining ResShift.

**Key Components of the Experiment:**

1.  **Model Under Test:** The official pre-trained 15-step ResShift SR model.
2.  **Test Datasets:** Datasets containing degradations *different* from ResShift's training data.
3.  **Comparison Models:** Models known for handling diverse/unknown degradations.
4.  **Evaluation Metrics:** Both quantitative (where possible) and qualitative metrics.

**Exploring Options:**

**Option 1: Real-World Dataset Evaluation**

*   **Concept:** Directly test ResShift on datasets known for diverse, real-world degradations.
*   **Datasets:**
    *   `DRealSR`: Contains pairs captured in the wild. Degradations are complex and unknown.
    *   `RealSRSet`: Similar to DRealSR, another collection of real-world images.
    *   *Challenge:* Ground truth (GT) might not be perfectly aligned or clean, making reference metrics like PSNR/SSIM less reliable interpretation-wise.
*   **Comparison Models:**
    *   `BSRGAN`: Designed specifically for blind SR (unknown degradations). Should be a strong baseline here.
    *   `Real-ESRGAN`: The method used for ResShift's *training* degradation. It's interesting to see how it performs on *different* real-world data compared to ResShift.
*   **Metrics:**
    *   **Primary:** No-reference metrics (LPIPS, CLIPIQA, MUSIQ) as GT quality is variable. Lower LPIPS is better; higher CLIPIQA/MUSIQ *usually* indicates better perceptual quality, though they can sometimes be fooled.
    *   **Secondary:** Qualitative visual comparison. This is crucial. We'd look for:
        *   Failure to remove specific real-world artifacts (e.g., blur, noise, compression blocks).
        *   Introduction of new artifacts.
        *   Overall realism and detail recovery compared to BSRGAN/Real-ESRGAN.
*   **Procedure Outline:**
    1.  Obtain datasets (DRealSR, RealSRSet).
    2.  Obtain pre-trained models (ResShift, BSRGAN, Real-ESRGAN).
    3.  Run inference for all models on the LR images from the datasets.
    4.  Calculate LPIPS, CLIPIQA, MUSIQ for the generated SR images. (Note: LPIPS technically needs a reference, but it's often calculated against the provided GT in these datasets, acknowledging the GT limitations).
    5.  Systematically compare outputs visually, categorizing failure modes.

**Option 2: Controlled Synthetic Degradation Evaluation**

*   **Concept:** Create new test sets using clean images, apply specific degradations *not* in ResShift's training pipeline, downsample, and then test.
*   **Datasets (Examples to Create):**
    *   **Motion Blur:** Start with a clean dataset (e.g., DIV2K, Flickr2K), apply realistic motion blur kernels, then downsample (e.g., bicubic x4).
    *   **Heavy Noise:** Start clean, add strong Gaussian noise (or other types), then downsample.
    *   **Severe JPEG:** Start clean, apply heavy JPEG compression (low quality factor), then downsample.
    *   **Mixed Degradations:** Combine blur + noise, or noise + JPEG in ways different from the RealESRGAN pipeline.
*   **Comparison Models:**
    *   `BSRGAN` / `Real-ESRGAN` (as above).
    *   Maybe a kernel-estimating method like `DAN` or `IKC` if easily available (especially relevant for the blur test).
    *   `Bicubic Interpolation` (as a basic baseline).
*   **Metrics:**
    *   **Primary:** Reference metrics (PSNR, SSIM) calculated against the *original clean* HR images. This directly measures restoration quality for the specific degradation.
    *   **Secondary:** No-reference metrics (LPIPS, CLIPIQA, MUSIQ) also calculated against the original clean HR.
    *   **Tertiary:** Qualitative visual comparison, focusing on how well the specific degradation was handled.
*   **Procedure Outline:**
    1.  Select a clean HR dataset.
    2.  Implement degradation functions (motion blur, noise, JPEG).
    3.  Generate the degraded LR test sets.
    4.  Obtain pre-trained models.
    5.  Run inference for all models on the generated LR images.
    6.  Calculate PSNR, SSIM, LPIPS, CLIPIQA, MUSIQ against the original clean HR images.
    7.  Visually compare results for artifact removal/introduction.

**Discussion & Next Steps:**

*   **Option 1 (Real-World):** Pros - Directly tests real-world applicability. Cons - Metrics are harder to interpret due to GT issues; relies more on visual judgment.
*   **Option 2 (Synthetic):** Pros - Controlled environment, reliable quantitative metrics (PSNR/SSIM against clean GT). Cons - Requires dataset creation effort; might not fully capture real-world complexity.

A good approach might be to start with **Option 1** as it's closer to the paper's stated limitation (Fig 9 failure). We can run ResShift, BSRGAN, and Real-ESRGAN on DRealSR or RealSRSet and primarily rely on visual comparisons and no-reference metrics like CLIPIQA/MUSIQ.

If Option 1 shows clear weaknesses, we could then design a specific **Option 2** experiment (e.g., motion blur) to quantify the performance drop using PSNR/SSIM on a controlled degradation type.

Does this plan sound good? Should we focus first on setting up the real-world dataset evaluation (Option 1)? We'd need to identify where to get the datasets and the pre-trained BSRGAN/Real-ESRGAN models.
