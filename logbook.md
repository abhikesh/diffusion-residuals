Experiment 1
python inference_resshift.py -i testdata/imagenet256/lq -o results/imagenet256/hq --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64

use_ycbr = true
Average metrics over 3000 images:
MEAN PSNR: 24.9339
MEAN SSIM: 0.6719
MEAN LPIPS-VGG: 0.3428
MEAN LPIPS-Alex: 0.2373
MEAN CLIPIQA: 0.5862
MEAN MUSIQ: 53.11

use_ycbr = false
Average metrics over 3000 images:
MEAN PSNR: 22.9987
MEAN SSIM: 0.6175
MEAN LPIPS-VGG: 0.3428
MEAN LPIPS-Alex: 0.2373
MEAN CLIPIQA: 0.5862
MEAN MUSIQ: 53.11


python inference_resshift.py -i testdata/RealSet80 -o results/RealSet80 --task realsr --scale 4 --version v1 --chop_size 512 --chop_stride 448

Average metrics over 80 images:
MEAN CLIPIQA: 0.6749
MEAN MUSIQ: 63.12

python inference_resshift.py -i testdata/RealSet65 -o results/RealSet65 --task realsr --scale 4 --version v1 --chop_size 512 --chop_stride 448
Average metrics over 65 images:
MEAN CLIPIQA: 0.6478
MEAN MUSIQ: 60.73

Journal Table 2
python inference_resshift.py -i testdata/imagenet256/lq -o testdata/imagenet256-v3/lq --task realsr --scale 4 --version v3 --chop_size 64 --chop_stride 64 --bs 64

Average metrics over 3000 images:
MEAN PSNR: 25.0132
MEAN SSIM: 0.6813
MEAN LPIPS-VGG: 0.3007
MEAN LPIPS-Alex: 0.2082
MEAN CLIPIQA: 0.5966
MEAN MUSIQ: 52.01

^These results look a little sus - abandon


Journal Celeb Face inpainting
python inference_resshift.py -i testdata/CelebA-Test/lq -o results/CelebA-Test/hq  --mask_path testdata/CelebA-Test/mask --task inpaint_face --scale 1 --chop_size 256 --chop_stride 256 --bs 32

Mask types: box, images: 500
  LPIPS-VGG: 0.0744
  LPIPS-Alex: 0.0549
  CLIPIQA: 0.7083
  MUSIQ: 68.21
Mask types: irregular, images: 500
  LPIPS-VGG: 0.1660
  LPIPS-Alex: 0.1173
  CLIPIQA: 0.7196
  MUSIQ: 67.89
Mask types: expand, images: 500
  LPIPS-VGG: 0.3605
  LPIPS-Alex: 0.2776
  CLIPIQA: 0.7568
  MUSIQ: 67.73
Mask types: half, images: 500
  LPIPS-VGG: 0.2023
  LPIPS-Alex: 0.1533
  CLIPIQA: 0.7179
  MUSIQ: 67.20
MEAN LPIPS-VGG: 0.2008
MEAN LPIPS-Alex: 0.1508
MEAN CLIPIQA: 0.7256
MEAN MUSIQ: 67.76



**AFTER HACKING PYIQA to avoid the [0, 1] normalization check and passing the
[-1, 1] normalized data

Mask types: box, images: 500
  LPIPS-VGG: 0.0744
  LPIPS-Alex: 0.0549
  CLIPIQA: 0.4911
  MUSIQ: 67.20
Mask types: irregular, images: 500
  LPIPS-VGG: 0.1660
  LPIPS-Alex: 0.1173
  CLIPIQA: 0.5001
  MUSIQ: 66.99
Mask types: expand, images: 500
  LPIPS-VGG: 0.3605
  LPIPS-Alex: 0.2776
  CLIPIQA: 0.5104
  MUSIQ: 66.08
Mask types: half, images: 500
  LPIPS-VGG: 0.2023
  LPIPS-Alex: 0.1533
  CLIPIQA: 0.5190
  MUSIQ: 66.34
MEAN LPIPS-VGG: 0.2008
MEAN LPIPS-Alex: 0.1508
MEAN CLIPIQA: 0.5052
MEAN MUSIQ: 66.65
