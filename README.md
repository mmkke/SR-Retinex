# SR-Retinex

Retinex-based image enhancement pipeline for 16-bit images, with optional ISD map prediction from a ViT model checkpoint.

## Setup

Use either Conda:

```bash
conda env create -f environment.yml
conda activate sr-retinex
```

Or `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Process a single image:

```bash
python src/retinex2_corrected.py \
  --input data/submission_images/tang_yilin_033.tif \
  --output_dir retinex2_output \
  --beta 100 \
  --max_abs_alpha 0.9 \
  --gamma 6.2 \
  --use_model
```

Process all `.tif` images in a directory:

```bash
bash run.sh
```

Custom input and output directories:

```bash
bash run.sh data/submission_images retinex2_output
```

## Outputs

Results are written under `retinex2_output/`. The plotting script can be run after summaries are generated:

```bash
python src/plot_metrics.py
```

## Notes

- The default model path is `model/vit_12_linear.pth`.
- `--use_model` uses the ViT-based ISD estimator.
- All Python source files live directly under `src/`.
