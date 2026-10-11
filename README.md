# Training Protocol Matters as Much as Architecture: ISIC 2018 Skin Lesion Segmentation

Code for the paper by Anushka Pokhriyal, Payal Mittal and Parveen Kumar (Thapar Institute of Engineering and Technology; IIT Roorkee), *Training Protocol Matters as Much as Architecture: A Multi-Seed Benchmark of Dermoscopic Skin Lesion Segmentation on ISIC 2018 with External Validation*.

The code trains 50 models. **Experiment A** has six architectures × five seeds under one shared recipe. **Experiment B** has the scratch U-Net under four training protocols × five seeds. All 50 models are evaluated on the official ISIC 2018 Task 1 test set, and the Experiment A models also on the ISIC 2017 Part 1 test set.

## Results (ISIC 2018 test set, 1,000 images, mean ± SD over 5 seeds)

| Model | Dice | Jaccard | Thresholded Jaccard | HD95 (px @256) |
|---|---|---|---|---|
| U-Net (scratch) | 0.8720 ± 0.0033 | 0.7935 ± 0.0040 | 0.7207 ± 0.0075 | 21.32 ± 0.74 |
| Attention U-Net (scratch) | 0.8742 ± 0.0037 | 0.7974 ± 0.0041 | 0.7291 ± 0.0077 | 20.73 ± 0.81 |
| U-Net (ResNet-34) | 0.8912 ± 0.0068 | 0.8171 ± 0.0111 | 0.7691 ± 0.0164 | 17.42 ± 1.00 |
| UNet++ (ResNet-34) | 0.8944 ± 0.0040 | 0.8230 ± 0.0058 | 0.7753 ± 0.0080 | 16.53 ± 0.51 |
| DeepLabV3+ (ResNet-34) | 0.8944 ± 0.0021 | 0.8237 ± 0.0033 | 0.7734 ± 0.0042 | 16.62 ± 0.36 |
| SegFormer (MiT-B2) | **0.8987 ± 0.0057** | **0.8287 ± 0.0095** | **0.7804 ± 0.0125** | **15.69 ± 0.65** |

Protocol study (scratch U-Net): T1 (5 epochs) reaches 0.7335 Jaccard and T3 (converged) 0.7811, a gain of +0.048 from training alone.

## Repository layout

| Path | Contents |
|---|---|
| `src/isicseg_core.py` | Data download and caching, augmentation, per-image metrics (Dice, Jaccard, thresholded Jaccard, HD95, …), statistics (seed CIs, Wilcoxon + Holm, Friedman) and figures |
| `src/isicseg_train.py` | Models (scratch U-Net / Attention U-Net and `segmentation_models_pytorch` models), resumable training loop, flip TTA inference |
| `src/isicseg_experiments.py` | Experiment definitions (shared recipe, protocols T1–T4), runners, analysis |
| `run_experiments.py` | Command-line entry point |
| `notebooks/ISIC2018_segmentation_study.ipynb` | Self-contained Colab/Kaggle notebook (same code) |

## Usage

```bash
git clone https://github.com/anushkap0/skin-lesion-segmentation.git
cd skin-lesion-segmentation
pip install -r requirements.txt
python run_experiments.py --project ./isic2018_study            # everything (about 20–30 GPU hours on a T4; the paper's runs used one A100 80 GB)
python run_experiments.py --project ./isic2018_study --seeds 0   # quick single-seed pass
```

The official ISIC zips (about 11 GB) are downloaded automatically from `isic-challenge-data.s3.amazonaws.com` and cached at 256 × 256. Runs are resumable: finished runs are skipped, and an interrupted run resumes from its last checkpoint. Results are written to `isic2018_study/analysis/`: `summary.md`, tables in Markdown, LaTeX and CSV, statistical tests, and 300-dpi figures.

To add PH2 as a further external test set, download it manually and pass `--ph2-root /path/to/PH2Dataset`.

## Data

ISIC 2018 Task 1 and ISIC 2017 Part 1 data are © their contributors and distributed by the International Skin Imaging Collaboration under their own licence terms (https://challenge.isic-archive.com/data/). This repository contains no images.

## Citation

```
A. Pokhriyal, P. Mittal, P. Kumar. Training Protocol Matters as Much as Architecture: A Multi-Seed Benchmark of
Dermoscopic Skin Lesion Segmentation on ISIC 2018 with External Validation. [Journal], 2026.
```

## License

MIT, see `LICENSE`.
