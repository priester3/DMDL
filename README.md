# DMDL: Dual-level Modality Debiasing Learning for Unsupervised Visible-Infrared Person Re-Identification

Official implementation of the paper ["Dual-level Modality Debiasing Learning for Unsupervised Visible-Infrared Person Re-Identification"](https://arxiv.org/abs/2512.03745).

The code is implemented based on PGM ([USL-VI-ReID](https://github.com/zesenwu23/USL-VI-ReID)).

This repository is a cleaned DMDL codebase for three USL-VI-ReID benchmarks:

- SYSU-MM01
- RegDB
- LLCM

The repository layout is intentionally kept close to public USL-VI-ReID code releases such as [USL-VI-ReID](https://github.com/zesenwu23/USL-VI-ReID): compact top-level entry scripts, a dedicated `prepare/` folder for dataset preprocessing, and the reusable method code under `clustercontrast/`.

## Repository Layout

```text
.
├── clustercontrast/         # reusable models, datasets, losses, evaluation, utils
├── ChannelAug.py            # modality-specific augmentation operators
├── dmdl_sysu.py             # SYSU legacy backend kept for method parity
├── dmdl_regdb.py            # RegDB legacy backend kept for method parity
├── dmdl_llcm.py             # LLCM legacy backend kept for method parity
├── prepare/
│   ├── common.py           # shared preprocessing helpers
│   ├── prepare_sysu.py     # SYSU-MM01 preprocessing
│   ├── prepare_regdb.py    # RegDB preprocessing
│   └── prepare_llcm.py     # LLCM preprocessing
├── run_train_sysu.sh        # one-command SYSU stage1+stage2 training
├── run_train_regdb.sh       # one-command RegDB stage1+stage2 training
├── run_train_llcm.sh        # one-command LLCM stage1+stage2 training
├── train.py                 # unified training entrypoint
├── evaluate.py              # unified evaluation entrypoint
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

If `faiss-gpu` is not available from `pip` on your platform, install the CUDA-compatible PyTorch and FAISS packages with `conda` first, then install the remaining Python dependencies.

The codebase targets the original GPU-oriented dependency stack in `requirements.txt`:

- `torch==1.13.0`
- `torchvision==0.14.0`
- `faiss-gpu==1.6.5`

The released training setup was run on 4 NVIDIA GeForce RTX 4090 GPUs. The code uses `nn.DataParallel` over visible CUDA devices, so exposing 4 GPUs is the recommended public reproduction setting.

## Dataset Preparation

Convert the original downloaded datasets into the directory layout expected by DMDL:

```bash
python prepare/prepare_sysu.py --data-root /path/to/raw/SYSU-MM01 --output-root /path/to/processed/sysu
python prepare/prepare_regdb.py --data-root /path/to/raw/RegDB --output-root /path/to/processed/regdb
python prepare/prepare_llcm.py --data-root /path/to/raw/LLCM --output-root /path/to/processed/llcm
```

`prepare/prepare_regdb.py` prepares all 10 RegDB trials by default.

Expected processed layout:

```text
<data-dir>/
├── ir_modify/
│   ├── query/
│   ├── bounding_box_test/
│   └── bounding_box_train/
└── rgb_modify/
    ├── query/
    ├── bounding_box_test/
    └── bounding_box_train/
```

For RegDB, each trial is created under `ir_modify/<trial>/` and `rgb_modify/<trial>/`.

## Training

Unified CLI:

```bash
python train.py --dataset sysu --stage 1 --data-dir /path/to/sysu
python train.py --dataset sysu --stage 2 --data-dir /path/to/sysu --stage1-name sysu_s1/dmdl
```

One-command training scripts:

```bash
DATA_DIR=/path/to/SYSU bash run_train_sysu.sh
DATA_DIR=/path/to/RegDB bash run_train_regdb.sh
DATA_DIR=/path/to/LLCM bash run_train_llcm.sh
```

All training scripts accept path overrides through environment variables without embedding machine-specific absolute paths.
The default public scripts correspond to the released paper settings for each dataset. On the first run, PyTorch may download ImageNet pretrained weights for the backbone automatically.

## Evaluation

```bash
python evaluate.py \
  --dataset sysu \
  --data-dir /path/to/sysu \
  --checkpoint /path/to/model_best.pth.tar
```

Use `python evaluate.py --help` to see the dataset-specific options.

## Notes

- `train.py` and `evaluate.py` are the recommended public entrypoints.
- `dmdl_sysu.py`, `dmdl_regdb.py`, and `dmdl_llcm.py` are retained as legacy-compatible backends for direct reproduction and debugging.
- No dataset path or log path is hard-coded to a specific machine in the public training scripts.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{Li2026DMDL,
  title={Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification},
  author={Li, Jiaze and Lu, Yan and Liu, Bin and Yin, Guojun and Ye, Mang},
  journal={Pattern Recognition},
  volume={176},
  pages={113257},
  year={2026},
  doi={10.1016/j.patcog.2026.113257},
  url={https://doi.org/10.1016/j.patcog.2026.113257}
}
```
