# Fastformer for News Recommendation

This repository implements **Fastformer** for news recommendation on the MIND dataset.

Based on:
- [Fastformer](https://github.com/wuch15/Fastformer)
- [PLM4NewsRec](https://github.com/wuch15/PLM4NewsRec)
- [SpeedyRec](https://github.com/microsoft/SpeedyRec/tree/main/speedy_mind)

---

## Table of Contents
1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Plotting Metrics](#plotting-metrics)
7. [Troubleshooting](#troubleshooting)

---

## Requirements

### Python Environment
```bash
# Create conda environment
conda create -n fastformer python=3.10 -y
conda activate fastformer

# Install dependencies
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.2
pip install tensorflow
pip install scikit-learn pandas tqdm matplotlib
```

### Hardware
- **GPU**: NVIDIA GPU with at least 10GB VRAM (tested on RTX A6000)
- **CPU**: Also works on CPU (slower)

---

## Project Structure

```
fastformer-for-rec/
├── data/
│   └── speedy_data/          # Processed data
│       ├── train/            # Training data (ProtoBuf_*.tsv)
│       └── dev/              # Validation data (ProtoBuf_*.tsv)
├── models/
│   └── bert-tiny/            # Pretrained model (config.json, vocab.txt, etc.)
├── saved_models/             # Saved checkpoints & metrics
├── train.py                  # Training script
├── evaluate_dev.py           # Evaluation on dev set
├── submission.py             # Generate submission for leaderboard
├── plot_metrics.py           # Plot training metrics
├── data_generation.py        # Preprocess raw MIND data
└── README.md
```

---

## Data Preparation

### Step 1: Download MIND Dataset
Download from [MIND official page](https://msnews.github.io/):
- `MINDlarge_train.zip`
- `MINDlarge_dev.zip`

Extract to a folder, e.g., `./data/`:
```
data/
├── MINDlarge_train/
│   ├── news.tsv
│   └── behaviors.tsv
└── MINDlarge_dev/
    ├── news.tsv
    └── behaviors.tsv
```

### Step 2: Generate Processed Data
```bash
python data_generation.py --raw_data_path ./data
```

This creates `./data/speedy_data/` with:
- `train/ProtoBuf_*.tsv` - Training samples
- `dev/ProtoBuf_*.tsv` - Validation samples
- `*_preprocessed_docs.pkl` - Cached news features

### Step 3: Download Pretrained Model

Run the download script to automatically download a lightweight BERT model:

```bash
python download_model.py
```

This downloads `bert-tiny` (~17MB) to `./models/bert-tiny/` with all necessary files:
- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- `vocab.txt`
- `tokenizer_config.json`

> **Note**: For better performance, you can modify `download_model.py` to use larger models like `bert-base-uncased` or `microsoft/deberta-base`.

---

## Training

### Single GPU Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --pretrained_model_path ./models/bert-tiny \
  --root_data_dir ./data/speedy_data/ \
  --filename_pat "ProtoBuf_*.tsv" \
  --epochs 5 \
  --world_size 1 \
  --batch_size 16 \
  --log_steps 100 \
  --news_dim 256 \
  --lr 1e-4 \
  --pretrain_lr 8e-6 \
  --savename my_fastformer
```

### Multi-GPU Training (Distributed)
```bash
python train.py \
  --pretrained_model_path ./models/bert-tiny \
  --root_data_dir ./data/speedy_data/ \
  --filename_pat "ProtoBuf_*.tsv" \
  --epochs 5 \
  --world_size 4 \
  --batch_size 42 \
  --log_steps 100 \
  --news_dim 256 \
  --savename my_fastformer_ddp
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pretrained_model_path` | Path to pretrained BERT model | Required |
| `--root_data_dir` | Path to processed data | Required |
| `--epochs` | Number of training epochs | 6 |
| `--world_size` | Number of GPUs (1 for single GPU) | -1 (all) |
| `--batch_size` | Batch size per GPU | 64 |
| `--news_dim` | News embedding dimension | 256 |
| `--lr` | Learning rate for model | 1e-4 |
| `--pretrain_lr` | Learning rate for pretrained encoder | 1e-4 |
| `--log_steps` | Log every N steps | 200 |
| `--savename` | Name prefix for saved models | speedy |

### Output
- Checkpoints: `./saved_models/{savename}-epoch-{N}.pt`
- Metrics JSON: `./saved_models/{savename}_metrics.json`

---

## Evaluation

### Evaluate on Dev Set
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_dev.py \
  --pretrained_model_path ./models/bert-tiny \
  --root_data_dir ./data/speedy_data/ \
  --load_ckpt_name ./saved_models/my_fastformer-epoch-5.pt \
  --batch_size 64 \
  --news_dim 256
```

### Expected Output
```
[INFO] FINAL RESULTS ON DEV SET:
[INFO] [0] all users Ed: 73088: 61.01   26.38   28.84   35.39
                          ^AUC    ^MRR   ^nDCG@5 ^nDCG@10
[INFO] Final AUC on dev set: 0.6101
```

### Generate Submission (for MIND Leaderboard)
```bash
CUDA_VISIBLE_DEVICES=0 python submission.py \
  --pretrained_model_path ./models/bert-tiny \
  --root_data_dir ./data/speedy_data/ \
  --load_ckpt_name ./saved_models/my_fastformer-epoch-5.pt \
  --batch_size 256 \
  --news_dim 256
```

This creates `prediction.zip` for submission.

---

## Plotting Metrics

After training, plot the metrics:

```bash
# Save plot to file
python plot_metrics.py \
  --metrics_file ./saved_models/my_fastformer_metrics.json \
  --output training_curves.png

# Or show interactively
python plot_metrics.py \
  --metrics_file ./saved_models/my_fastformer_metrics.json
```

### Output
- 4 subplots: Train Loss, Train Accuracy, Dev AUC/MRR, Dev nDCG
- Summary table in console

---

## Troubleshooting

### 1. CUDA Out of Memory
- Reduce `--batch_size`
- Use smaller model (`bert-tiny` instead of `bert-base`)

### 2. Training Hangs (world_size=1)
- Make sure you use `CUDA_VISIBLE_DEVICES=X` to specify GPU
- The code automatically uses single-GPU mode when `world_size=1`

### 3. "nan" Metrics on Dev Set
- Check that `--news_dim` matches training (default: 256)
- Ensure data files exist in `./data/speedy_data/dev/`

### 4. Tokenizer Warning
The warning about `BertTokenizer` vs `TuringNLRv3Tokenizer` is expected and can be ignored.

---

## Metrics Reference

| Metric | Description |
|--------|-------------|
| **AUC** | Area Under ROC Curve (higher = better) |
| **MRR** | Mean Reciprocal Rank (higher = better) |
| **nDCG@5** | Normalized DCG at top 5 (higher = better) |
| **nDCG@10** | Normalized DCG at top 10 (higher = better) |

### Baseline Results (bert-tiny, 1 epoch)
| AUC | MRR | nDCG@5 | nDCG@10 |
|-----|-----|--------|---------|
| 61.01 | 26.38 | 28.84 | 35.39 |

---

## Citation

If you use this code, please cite:
```bibtex
@article{wu2021fastformer,
  title={Fastformer: Additive Attention Can Be All You Need},
  author={Wu, Chuhan and Wu, Fangzhao and Qi, Tao and Huang, Yongfeng},
  journal={arXiv preprint arXiv:2108.09084},
  year={2021}
}
```

---

## License
MIT License