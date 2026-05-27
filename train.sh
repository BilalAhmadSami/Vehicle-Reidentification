#!/bin/bash
# =============================================================================
# train.sh — Experiment Configuration
#
# Documents the command-line configurations used across all experiments for
# vehicle re-identification using the university-provided training script.
# The active command (Section 1, Q1 baseline) is uncommented; all others
# are commented out with their test-set results for reference.
#
# Usage:
#   Update DATASET_ROOT to point to your local VeRi dataset path,
#   then run: bash train.sh
#
# Note: main.py is not included in this repository (university copyright).
# =============================================================================

DATASET_ROOT="path/to/VeRi"

# =============================================================================
# SECTION 1 — Architecture Comparison
# Identical hyperparameters across all three runs; only -a (arch) changes.
# AMSGrad | lr=3e-4 | batch=64 | epochs=10
# =============================================================================

# Q1 — MobileNetV3-Small (default)   |  mAP: 44.5%  |  Rank-1: 80.2%
python main.py \
  -s veri -t veri \
  -a mobilenet_v3_small \
  --root "$DATASET_ROOT" \
  --height 224 --width 224 \
  --optim amsgrad \
  --lr 0.0003 \
  --max-epoch 10 \
  --stepsize 20 40 \
  --train-batch-size 64 \
  --test-batch-size 100 \
  --save-dir logs/mobilenet_v3_small-veri

# Q2 — ResNet50 (best architecture)  |  mAP: 51.1%  |  Rank-1: 83.4%
# python main.py \
#   -s veri -t veri \
#   -a resnet50 \
#   --root "$DATASET_ROOT" \
#   --height 224 --width 224 \
#   --optim amsgrad \
#   --lr 0.0003 \
#   --max-epoch 10 \
#   --stepsize 20 40 \
#   --train-batch-size 64 \
#   --test-batch-size 100 \
#   --save-dir logs/resnet50-veri

# Q3 — VGG16                          |  mAP: 18.2%  |  Rank-1: 53.2%
# python main.py \
#   -s veri -t veri \
#   -a vgg16 \
#   --root "$DATASET_ROOT" \
#   --height 224 --width 224 \
#   --optim amsgrad \
#   --lr 0.0003 \
#   --max-epoch 10 \
#   --stepsize 20 40 \
#   --train-batch-size 64 \
#   --test-batch-size 100 \
#   --save-dir logs/vgg16-veri

# =============================================================================
# SECTION 2 — Data Augmentation (MobileNetV3-Small)
# Default augmentation = random horizontal flip + Random2DTranslation
# =============================================================================

# Q1a — Default + Colour Jitter       |  mAP: 44.6%  |  Rank-1: 81.0%
# python main.py \
#   -s veri -t veri \
#   -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" \
#   --height 224 --width 224 \
#   --optim amsgrad --lr 0.0003 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 64 --test-batch-size 100 \
#   --color_jitter \
#   --save-dir logs/mobilenet-color-jitter

# Q1b — Default + Random Erase        |  mAP: 46.3%  |  Rank-1: 80.0%
# python main.py \
#   -s veri -t veri \
#   -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" \
#   --height 224 --width 224 \
#   --optim amsgrad --lr 0.0003 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 64 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-random-erase

# Q2 — Best combination: Random Erase + Colour Jitter  |  mAP: 46.6%  |  Rank-1: 81.0%
# python main.py \
#   -s veri -t veri \
#   -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" \
#   --height 224 --width 224 \
#   --optim amsgrad --lr 0.0003 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 64 --test-batch-size 100 \
#   --random_erase --color_jitter \
#   --save-dir logs/mobilenet-erase-jitter

# =============================================================================
# SECTION 3.1 — Learning Rate (MobileNetV3-Small, batch=64)
# =============================================================================

# LR=1e-5  (too slow — underfits)     |  mAP: 21.2%  |  Rank-1: 46.5%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim amsgrad --lr 0.00001 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 64 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-lr-1e5

# LR=5e-5                              |  mAP: 36.0%  |  Rank-1: 64.9%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim amsgrad --lr 0.00005 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 64 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-lr-5e5

# LR=1e-4                              |  mAP: 42.2%  |  Rank-1: 75.3%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim amsgrad --lr 0.0001 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 64 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-lr-1e4

# LR=1e-3  (too high — overshoots)    |  mAP: 41.0%  |  Rank-1: 79.0%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim amsgrad --lr 0.001 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 64 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-lr-1e3

# =============================================================================
# SECTION 3.2 — Batch Size (MobileNetV3-Small, LR=3e-4)
# =============================================================================

# BS=32                                |  mAP: 47.0%  |  Rank-1: 82.7%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim amsgrad --lr 0.0003 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 32 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-bs32

# BS=48 (best)                         |  mAP: 47.6%  |  Rank-1: 82.4%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim amsgrad --lr 0.0003 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 48 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-bs48

# BS=72                                |  mAP: 47.0%  |  Rank-1: 81.8%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim amsgrad --lr 0.0003 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 72 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-bs72

# BS=84                                |  mAP: 47.2%  |  Rank-1: 80.8%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim amsgrad --lr 0.0003 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 84 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-bs84

# =============================================================================
# SECTION 3.3 — Optimiser (MobileNetV3-Small, LR=3e-4, BS=48)
# =============================================================================

# SGD (poor convergence)              |  mAP: 19.9%  |  Rank-1: 44.4%
# python main.py \
#   -s veri -t veri -a mobilenet_v3_small \
#   --root "$DATASET_ROOT" --height 224 --width 224 \
#   --optim sgd --lr 0.0003 \
#   --max-epoch 10 --stepsize 20 40 \
#   --train-batch-size 48 --test-batch-size 100 \
#   --random_erase \
#   --save-dir logs/mobilenet-sgd
