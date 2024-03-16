## Introduction

We provide the processed data used in our paper to be appeared at NAACL 2024: [Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models](https://arxiv.org/abs/2311.08921).

The processed WikiGold dataset is provided in this repository for quick start. The rest of the processed datasets can be downloaded in [Google Drive](https://drive.google.com/file/d/13ODu2-PQWshJTVf-LFt8zdVrzTF9MAfE/view?usp=sharing).

## Data splits

We use the original train split as our unlabaled set. For those datasets having original train/dev/test splits, we obtain our unlabeled set by combining the original train and dev splits. We use the original test split for evaluation.

## Test set sampling

For cost saving, we evaluate on two set of randomly sampled 300 samples of the original test set, and report the average results in our paper.

We also provide the sampled 300 test samples used in our paper. We sampled with two random seeds, 42, 52. For example, the folder ***conll2003_300_42*** contains the randomly sampled 300 test samples of CoNLL2003 and is sampled with seed 42.
