# Generalizable Multimodal Reward Model

This repository contains code for the ICML 2025 paper [The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](https://arxiv.org/abs/2503.03122).

## Abstract

Multimodal Reward Models (MM-RMs) are crucial for aligning Large Language Models (LLMs) with human preferences, particularly as LLMs increasingly interact with multimodal data. However, we find that MM-RMs trained on existing datasets often struggle to generalize to out-of-distribution data due to their reliance on unimodal spurious correlations, primarily text-only shortcuts within the training distribution, which prevents them from leveraging true multimodal reward functions. To address this, we introduce a Shortcut-aware MM-RM learning algorithm that mitigates this issue by dynamically reweighting training samples, shifting the distribution toward better multimodal understanding, and reducing dependence on unimodal spurious correlations. Our experiments demonstrate significant improvements in generalization, downstream task performance, and scalability, establishing a more robust framework for multimodal reward modeling.


## Usage

### Training

#### Data Prepare

To train a multimodal reward model, you need to process your intended training data into a JSONL file. Our training framework supports two file formats:

* The first format is similar to the data structure of [VLFeedback](https://huggingface.co/datasets/MMInstruction/VLFeedback). Each data item contains responses from multiple models, with response quality determined based on annotations. Each line of data follows this format:

```
{
  "id": "",
  "prompt": "What is the main focus of this image?",
  "img_path": "path_to/svit-conversation-1219.jpg"
  "completions": {
    "annotations": [...],
    "model": [...],
    "response": [...]
  },
}
```

* The second format adopts a more common pairwise structure, where each data item contains one chosen response and one rejected response. We have processed [POVID](https://huggingface.co/datasets/YiyangAiLab/POVID_preference_data_for_VLLMs) and [RLHF-V](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset) into this format. Each line of data follows this example format:

```
{
  "id": "",
  "prompt": "What are the key features you observe in the image?",
  "img_path": "path_to/llava1.5_raw_images/00013/000139279.jpg",
  "chosen": "",
  "rejected": ""
}
```

#### Launch Running

To launch the training process, we first need to prepare the configuration file to set up the base model, data paths, and related parameters, as demonstrated in the `configs/` folder. The training script implementations for both the standard RM algorithm and our shortcut-aware algorithm are available in `scripts/train_standard.sh` and `scripts/train_robust.sh`, respectively.

### Inference

#### Data Prepare

After completing RM training, the data to be inferred needs to be organized into a JSONL file where each line follows this format:

```json
{
  "prompt": "",
  "img_path": "",
  "responses": [...]
}
```

#### RM Scoring

Whether implemented using the standard algorithm or the shortcut-aware algorithm, all MM-RMs can perform inference through a unified implementation. Specifically, we have implemented a reward score sampling strategy based on Ray scheduling. The corresponding implementation scripts are demonstrated in `scripts/inference.sh`.

## Model

In the original paper, for analytical purposes in cross-distribution experiments, we limited our training data to a single data environment. To better serve the community, we plan to train a more robust multimodal reward model by combining the proposed shortcut-aware MM-RM algorithm with a more comprehensive preference dataset (which is a more common practice in the open-source community). We plan to release this reward model upon completion.


## Citation

``` bibtex
@inproceedings{li2025devil,
  title={The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models},
  author={Li, Zichao and Wen, Xueru and Lou, Jie and Ji, Yuqiu and Lu, Yaojie and Han, Xianpei and Zhang, Debing and Sun, Le},
  booktitle={International Conference on Machine Learning}
  year={2025}
}
```

## Acknowledgment

This repo is built upon [trl](https://github.com/huggingface/trl) and [InternVL](https://github.com/OpenGVLab/InternVL), with also inspiration from [VLFeedback](https://github.com/vlf-silkie/VLFeedback).
