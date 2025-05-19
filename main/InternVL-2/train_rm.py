"""An example of implementing InternVL2 Reward Modeling (RM)."""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional
# from datasets import Dataset
from torch.utils.data import Dataset
from datasets import Dataset as HF_Dataset
from PIL import Image
import datasets
import numpy as np
import torch.distributed
import transformers
import torch.nn as nn
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, prepare_model_for_kbit_training

from transformers import GPTQConfig, PreTrainedModel
from transformers.trainer_pt_utils import LabelSmoother
from trl.trainer import DPOTrainer, RewardTrainer
from trl.trainer.utils import DPODataCollatorWithPadding, RewardDataCollatorWithPadding
from trl import RewardConfig
from typing import Union, Any, Tuple
from arguments import TrainingArguments, DataTrainingArguments, LoraArguments, ModelArguments
from models.constants import (BOX_END_TOKEN, BOX_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
    IMG_START_TOKEN, QUAD_END_TOKEN, QUAD_START_TOKEN, REF_END_TOKEN, REF_START_TOKEN)
from models.utils import build_transform, dynamic_process_image, replace_llama_rmsnorm_with_fused_rmsnorm
from models.preprocess import preprocess, preprocess_internlm, preprocess_mpt, preprocess_phi3
from copy import deepcopy
from models.internvl_chat.modeling_internvl_reward import InternVLRewardModel
from models.internlm2.tokenization_internlm2 import InternLM2Tokenizer
import pathlib


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

import warnings
warnings.filterwarnings('ignore')

replace_llama_rmsnorm_with_fused_rmsnorm()

# Set constants for image processing
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class InternRewardDataCollator(RewardDataCollatorWithPadding):
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    def __call__(self, features):
        features_chosen = []
        features_rejected = []
        features_common = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        for feature in features:
            assert ("input_ids_chosen" in feature and "input_ids_rejected" in feature and 'attention_mask_chosen' in feature
                and "attention_mask_rejected" in feature and 'pixel_values' in feature and 'image_flags' in feature)

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            features_common.append(
                {
                    "pixel_values": feature["pixel_values"],
                    "image_flags": feature["image_flags"],
                }
            )
            if has_margin:
                margin.append(feature["margin"])

        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "pixel_values": torch.concat([f['pixel_values'] for f in features_common]),
            "image_flags": torch.concat([f['image_flags'] for f in features_common]),
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch



def make_conv(prompt, answer):

    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]



def compute_loss(
    self,
    model: Union[PreTrainedModel, nn.Module],
    inputs: Dict[str, Union[torch.Tensor, Any]],
    return_outputs=False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    warnings.warn(
        "Using self-defined compute_loss function"
    )
    rewards_chosen = model(
        input_ids=inputs["input_ids_chosen"],
        attention_mask=inputs["attention_mask_chosen"],
        pixel_values=inputs['pixel_values'],
        image_flags=inputs['image_flags'],
        return_dict=True,
    )["logits"]
    rewards_rejected = model(
        input_ids=inputs["input_ids_rejected"],
        attention_mask=inputs["attention_mask_rejected"],
        pixel_values=inputs['pixel_values'],
        image_flags=inputs['image_flags'],
        return_dict=True,
    )["logits"]
    # calculate loss, optionally modulate with margin
    if "margin" in inputs:
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
    else:
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

    # if self.args.center_rewards_coefficient is not None:
    #     loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

    if return_outputs:
        return loss, {
            "rewards_chosen": rewards_chosen,
            "rewards_rejected": rewards_rejected,
        }
    return loss


def make_batch_pairs(sample):
    if 'chosen' in sample and 'rejected' in sample:
        assert 'prompt' in sample and 'img_path' in sample
        return sample
    
    converted_sample = defaultdict(list)
    for sample_idx, comps in enumerate(sample["completions"]):
        prompt = sample["prompt"][sample_idx]
        img_path = sample['img_path'][sample_idx]

        for comp_idx1, comp_idx2 in combinations(range(len(comps["annotations"])), 2):
            anno1, anno2 = comps["annotations"][comp_idx1], comps["annotations"][comp_idx2]

            # get average scores
            try:
                avg_score1 = np.mean(
                    [
                        float(anno1[aspect]["Rating"])
                        for aspect in anno1
                    ]
                )
                avg_score2 = np.mean(
                    [
                        float(anno2[aspect]["Rating"])
                        for aspect in anno2
                    ]
                )
            except ValueError:
                continue

            # get chosen and rejected responses
            if avg_score1 > avg_score2:
                chosen = comps["response"][comp_idx1]
                rejected = comps["response"][comp_idx2]
            elif avg_score2 > avg_score1:
                chosen = comps["response"][comp_idx2]
                rejected = comps["response"][comp_idx1]
            else:
                continue
            converted_sample["prompt"].append(prompt)
            converted_sample['img_path'].append(img_path)
            converted_sample["chosen"].append(chosen)
            converted_sample["rejected"].append(rejected)

    return converted_sample



class LazyInternRewardDataset(Dataset):
    def __init__(self, template_name, tokenizer, ds, dataset_name, num_image_token, image_size, pad2square=False, dynamic_image_size=False, 
        use_thumbnail=False, min_dynamic_patch=1, max_dynamic_patch=6, normalize_type='imagenet', use_image_only = False, use_text_only = False):

        super(LazyInternRewardDataset, self).__init__()

        if ds is None:
            ds = datasets.load_dataset(dataset_name, split = 'train')
        
        # make comparison pairs from completion list
        if local_rank > 0:
            print("Waiting for main process to perform the mapping")
            torch.distributed.barrier()
        ds = ds.map(
            make_batch_pairs, batched=True,
            remove_columns=set(ds.column_names) - set(["prompt", "img_path", "chosen", "rejected"]),)
        
        if local_rank == 0:
            print("Loading results from main process")
            torch.distributed.barrier()

        self.dataset = ds
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.pad2square = pad2square
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        
        self.transform = build_transform(self.image_size, self.pad2square, self.normalize_type)
        
        self.use_image_only = use_image_only
        self.use_text_only = use_text_only


    def __len__(self):
        return len(self.dataset)
    
    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function
    

    def get_item(
        self,
        item,
    ) -> Dict:

        if self.use_image_only:
            prompt: str = '<image>\n'
        else:
            prompt: str = '<image>\n' + item['prompt']
            
        img_path: str = item['img_path']
        chosen: str = item['chosen']
        rejected: str = item['rejected']

        '''make_conv: convert data into unified dict format'''
        chosen_conv = make_conv(prompt, chosen)
        rejected_conv = make_conv(prompt, rejected)

        # load image
        if self.use_text_only:
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            self.max_dynamic_patch = 1
        else:
            image = Image.open(img_path).convert('RGB')
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_process_image(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]
        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if self.use_text_only or not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()
        chosen_ret = preprocess_function(self.template_name, [deepcopy(chosen_conv)],
                                  self.tokenizer, [self.num_image_token * num_patches], truncation = True)
        rejected_ret = preprocess_function(self.template_name, [deepcopy(rejected_conv)],
                                  self.tokenizer, [self.num_image_token * num_patches], truncation = True)

        return {
                "input_ids_chosen": chosen_ret['input_ids'][0],
                "attention_mask_chosen": chosen_ret["attention_mask"][0],
                "input_ids_rejected": rejected_ret['input_ids'][0],
                "attention_mask_rejected": rejected_ret["attention_mask"][0],
                "pixel_values": pixel_values,
                "image_flags": torch.tensor([1] * num_patches, dtype=torch.long)
            }


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_item = self.dataset[i]
        ret = self.get_item(data_item)
        return ret



def wrap_model_tokenizer_config(model_args, training_args, data_args, lora_args):

    print(model_args.model_name_or_path)
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        num_labels = 1  # RM standard setting
    )
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    if config.llm_config.model_type == 'internlm2':
        config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        print('Using flash_attention_2 for InternLM')
    else:
        config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
        print('Using flash_attention_2 for LLaMA')
    config.template = data_args.conv_style
    config.select_layer = model_args.vision_select_layer
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = model_args.ps_version
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.max_length, 
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
            add_eos_token = False)
    except:
        tokenizer = InternLM2Tokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.max_length, 
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
            add_eos_token = False)
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # Load model and tokenizer
    model = InternVLRewardModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # model.language_model.config.output_hidden_states = True
    # model.config.llm_config.output_hidden_states = True

    model.img_context_token_id = img_context_token_id
    assert model.config.downsample_ratio == data_args.down_sample_ratio

    patch_size = model.config.vision_config.patch_size
    if model.config.vision_config.image_size != data_args.force_image_size:
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                new_size=data_args.force_image_size, patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    model.config.llm_config.pad_token_id = tokenizer.pad_token_id
    model.language_model.config.pad_token_id = tokenizer.pad_token_id

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=lora_args.lora_r, lora_alpha=lora_args.lora_alpha, lora_dropout=lora_args.lora_dropout,)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=lora_args.lora_r, lora_alpha=lora_args.lora_alpha, lora_dropout=lora_args.lora_dropout,)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    try:
        _freeze_params(model.language_model.lm_head)
    except:
        _freeze_params(model.language_model.output)

    return model, tokenizer, config



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, DataTrainingArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        data_args
    ) = parser.parse_args_into_dataclasses()
    
    if getattr(training_args, "deepspeed", None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    model, tokenizer, config = wrap_model_tokenizer_config(model_args, training_args, data_args, lora_args)

    if local_rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if training_args.dataset_path != None:
        with open(training_args.dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]
            train_dataset = HF_Dataset.from_list(data)
            train_dataset = LazyInternRewardDataset(data_args.conv_style, tokenizer, train_dataset, 'None', num_image_token=model.num_image_token, image_size=data_args.force_image_size, 
                    pad2square=data_args.pad2square, dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail, min_dynamic_patch=data_args.min_dynamic_patch, 
                    max_dynamic_patch=data_args.max_dynamic_patch, normalize_type=data_args.normalize_type, use_image_only=training_args.use_image_only, use_text_only=training_args.use_text_only)            
            eval_dataset = None

    else:
        raise NotImplementedError

    RewardTrainer.compute_loss = compute_loss
    print(training_args.gradient_checkpointing)
    print("The process idx is: ", training_args.process_index)
    # Start trainner
    trainer = RewardTrainer(
        model,
        args=training_args,
        # beta=training_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator = InternRewardDataCollator(tokenizer=tokenizer, max_length=training_args.max_length),
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-last"))


if __name__ == "__main__":
    train()
