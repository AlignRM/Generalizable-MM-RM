import asyncio
from collections import defaultdict
import json
from time import sleep
from abc import ABC, abstractmethod
import ray
import torch
from ray.util.queue import Queue
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict
import transformers
from models.internlm2.tokenization_internlm2 import InternLM2Tokenizer
from models.constants import (BOX_END_TOKEN, BOX_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
    IMG_START_TOKEN, QUAD_END_TOKEN, QUAD_START_TOKEN, REF_END_TOKEN, REF_START_TOKEN)
from models.internvl_chat.modeling_internvl_reward import InternVLRewardModel
from torch.utils.data import Dataset
from models.utils import build_transform, dynamic_process_image, replace_llama_rmsnorm_with_fused_rmsnorm
from models.preprocess import preprocess, preprocess_internlm, preprocess_mpt, preprocess_phi3
from PIL import Image
from copy import deepcopy
from train_rm import make_conv
import math

replace_llama_rmsnorm_with_fused_rmsnorm()

class RMPredictor(ABC):
    def __init__(self, predictor_idx):
        self.predictor_idx = predictor_idx

    @abstractmethod
    def get_rewards(self, items: List):
        pass


@ray.remote(num_gpus=1)
class InternVL2RewardPredictor(RMPredictor):
    TYPE = "reward"

    def __init__(self, predictor_idx, model_args, training_args, data_args):
        super().__init__(predictor_idx)

        self.model, self.tokenizer, self.config = self.wrap_model_tokenizer_config_for_inference(model_args, training_args, data_args)
        self.padding = True
        self.pad_to_multiple_of = None
        self.return_tensors = "pt"


    def get_rewards(self, items: List):

        try:
            text_features = [{ "input_ids": feature["input_ids"],
                        "attention_mask": feature["attention_mask"]} for feature in items]
            text_batch = self.tokenizer.pad(
                text_features,
                padding=self.padding,
                max_length=self.tokenizer.model_max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            image_features = [{"pixel_values": feature["pixel_values"],
                        "image_flags": feature["image_flags"],} for feature in items]
            with torch.no_grad():
                scores = self.model(
                    input_ids = text_batch['input_ids'].to(self.model.device),
                    attention_mask = text_batch['attention_mask'].to(self.model.device),
                    pixel_values = torch.concat([f['pixel_values'] for f in image_features]).to(self.model.device).to(torch.bfloat16),
                    image_flags = torch.concat([f['image_flags'] for f in image_features]).to(self.model.device),).logits.cpu().tolist()

            data = defaultdict(list)
            data['score'] = [score[0] for score in scores]
            for item in items:
                for key, value in item.items():
                    if key in ['img_path', 'prompt', 'chosen']:
                        data[key].append(value)

            return self.predictor_idx, data
        except Exception as e:
            print(e, flush=True)
            return self.predictor_idx, None


    def wrap_model_tokenizer_config_for_inference(self, model_args, training_args, data_args,):

        print(model_args.model_name_or_path)
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            num_labels = 1  # RM standard setting
        )
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
                model_max_length=training_args.max_length, 
                padding_side="right",
                use_fast=False,
                trust_remote_code=True,
                add_eos_token = False)
        except:
            tokenizer = InternLM2Tokenizer.from_pretrained(
                model_args.model_name_or_path,
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

        print(ray.get_gpu_ids())
        # Load model and tokenizer
        model = InternVLRewardModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # use_flash_attn=True,
            device_map=split_model(config.llm_config.num_hidden_layers, ray.get_gpu_ids())
        ).eval()

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

        model.config.llm_config.pad_token_id = tokenizer.pad_token_id
        model.language_model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer, config



class LazySequenceDatasetForInference(Dataset):
    def __init__(self, template_name, tokenizer, ds_path, num_image_token, image_size, pad2square=False, dynamic_image_size=False, use_thumbnail=False, 
                 min_dynamic_patch=1, max_dynamic_patch=6, normalize_type='imagenet', use_image_only = False, use_text_only = False, dataset = None):

        super(LazySequenceDatasetForInference, self).__init__()

        if dataset is None:
            self.dataset = []
            with open(ds_path, 'r') as f: ## ds_path represents a .jsonl file
                for line in f.readlines():
                    item = json.loads(line)
                    if 'responses' in item:
                        for response in item['responses']:
                            self.dataset.append({"prompt": item['prompt'], 'img_path': item['img_path'], 'chosen': response})

        else:
            self.dataset = dataset
            
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
        # prompt: str = '<image>\n' + item['prompt']
        img_path: str = item['img_path']
        chosen: str = item['chosen']
        chosen_conv = make_conv(prompt, chosen)

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

        return {
                "prompt": item['prompt'], "img_path": item['img_path'], 'chosen': item['chosen'],
                "input_ids": chosen_ret['input_ids'][0],
                "attention_mask": chosen_ret["attention_mask"][0],
                "pixel_values": pixel_values,
                "image_flags": torch.tensor([1] * num_patches, dtype=torch.long)
            }

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_item = self.dataset[i]
        ret = self.get_item(data_item)
        return ret


def split_model(num_layers, gpu_list, vit_alpha=0.5):
    device_map = {}
    world_size = len(gpu_list)
    gpu_dict = dict()
    for idx, number in enumerate(gpu_list):
        gpu_dict[idx] = idx
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - vit_alpha))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vit_alpha))
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = gpu_dict[i]
            layer_cnt += 1
    device_map['vision_model'] = gpu_dict[0]
    device_map['mlp1'] = gpu_dict[0]
    device_map['language_model.model.tok_embeddings'] = gpu_dict[0]
    device_map['language_model.model.embed_tokens'] = gpu_dict[0]
    device_map['language_model.output'] = gpu_dict[0]
    device_map['language_model.model.norm'] = gpu_dict[0]
    device_map['language_model.lm_head'] = gpu_dict[0]
    device_map['score'] = gpu_dict[0]
    device_map[f'language_model.model.layers.{num_layers - 1}'] = gpu_dict[0]

    return device_map
