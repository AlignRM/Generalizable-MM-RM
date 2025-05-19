import asyncio
from collections import defaultdict
import json
import ray
import torch
from ray.util.queue import Queue
from tqdm import tqdm
from collections import defaultdict
from typing import List
import os
from infer_rm import InternVL2RewardPredictor, LazySequenceDatasetForInference
import transformers
from arguments import TrainingArguments, DataTrainingArguments, ModelArguments
from models.internlm2.tokenization_internlm2 import InternLM2Tokenizer


async def get_rewards(predictors, predictor_queue, batches, dest_file):
    tasks = []
    num_submit, num_done, num_fail = 0, 0, 0
    score_bar = tqdm(
        total=sum(map(len, batches)),
        desc=f"Scoring {num_done}/{num_submit}",
    )
    # all_data = defaultdict(list)
    all_data = []
    while num_done < len(batches):
        while not predictor_queue.empty() and num_submit < len(batches):
            predictor = predictors[predictor_queue.get()]
            tasks.append(predictor.get_rewards.remote([lazy_dataset.get_item(item) for item in batches[num_submit]]))
            num_submit += 1
            score_bar.set_description(f"Scoring {num_done}/{num_submit}")

        if tasks:
            completed_tasks, tasks = ray.wait(tasks, num_returns=1, timeout=None)
            for task in completed_tasks:
                predictor_idx, data = await task
                if data is not None:
                    for idx, prompt in enumerate(data['prompt']):
                        with open(dest_file, 'a+') as f:
                            item = {'prompt': prompt, 'img_path': data['img_path'][idx], 'response': data['chosen'][idx], 'score': data['score'][idx]}
                            f.write(json.dumps(item) + '\n')

                else:
                    num_fail += 1
                    print(f"Number of failing batch: {num_fail}", flush=True)

                predictor_queue.put(predictor_idx)
                num_done += 1
                score_bar.set_description(f"Scoring {num_done}/{num_submit}")
                score_bar.update(len(data['score']))
                
    return all_data


def evaluate(
    model_args, training_args, data_args,
    predictor_class=InternVL2RewardPredictor,
    num_gpus_per_predictor: int = 1,
    data_path: str = "",
    batch_size: int = 8,
    dest_file: str = ""
):
    global lazy_dataset

    # Initialize a Ray cluster
    ray.init(ignore_reinit_error=True)
    num_gpus = int(ray.cluster_resources()['GPU'])
    # Create predictor instances
    predictors = [
        predictor_class.options(num_gpus=num_gpus_per_predictor).remote(i, model_args, training_args, data_args)
        for i in range(num_gpus // num_gpus_per_predictor)
    ]
    predictor_queue = Queue()
    for i in range(len(predictors)):
        predictor_queue.put(i)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    num_image_token = int((data_args.force_image_size // config.vision_config.patch_size) ** 2 * (data_args.down_sample_ratio ** 2))
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

    scored_set = set()
    if os.path.exists(dest_file):
        with open(dest_file, 'r') as f:
            for line in f.readlines():
                item = json.loads(line)
                scored_set.add(item['prompt'] + " " + item['img_path'] + ' ' + item['response'])

    lazy_dataset = LazySequenceDatasetForInference(data_args.conv_style, tokenizer, data_path, num_image_token=num_image_token, image_size=data_args.force_image_size, 
                pad2square=data_args.pad2square, dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail, use_image_only=training_args.use_image_only, use_text_only=training_args.use_text_only,
                min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch, normalize_type=data_args.normalize_type)
    dataset = lazy_dataset.dataset
    # filter processed prompts
    dataset = [item for item in dataset if item['prompt'] + " " + item['img_path'] + ' ' + item['chosen'] not in scored_set]
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    print(len(batches), flush=True)
    all_data = asyncio.run(get_rewards(predictors, predictor_queue, batches, dest_file))

    return all_data


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, DataTrainingArguments)
    )
    (
        model_args,
        training_args,
        data_args
    ) = parser.parse_args_into_dataclasses()
    
    evaluate(model_args, training_args, data_args, num_gpus_per_predictor=int(os.environ['PER_GPUS']), data_path = os.environ['DATA_PATH'], dest_file = os.environ['DEST_FILE'])
        