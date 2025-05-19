from torchvision.transforms.functional import InterpolationMode
import transformers
from models.constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        SIGLIP_MEAN, SIGLIP_STD)
import torchvision.transforms as T
from PIL import Image
import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer import TRAINER_STATE_NAME, logger
import numpy as np

def replace_llama_rmsnorm_with_fused_rmsnorm():
    try:
        from functools import partial

        from apex.normalization import FusedRMSNorm
        LlamaRMSNorm = partial(FusedRMSNorm, eps=1e-6)   # noqa
        transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        print('Discovered apex.normalization.FusedRMSNorm - will use it instead of LlamaRMSNorm')
    except ImportError:
        # using the normal LlamaRMSNorm
        pass
    except Exception:
        print('discovered apex but it failed to load, falling back to LlamaRMSNorm')
        pass

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def build_transform(input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError

    if pad2square is False:  # now we use this transform function by default
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_process_image(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _save_checkpoint(self, model, trial, metrics=None):
    # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    # want to save except FullyShardedDDP.
    # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    # Save model checkpoint
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    if self.hp_search_backend is None and trial is None:
        self.store_flos()

    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        logger.warning(
            f"Checkpoint destination directory {output_dir} already exists and is non-empty. "
            "Saving will proceed but saved results may be invalid."
        )
        staging_output_dir = output_dir
    else:
        staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
    self.save_model(staging_output_dir, _internal_call=True)

    if not self.args.save_only_model:
        # Save optimizer and scheduler
        self._save_optimizer_and_scheduler(staging_output_dir)
        # Save RNG state
        self._save_rng_state(staging_output_dir)

    # Determine the new best metric / best model checkpoint
    if metrics is not None and self.args.metric_for_best_model is not None:
        metric_to_check = self.args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics[metric_to_check]

        operator = np.greater if self.args.greater_is_better else np.less
        if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
        ):
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = output_dir

    # Save the Trainer state
    if self.args.should_save:
        self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

    if self.args.push_to_hub:
        self._push_from_checkpoint(staging_output_dir)

    # Place checkpoint in final location after all saving is finished.
    # First wait for everyone to finish writing
    self.args.distributed_state.wait_for_everyone()

    print("save on each node, ", self.args.save_on_each_node)
    print("is world process zero, ", self.is_world_process_zero())
    print("process index: ", self.args.process_index)
    # Then go through the rewriting process, only renaming and rotating from main process(es)
    if self.is_local_process_zero() if self.args.save_on_each_node else self.is_world_process_zero():
        if staging_output_dir != output_dir:
            if os.path.exists(staging_output_dir):
                os.rename(staging_output_dir, output_dir)

                # Ensure rename completed in cases where os.rename is not atomic
                # And can only happen on non-windows based systems
                if os.name != "nt":
                    fd = os.open(output_dir, os.O_RDONLY)
                    os.fsync(fd)
                    os.close(fd)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    self.args.distributed_state.wait_for_everyone()