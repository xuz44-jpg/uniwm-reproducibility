import os
import json
import pickle
import torch
import importlib.metadata
import re
import random

from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
from evo.core import metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from transformers import Trainer, Seq2SeqTrainer

from transformers.utils import is_peft_available
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)

import time
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from packaging import version
from torch import nn

from PIL import Image

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.file_utils import is_datasets_available

from transformers.trainer_utils import EvalPrediction

from transformers.trainer_utils import PredictionOutput, speed_metrics

from scripts.training_arguments import WrappedSeq2SeqTrainingArguments
from scripts.postprocess_logits_utils import split_token_sequence
from scripts.action_utils import generate_bin_tokens, extract_bin_values, DATASET_RANGES, DEFAULT_RANGES, action_to_text
from uniwm.memory_bank import MemoryBankAnoleForConditionalGeneration
from uniwm.extra_tokens import unwrap as unwrap_hf
from uniwm.pose_memory import MemoryEncoder, PoseMemory, splice_memory, MEM_PLACEHOLDER
from scripts.prompt_builder import build_action_prompt, build_viz_prompt
from scripts.metrics import coords_to_evo_traj, eval_ate_rpe, ImageMetricsCalculator


_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


from torch.utils.data import DataLoader, Dataset
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    EvalLoopOutput,
    PredictionOutput,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    EvalLoopContainer,
    IterableDatasetShard,
    find_batch_size
)
from transformers.utils import (
    is_peft_available,
    logging,
)


logger = logging.get_logger(__name__)

VERBOSE = os.environ.get("UNIWM_VERBOSE", "0") == "1"


def vprint(*a, **k):
    if VERBOSE:
        print(*a, **k)


class MetricEvalPrediction(NamedTuple):
    predictions: List[dict]
    items: List[dict]
    sketches: Union[List[str], np.ndarray]     # the file path for the sketch


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    sketches: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


if is_peft_available():
    from peft import PeftModel


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

def get_concat_h(im1, im2):
    # resize the image first wo the same height
    height = im1.height
    ratio = im1.height / im2.height
    im2_width = int(im2.width * ratio)
    im2 = im2.resize((im2_width, height))

    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

class MemoryBankManager:
    def __init__(self, model: nn.Module, trainer, use_memory_bank_inference: bool):
        self.model = model
        self.trainer = trainer # To access is_world_process_zero
        self.is_enabled = use_memory_bank_inference
        self.current_step = 0
    
    def is_world_process_zero(self):
        """Helper method to access the trainer's is_world_process_zero."""
        return self.trainer.is_world_process_zero()

    def setup_for_trajectory(self, traj_dirname: str):
        """Prepares the memory bank for a new trajectory."""
        if not self.is_enabled:
            return

        # Reset memory bank for each trajectory to ensure independence
        if hasattr(self.model, 'reset_memory_bank'):
            self.model.reset_memory_bank()
            if self.is_world_process_zero():
                vprint(f"  Intra-step memory bank reset for trajectory {traj_dirname}")
        elif hasattr(self.model, 'memory_bank_initialized'):
            # Fallback: manually reset memory bank state
            self.model.memory_bank_initialized = False
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                for layer in self.model.model.model.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'reset_memory_bank'):
                        layer.self_attn.reset_memory_bank()
            if self.is_world_process_zero():
                vprint(f"  Fallback intra-step memory bank reset for trajectory {traj_dirname}")
        
        # Reset global cross-step memory bank for each trajectory
        if hasattr(self.model, 'reset_global_memory_bank'):
            self.model.reset_global_memory_bank()
            if self.is_world_process_zero():
                vprint(f"  Cross-step memory bank reset for trajectory {traj_dirname}")
        
        # Enable global memory bank functionality
        if hasattr(self.model, 'enable_global_memory_bank'):
            self.model.enable_global_memory_bank()
            if self.is_world_process_zero():
                vprint(f"  Global memory bank enabled for trajectory {traj_dirname}")
        
        # Enable memory bank functionality if available
        if hasattr(self.model, 'enable_memory_bank'):
            self.model.enable_memory_bank()
            if self.is_world_process_zero():
                vprint(f"  Memory bank functionality enabled for trajectory {traj_dirname}")
        
        self.current_step = 0

    def start_new_step(self):
        """Prepares for a new step within a trajectory."""
        if not self.is_enabled:
            return
        
        self.current_step += 1
        if self.is_world_process_zero():
            vprint(f"\n=== Step {self.current_step} Action Prediction Substep ===")
        
        # Reset intra memory bank for action prediction substep (but keep cross memory bank)
        if hasattr(self.model, 'reset_memory_bank'):
            self.model.reset_memory_bank()
            if self.is_world_process_zero():
                vprint(f"  Step {self.current_step}: intra memory bank reset for action prediction")
    
    def get_action_kwargs(self, action_inputs, action_gen_kwargs, step):
        if not self.is_enabled:
            return action_gen_kwargs
        
        # Check for memory bank initialization (third pair of 8197 and 8196 tokens)
        input_ids_list = action_inputs['input_ids'][0].tolist()
        # Always try to initialize memory bank for each step (intra-step memory bank)
        if hasattr(self.model, 'initialize_memory_bank') and not getattr(self.model, 'memory_bank_initialized', False):
            
            # Count pairs of 8197 and 8196 tokens
            pairs_count = 0
            i = 0
            while i < len(input_ids_list) - 1:
                if input_ids_list[i] == 8197:
                    for j in range(i + 1, len(input_ids_list)):
                        if input_ids_list[j] == 8196:
                            pairs_count += 1
                            i = j
                            break
                    else:
                        break
                i += 1
              
            # Initialize memory bank if we have at least 3 pairs, or fallback to any image tokens
            should_initialize = False
            init_method = ""
            
            if pairs_count >= 3:
                should_initialize = True
                init_method = f"special token pairs (found {pairs_count})"
            
            if should_initialize:
                if self.is_world_process_zero():
                    vprint(f"  Step {step+1}: Initializing memory bank using {init_method}")

                # try:
                self.model.initialize_memory_bank(
                    input_ids=action_inputs['input_ids'],
                    pixel_values=action_inputs['pixel_values'],
                    attention_mask=action_inputs['attention_mask']
                )
                if self.is_world_process_zero():
                    vprint(f"  Step {step+1}: Memory bank initialization completed successfully")
                    # Print memory bank storage details if available
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                        for layer_idx, layer in enumerate(self.model.model.model.layers):
                            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'stored_keys'):
                                if layer.self_attn.stored_keys is not None:
                                    vprint(f"    - Layer {layer_idx}: Stored keys shape: {layer.self_attn.stored_keys.shape}")
                                    vprint(f"    - Layer {layer_idx}: Stored values shape: {layer.self_attn.stored_values.shape}")
                                    vprint(f"    - Layer {layer_idx}: Memory bank storage size: {layer.self_attn.stored_keys.numel() + layer.self_attn.stored_values.numel()} elements")
                # except Exception as e:
                #     if self.is_world_process_zero():
                #         vprint(f"  Warning: Memory bank initialization failed: {e}")
            else:
                if self.is_world_process_zero():
                    vprint(f"  Step {step+1}: Warning - No suitable tokens found for memory bank initialization")
        
        # Print memory bank usage details before generation
        if hasattr(self.model, 'memory_bank_initialized') and self.model.memory_bank_initialized:
            if self.is_world_process_zero():
                vprint(f"  Step {step+1}: Using memory bank for action generation")

                # Print stored K,V sizes if available
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                    for layer_idx, layer in enumerate(self.model.model.model.layers):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'stored_keys'):
                            if layer.self_attn.stored_keys is not None:
                                vprint(f"    - Layer {layer_idx}: Using stored K shape: {layer.self_attn.stored_keys.shape}")
                                vprint(f"    - Layer {layer_idx}: Using stored V shape: {layer.self_attn.stored_values.shape}")
        
        with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
            # Two-phase action generation with memory bank
            action_gen_kwargs_with_memory = action_gen_kwargs.copy()
            # Remove any existing memory bank parameters to avoid conflicts
            action_gen_kwargs_with_memory.pop('use_memory_bank', None)
            action_gen_kwargs_with_memory.pop('is_memory_bank_init', None)
            action_gen_kwargs_with_memory.pop('current_step', None)
            action_gen_kwargs_with_memory.pop('current_substep', None)
            action_gen_kwargs_with_memory.pop('use_global_memory_bank', None)
            
            # Use global memory bank if we have previous steps (current_step > 1)
            use_global_mb = self.current_step > 1
            
            # Phase 1: Initialize memory bank (dummy generation to extract K,V)
            init_kwargs = action_gen_kwargs_with_memory.copy()
            init_kwargs.update({
                'use_memory_bank': True,
                'is_memory_bank_init': True,  # Initialize memory bank
                'current_step': self.current_step,
                'current_substep': 'action',
                'use_global_memory_bank': False,  # Don't use global during init
                'max_new_tokens': 1  # Minimal generation for initialization
            })
            
            if self.is_world_process_zero():
                vprint(f"  Step {step+1}: Memory bank initialization phase")
            _ = self.model.generate(**action_inputs, **init_kwargs)
            
            # Phase 2: Actual action generation using initialized memory bank
            gen_kwargs = action_gen_kwargs_with_memory.copy()
            gen_kwargs.update({
                'use_memory_bank': True,
                'is_memory_bank_init': False,  # Use existing memory bank
                'current_step': self.current_step,
                'current_substep': 'action',
                'use_global_memory_bank': use_global_mb
            })
            
            if self.is_world_process_zero():
                vprint(f"  Step {step+1}: Action generation using memory bank (global: {use_global_mb})")
            return gen_kwargs
            
    def get_viz_kwargs(self, viz_inputs, viz_gen_kwargs, step):
        if not self.is_enabled:
            return viz_gen_kwargs
        if self.is_world_process_zero():
            vprint(f"\n=== Step {self.current_step} Visualization Substep ===")
        
        
        with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
            # Enable memory bank for visualization generation
            viz_gen_kwargs_with_memory = viz_gen_kwargs.copy()
            # Use global memory bank for visualization (always available since we're in step >= 1)
            use_global_mb_viz = self.current_step >= 1
            viz_gen_kwargs_with_memory.update({
                'use_memory_bank': True,
                'is_memory_bank_init': False,  # Use existing memory bank for visualization
                'current_step': self.current_step,
                'current_substep': 'visualization',
                'use_global_memory_bank': use_global_mb_viz
            })
            return viz_gen_kwargs_with_memory

    def store_step_memory(self):
        """Stores the current step's K,V pairs into the global memory bank."""
        if not self.is_enabled:
            return
            
        # Store current step's intra-step K,V to global cross-step memory bank
        # This happens after both action prediction and visualization substeps are completed
        if hasattr(self.model, 'store_to_global_memory_bank'):
            self.model.store_to_global_memory_bank(self.current_step)
            if self.is_world_process_zero():
                vprint(f"  Step {self.current_step}: Stored intra-step K,V to global memory bank")
    
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                    for layer_idx, layer in enumerate(self.model.model.model.layers):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'global_stored_keys'):
                            if len(layer.self_attn.global_stored_keys) > 0:
                                vprint(f"    - Layer {layer_idx}: Global memory bank now has {len(layer.self_attn.global_stored_keys)} steps")
                                vprint(f"    - Layer {layer_idx}: Latest stored K shape: {layer.self_attn.global_stored_keys[-1].shape}")
                                vprint(f"    - Layer {layer_idx}: Latest stored V shape: {layer.self_attn.global_stored_values[-1].shape}")
                                break  # Only print for first layer to avoid spam



class CustomizeSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
            self, 
            evaluator,
            *args: WrappedSeq2SeqTrainingArguments,
            eval_examples: Optional[Dataset] = None,
            ignore_pad_token_for_loss: bool = True,
            wandb_run_dir: Optional[str] = None,
            image_loss_func: Optional[torch.nn.Module],
            action_cfg: Optional[Dict] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.wandb_run_dir = wandb_run_dir

        self.image_loss_func = image_loss_func

        if action_cfg:
            self.action_cfg = action_cfg
        else:
            print("[WARNING] action_cfg not provided to Trainer. Using default values for loss calculation.")
            self.action_cfg = {
                'min_dxy': -2.46, 'max_dxy': 2.46,
                'min_dyaw': -2.82, 'max_dyaw': 2.82,
                'bin_step': 0.01
            }
    
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            eval_examples: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs,
            # max_length: Optional[int] = None,
            # max_time: Optional[int] = None,
            # num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        gen_kwargs = gen_kwargs.copy()

        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_new_tokens is not None
        ):
            gen_kwargs["max_new_tokens"] = self.args.generation_max_new_tokens
        
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        
        if hasattr(self.args, 'customize_gen_stopping_criteria'):
            if self.args.customize_gen_stopping_criteria:
                gen_kwargs['stopping_criteria'] = self.args.customize_gen_stopping_criteria
        
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        


        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                output.predictions,
                "eval_{}".format(self.state.epoch)
            )
            summary = self.compute_metrics(eval_preds, section="dev", finish=True)
            # output.metrics.update(summary)
            if summary is not None:
                output.metrics.update(summary)
            else:
                print("[Warning] compute_metrics returned None — skipping metric update.")

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    def predict(
            self,
            test_dataset: Optional[Dataset],
            test_examples: Optional[Dataset],
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            predict_task: Optional[str] = None,
            **gen_kwargs
            # max_length: Optional[int] = None,
            # max_time: Optional[int] = None,
            # num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        gen_kwargs = gen_kwargs.copy()

        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_new_tokens is not None
        ):
            gen_kwargs["max_new_tokens"] = self.args.generation_max_new_tokens

        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        
        if test_examples and test_examples[0].get('train_task') == 'task_level_evaluation':
            if predict_task == "task_level_evaluation":
                metrics = self.task_level_evaluation_loop(test_dataset, test_examples, metric_key_prefix)
            if predict_task == "rollout_evaluation":
                metrics = self.run_gt_action_rollout(test_dataset, test_examples, metric_key_prefix)
            if predict_task == "action_probe":
                metrics = self.run_action_probe(test_dataset, test_examples, metric_key_prefix)
            if predict_task == "action_likelihood":
                metrics = self.run_action_likelihood(test_dataset, test_examples, metric_key_prefix)
            if predict_task == "action_intervention":
                metrics = self.run_action_intervention(test_dataset, test_examples, metric_key_prefix)
            if predict_task == "variance_decomp":
                metrics = self.run_variance_decomp(test_dataset, test_examples, metric_key_prefix)
            return PredictionOutput(predictions=None, sketches=None, label_ids=None, metrics=metrics)

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics


        if self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                test_examples,
                output.predictions,
                metric_key_prefix
            )
            output.metrics.update(self.compute_metrics(eval_preds, section="test", finish=True))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def _post_process_function(
            self, 
            examples: Dataset, 
            predictions: np.ndarray, 
            # sketches: np.ndarray, 
            stage: str
    ) -> MetricEvalPrediction:
        # assert isinstance(examples, Dataset)
        tokens = []
        sketches = []
        for r_ids in predictions:
            generated_results = split_token_sequence(
                tokens=torch.tensor(r_ids).unsqueeze(0).to(self.model.device), 
                image_seq_length=self.model.image_token_num,
                boi=self.model.config.boi_token_id, 
                eoi=self.model.config.eoi_token_id,
                max_length=predictions.shape[-1],
                pad_token_id=self.model.config.pad_token_id
            )
            tokens.append(generated_results['texts'])
            if generated_results["images"]:
                generated_imgs = torch.cat(generated_results["images"], dim=0).to(self.model.device)
                generated_imgs = self.model.decode_image_tokens(generated_imgs)
                generated_imgs = self.tokenizer.postprocess_pixel_values(generated_imgs)
            else:
                generated_imgs = None
            sketches.append(generated_imgs)
        
        predictions = torch.cat(tokens, dim=0)

        predictions[predictions == -100] = self.tokenizer.tokenizer.pad_token_id

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)

        predictions = [i.split("<reserved08706>")[0] for i in predictions]

        sketch_dir = f"{self.args.output_dir}/sketch_{stage}"

        # Save locally.
        if self.is_world_process_zero():
            with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(
                        **{
                            "idx": examples[idx]['idx'],
                            "prediction": predictions[idx],
                            "text": examples[idx]["input_text"],
                            "labels": examples[idx]['label_text'],
                            "predicted_sketch_paths": [os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png") for i in range(len(sketches[idx]))] if sketches[idx] is not None else None, 
                            "label_img_paths": examples[idx]['label_img_paths'],
                            "input_img_paths": examples[idx]['input_img_paths']
                        }
                        ) for idx in range(len(examples))],
                    f,
                    indent=4,
                )
        
            if stage.startswith("eval"):
                
                if not os.path.exists(sketch_dir):
                    os.mkdir(sketch_dir)
                
                sketch_files = []
                for idx in range(len(examples)):
                    sketches_per_item = sketches[idx]
                    sketch_files_per_item = []
                    if sketches_per_item is not None:
                        for i in range(len(sketches_per_item)):
                            file_path = os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png")
                            tensor_img = sketches_per_item[i, :, :, :]

                            np_img = tensor_img.cpu().detach().to(torch.uint8).numpy()
                            np_img = np.transpose(np_img, (1, 2, 0))  # (H, W, C)

                            img = Image.fromarray(np_img.astype(np.uint8))

                            if len(examples[idx]["label_imgs"]) != 0:
                                concat_img = get_concat_h(im1=examples[idx]["label_imgs"][-1], im2=img)
                                concat_img = get_concat_h(im1=examples[idx]['input_imgs'][-1], im2=concat_img)
                            else:
                                concat_img = img
                            concat_img.save(file_path)

                            sketch_files_per_item.append(file_path)

                    sketch_files.append(sketch_files_per_item)

        # Save to wandb.
        if self.wandb_run_dir and self.is_world_process_zero():
            with open(f"{self.wandb_run_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(
                        **{
                            "idx": examples[idx]['idx'],
                            "prediction": predictions[idx],
                            "text": examples[idx]["input_text"],
                            "labels": examples[idx]['label_text'],
                            "predicted_sketch_paths": [os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png") for i in range(len(sketches[idx]))] if sketches[idx] is not None else None, 
                            "label_img_path": examples[idx]['label_img_paths'],
                            "input_img_path": examples[idx]['input_img_paths']
                        }
                        ) for idx in range(len(examples))],
                    f,
                    indent=4,
                )
        if not self.is_world_process_zero():
            sketch_files = []
            for idx in range(len(examples)):
                sketches_per_item = sketches[idx]
                sketch_files_per_item = []
                if sketches_per_item is not None:
                    for i in range(len(sketches_per_item)):
                        file_path = os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png")
                        sketch_files_per_item.append(file_path)
                sketch_files.append(sketch_files_per_item)

        return MetricEvalPrediction(predictions=predictions, sketches=sketches, items=[examples[idx] for idx in range(len(examples))])

    def _compute_metrics(self, eval_prediction: MetricEvalPrediction, section, finish=False) -> dict:
        return self.evaluator.evaluate(eval_prediction.predictions, eval_prediction.items, eval_prediction.sketches, section, finish=finish)


    def _loss_index_tensors(self, anole, device):
        """Token-id tensors used by the auxiliary losses (built once per device)."""
        cache = getattr(self, "_loss_ids", None)
        if cache is not None and cache["device"] == device:
            return cache
        tok = self.tokenizer.tokenizer
        bins = {}
        for name, lo, hi in (("dx", self.action_cfg['min_dxy'], self.action_cfg['max_dxy']),
                             ("dy", self.action_cfg['min_dxy'], self.action_cfg['max_dxy']),
                             ("dyaw", self.action_cfg['min_dyaw'], self.action_cfg['max_dyaw'])):
            ids = tok.convert_tokens_to_ids(generate_bin_tokens(name, lo, hi, self.action_cfg['bin_step']))
            bins[name] = torch.tensor(sorted(set(i for i in ids if i is not None and i >= 0)), device=device, dtype=torch.long)
        cache = {"device": device, "bins": bins,
                 "bpe": torch.tensor(anole.bpe_indices, device=device, dtype=torch.long)}
        self._loss_ids = cache
        return cache

    def compute_loss(self, model, inputs, return_outputs=False):
        """Label-smoothed CE (text + image tokens) + action-bin CE + codebook discrepancy loss.

        Logits are computed only at supervised positions (see `loss_positions` in custom_chameleon.py).
        """
        labels = inputs.pop("labels", None)
        inputs.pop("img_label", None)
        if labels is None:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

        # position t predicts labels[t + 1]
        shift_labels = labels[:, 1:]
        keep = shift_labels != -100
        loss_positions = torch.cat([keep, torch.zeros_like(keep[:, :1])], dim=1)
        outputs = model(**inputs, loss_positions=loss_positions)
        logits = outputs.logits                                    # [N, vocab], fp32
        targets = shift_labels[keep]                               # [N]

        # label-smoothed cross-entropy (same formula as transformers.trainer_pt_utils.LabelSmoother)
        eps = float(self.args.label_smoothing_factor or 0.0)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, targets[:, None]).squeeze(-1)
        loss = ((1.0 - eps) * nll - eps * log_probs.mean(-1)).mean()

        unwrapped_model = self.accelerator.unwrap_model(model)
        anole = unwrap_hf(unwrapped_model)
        ids = self._loss_index_tensors(anole, logits.device)
        should_log = self.state.global_step == self._globalstep_last_logged and self.state.global_step != 0

        # action-bin loss (dx / dy / dyaw tokens of the action-reasoning samples)
        comps = []
        for name in ("dx", "dy", "dyaw"):
            mask = torch.isin(targets, ids["bins"][name])
            if bool(mask.any()):
                comps.append(nn.CrossEntropyLoss()(logits[mask], targets[mask]))
        if comps:
            bc_loss = sum(comps) / len(comps)
            loss = loss + bc_loss
            if should_log:
                extra = {"bc_loss": float(bc_loss)}
                scaler = getattr(self.accelerator, "scaler", None)
                if scaler is not None:
                    extra["loss_scale"] = float(scaler.get_scale())
                self.log(extra)

        # codebook discrepancy loss: expected codebook distance between the predicted image-token distribution and the GT code
        if self.image_loss_func:
            image_mask = torch.isin(targets, ids["bpe"])
            if bool(image_mask.any()):
                image_probs = torch.nn.functional.softmax(logits[image_mask][:, ids["bpe"]], dim=-1)
                vis_img_tokens = anole.model.convert_bpe2img_tokens(targets[image_mask]).reshape(-1).to(torch.long)
                sim = anole.codebook_sim_matrix.to(logits.device)[vis_img_tokens]
                discrepancy_loss = torch.mean(torch.sum(sim.float() * image_probs, -1))
                loss = loss + discrepancy_loss
                if should_log:
                    self.log({"discrepancy_loss": float(discrepancy_loss)})

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()

        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        if "pixel_values" in inputs:
            model_dtype = next(model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

            if inputs["pixel_values"].ndim == 5:
                # [1, 3, 3, 448, 448] -> [1, 3, 448, 448]
                inputs["pixel_values"] = inputs["pixel_values"][:, 0, :, :, :]

        if hasattr(self.model, "image_decoder"):
            generated_tokens, generated_sketch = self.model.generate(**inputs, **gen_kwargs)
        else:
            generated_tokens = self.model.generate(**inputs, **gen_kwargs)
            generated_sketch = None
        

        if hasattr(self.model, "image_postprocess"):
            if self.model.image_postprocess and generated_sketch is not None:
                generated_sketch["sketch"] = self.tokenizer.postprocess_pixel_values(generated_sketch["sketch"])
        
        if hasattr(generated_tokens, "sequences"):
            generated_tokens = generated_tokens.sequences

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                loss = self.compute_loss(
                    model=self.model,
                    inputs=inputs,
                    return_outputs=False
                )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        if generated_sketch is not None:
            return loss, (generated_tokens, generated_sketch), labels
        else:
            return loss, generated_tokens, labels
    

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # comment it for now
            # if is_torch_xla_available():
            #     xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)
        
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # calculate the metric in a post-hoc way
            if self.args.batch_eval_metrics:

                del losses, logits, labels, inputs

                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs

                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        if metrics is None:
            metrics = {}

        # # Compute metrics if a compute_metrics function is provided
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        #     metrics = self.compute_metrics(
        #         EvalPrediction(predictions=all_preds, label_ids=all_labels)
        #     )

        return EvalLoopOutput(
            predictions=all_preds, 
            label_ids=all_labels, 
            metrics=metrics, 
            num_samples=num_samples
            )

    def _sanitize_generated_image(self, pil_img: Image) -> Image:
        if pil_img is None:
            return None
        inputs = self.tokenizer(text="<image>", images=pil_img, return_tensors="pt").to(self.args.device)
        pixel_values = inputs['pixel_values'].to(self.model.dtype)
        
        with torch.no_grad():
            img_tokens = self.model.model.model.get_image_tokens(pixel_values)
            reconstructed_pixels = self.model.model.decode_image_tokens(img_tokens)
        
        processed_pixels = self.tokenizer.postprocess_pixel_values(reconstructed_pixels)[0]
        np_img = processed_pixels.cpu().numpy().transpose(1, 2, 0)
        sanitized_img = Image.fromarray(np_img.astype(np.uint8))
        
        return sanitized_img

    def _parse_and_save_image(self, generated_tokens, model, tokenizer, save_path):
        if not isinstance(generated_tokens, torch.Tensor):
            raise TypeError(f"generated_tokens must be a torch.Tensor, but got {type(generated_tokens)}")

        if generated_tokens.dim() == 1:
            tokens_for_split = generated_tokens.unsqueeze(0)
        elif generated_tokens.dim() == 2:
            if generated_tokens.shape[0] != 1:
                tokens_for_split = generated_tokens[0].unsqueeze(0)
            else:
                tokens_for_split = generated_tokens
        else:
            raise ValueError(f"Unsupported shape for generated_tokens: {generated_tokens.shape}. Expected 1D or 2D tensor.")
        
        try:
            generated_output = split_token_sequence(
                tokens=tokens_for_split,
                image_seq_length=model.image_token_num,
                boi=model.config.boi_token_id,
                eoi=model.config.eoi_token_id,
                max_length=tokens_for_split.shape[-1],
                pad_token_id=tokenizer.tokenizer.pad_token_id,
            )
            predicted_image_tokens = generated_output['images']
        except AssertionError:  # an image segment of the wrong length (generation cut off mid-image)
            predicted_image_tokens = []

        if predicted_image_tokens and len(predicted_image_tokens) > 0:
            with torch.no_grad():
                decoded_img = model.decode_image_tokens(predicted_image_tokens[0])
            processed_img = tokenizer.postprocess_pixel_values(decoded_img)[0]
            np_img = processed_img.cpu().numpy().transpose(1, 2, 0)
            new_obs_img = Image.fromarray(np_img.astype(np.uint8))
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                new_obs_img.save(save_path)
            return new_obs_img
        print(f"  Generated failed visualization tokens: {generated_tokens}")
        return None

    
    def _pairwise_lpips(self, images):
        """Symmetric matrix of LPIPS (AlexNet) distances between PIL images, at 256x256."""
        if getattr(self, "_lpips", None) is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.args.device).eval()
        import torchvision.transforms as T
        t = torch.stack([T.ToTensor()(im.convert("RGB").resize((256, 256))) for im in images]).to(self.args.device) * 2 - 1
        n = len(images); D = np.zeros((n, n), dtype=np.float32)
        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    D[i, j] = D[j, i] = float(self._lpips(t[i:i + 1], t[j:j + 1]).item())
        return D

    def _pose_memory(self, model):
        """PoseMemory for closed-loop evaluation (None when --memory none)."""
        mode = getattr(self.args, "memory", "none")
        if mode == "none":
            return None
        if getattr(self, "_pose_mem", None) is None:
            self._pose_mem = PoseMemory(MemoryEncoder(model, self.tokenizer, res=getattr(self.args, "mem_res", 160)),
                                        k=getattr(self.args, "mem_k", 2), mode=mode, text=getattr(self.args, "mem_text", "pose"),
                                        ablate_image=(Image.open(self.args.mem_ablate_image).convert("RGB") if getattr(self.args, "mem_ablate_image", "") else None))
        return self._pose_mem

    def _prompt_inputs(self, text, images, mem_tokens):
        """Tokenize a prompt with its 3 main images and splice memory frames at the placeholder token."""
        # truncation=False: the tokenizer saved with the checkpoint carries max_length=3050 and would otherwise
        # silently truncate long prompts ('longest_first'), cutting off the memory block at the end
        inputs = self.tokenizer(text=[text], images=images, return_tensors="pt", padding=False, truncation=False).to(self.args.device)
        if mem_tokens or MEM_PLACEHOLDER in text:
            tok = self.tokenizer.tokenizer
            inputs["input_ids"] = splice_memory(inputs["input_ids"], mem_tokens, tok.convert_tokens_to_ids(MEM_PLACEHOLDER),
                                                self.model.config.boi_token_id, self.model.config.eoi_token_id)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs

    def _process_single_trajectory(self, raw_item: Dict, output_dir: str, model: Any,
                                   action_gen_kwargs: Dict, viz_gen_kwargs: Dict,
                                   manager: MemoryBankManager) -> Dict[str, Any]:
        # --- 1. Initialization ---
        traj_dirname = os.path.basename(os.path.dirname(raw_item['input_img_paths'][0]))
        if self.is_world_process_zero():
            vprint(f"\n--- Processing Trajectory {traj_dirname} ---")
        
        manager.setup_for_trajectory(traj_dirname)

        traj_output_dir = os.path.join(output_dir, traj_dirname)
        os.makedirs(traj_output_dir, exist_ok=True)

        start_pose = raw_item['coords'][0]
        current_pose = list(start_pose)  # [x, y, yaw]
        predicted_coords = [list(start_pose)]

        gt_coords_arr = np.array(raw_item['coords'])
        gt_goal_pos = gt_coords_arr[-1, :2] if len(gt_coords_arr) > 1 else None

        start_img = raw_item['input_imgs'][0].convert("RGB").resize((256, 256))
        goal_img = raw_item['input_imgs'][1].convert("RGB").resize((256, 256))
        current_observation = self._sanitize_generated_image(start_img)
        
        start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"
        pose_mem = self._pose_memory(model)
        if pose_mem is not None:
            pose_mem.reset(); pose_mem.add(current_pose, current_observation)

        # --- 2. Step-by-step Loop ---
        current_trajectory_steps = []
        max_steps = int(getattr(self.args, 'eval_max_steps', 10) or 10)   # paper protocol: 10; 1x_nav trajectories are up to 24 steps

        all_actions = []
        all_decoded = []

        def _get_dataset_ranges(image_path):
            """ Determine the dataset ranges based on the image path."""
            for name, ranges in DATASET_RANGES.items():
                if f"/{name}/" in image_path.replace("\\", "/"):
                    return ranges
            return DEFAULT_RANGES

        ranges = _get_dataset_ranges(raw_item['input_img_paths'][0])
        vprint(f"  Using ranges: {ranges} for trajectory {raw_item['input_img_paths'][0]}")
        dxy_range, dyaw_range = ranges['dxy'], ranges['dyaw']
        

        for step in range(max_steps):
            step_log = {"step": step + 1}

            manager.start_new_step()
            
            # --- Action Prediction ---
            mem_text, mem_tokens = pose_mem.query(current_pose) if pose_mem is not None else ("", [])
            action_prompt = build_action_prompt(
                start_pose_str=start_pose_str,
                dxy_range=dxy_range,
                dyaw_range=dyaw_range,
                memory_block=mem_text,
            )
            action_inputs = self._prompt_inputs(action_prompt, [start_img, goal_img, current_observation], mem_tokens)
            action_gen_kwargs = manager.get_action_kwargs(action_inputs, action_gen_kwargs, step)
            if getattr(self.args, "action_greedy", False):
                action_gen_kwargs = {**action_gen_kwargs, "do_sample": False}

            n_samples = max(1, int(getattr(self.args, "action_samples", 1) or 1))
            candidates = []   # (decoded_text, is_stop, (dx, dy, dyaw) or None)
            bin_step = self.action_cfg['bin_step']
            for _ in range(n_samples):
                with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                    action_outputs = model.generate(**action_inputs, **action_gen_kwargs)
                cand = self.tokenizer.batch_decode(action_outputs[0], skip_special_tokens=False)[0].strip()
                # the decoded text keeps the end-of-turn token ("Stop<reserved08706>"); strip special tokens before comparing
                cand_text = re.sub(r"<reserved\d+>|</s>|<pad>", "", cand).strip()
                cand_stop = cand_text.lower().startswith("stop")
                cand = re.sub(r'(<d[^>]+>)+(<d[^>]+>)', r'\2', cand)
                vals = None
                if not cand_stop:
                    ok = all(re.search(rf"<{k}_(pos|neg)_bin_\d+>", cand) for k in ("dx", "dy", "dyaw"))
                    vals = tuple(extract_bin_values(cand, k, bin_step) for k in ("dx", "dy", "dyaw")) if ok else None
                candidates.append((cand, cand_stop, vals))

            # aggregate: Stop on majority; otherwise the component-wise median of the well-formed candidates
            # (a single sample reduces to the released behaviour: its own value, malformed -> zeros)
            decoded = candidates[0][0]
            n_stop = sum(c[1] for c in candidates)
            valid = [c[2] for c in candidates if c[2] is not None]
            is_stop = n_stop * 2 > n_samples if n_samples > 1 else candidates[0][1]
            all_decoded.append(decoded)
            dx, dy, dyaw = 0.0, 0.0, 0.0
            if not is_stop:
                if n_samples == 1:
                    if candidates[0][2] is not None:
                        dx, dy, dyaw = candidates[0][2]
                    else:   # malformed single sample: the released parser's per-component fallback (missing -> 0)
                        dx, dy, dyaw = (extract_bin_values(decoded, k, bin_step) for k in ("dx", "dy", "dyaw"))
                elif valid:
                    arr = np.array(valid)
                    dx, dy, dyaw = (float(v) for v in np.median(arr, axis=0))
                    dx, dy, dyaw = round(dx, 2), round(dy, 2), round(dyaw, 2)
                    # disagreement gate: when the candidates disagree, take a shorter step
                    gate_std = float(getattr(self.args, "gate_std", 0.0) or 0.0)
                    if gate_std > 0 and len(valid) > 1 and float(np.std(arr, axis=0).max()) > gate_std:
                        g = float(getattr(self.args, "gate_scale", 0.5))
                        dx, dy, dyaw = round(dx * g, 2), round(dy * g, 2), round(dyaw * g, 2)
                        step_log["gated"] = True
                action_string = f"dx: {dx}, dy: {dy}, dyaw: {dyaw}"
            else:
                action_string = "Stop"
            if n_samples > 1:
                step_log["candidates"] = [c[0] for c in candidates]
                step_log["n_stop"] = int(n_stop)
                step_log["n_malformed"] = int(sum(1 for c in candidates if not c[1] and c[2] is None))
                step_log["action_std"] = [float(v) for v in np.std(np.array(valid), axis=0)] if len(valid) > 1 else None

            step_log["predicted_action"] = action_string
            if self.is_world_process_zero():
                vprint(f"  Step {step+1}: Predicted Action: '{action_string}'")

            all_actions.append(action_string)

            current_pose[0] += dx
            current_pose[1] += dy
            current_pose[2] += dyaw
            predicted_coords.append(list(current_pose))

            # Termination condition
            should_stop = is_stop or (dx == 0.0 and dy == 0.0 and dyaw == 0.0)
            if should_stop or step == max_steps - 1:
                current_trajectory_steps.append(step_log)
                if self.is_world_process_zero():
                    vprint("  Stop action predicted or max steps reached. Ending trajectory.")
                break

            # --- Visualization Generation ---
            
            # feed the canonical action text (as in training), never the raw decoded string, which can be
            # hundreds of garbage tokens for an untrained prompt format and would overflow the context
            viz_prompt = build_viz_prompt(
                decoded_action=action_to_text([dx, dy, dyaw]),
                start_pose_str=start_pose_str,
                memory_block=mem_text,
            )
            viz_inputs = self._prompt_inputs(viz_prompt, [start_img, goal_img, current_observation], mem_tokens)
                      
            viz_gen_kwargs = manager.get_viz_kwargs(viz_inputs, viz_gen_kwargs, step)
                
            save_path = os.path.join(traj_output_dir, f"step_{step+1}_observation.png")
            n_frames = max(1, int(getattr(self.args, "frame_samples", 1) or 1))
            if n_frames == 1:
                with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                    viz_outputs = model.generate(**viz_inputs, **viz_gen_kwargs)
                new_obs_img = self._parse_and_save_image(viz_outputs[0], model, self.tokenizer, save_path)
            else:
                # frame-level self-consistency: sample several imagined frames and keep the medoid (the frame with
                # the smallest mean LPIPS distance to the others); the mean pairwise distance is logged as the
                # frame disagreement of the step
                frame_cands = []
                for _ in range(n_frames):
                    with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                        viz_outputs = model.generate(**viz_inputs, **viz_gen_kwargs)
                    img = self._parse_and_save_image(viz_outputs[0], model, self.tokenizer, None)
                    if img is not None:
                        frame_cands.append(img)
                new_obs_img = None
                step_log["frame_failed"] = n_frames - len(frame_cands)
                if frame_cands:
                    D = self._pairwise_lpips(frame_cands)
                    sel = int(np.argmin(D.sum(1))) if len(frame_cands) > 1 else 0
                    new_obs_img = frame_cands[sel]
                    new_obs_img.save(save_path)
                    step_log["frame_selected"] = sel
                    step_log["frame_disagreement"] = float(D[np.triu_indices(len(frame_cands), 1)].mean()) if len(frame_cands) > 1 else None
            
            step_log["predicted_observation_path"] = save_path if new_obs_img else "GENERATION_FAILED"
            current_trajectory_steps.append(step_log)

            if not new_obs_img:
                if self.is_world_process_zero():
                    logger.warning("  Visualization failed to generate an image. Ending trajectory.")
                break
            
            current_observation = self._sanitize_generated_image(new_obs_img)
            if pose_mem is not None:
                pose_mem.add(current_pose, current_observation)

            manager.store_step_memory()
            
        # --- 3. Trajectory Evaluation ---
        final_position = np.array(current_pose[:2])
        distance_to_goal = np.linalg.norm(final_position - gt_goal_pos) if gt_goal_pos is not None else None

        pred_coords_arr = np.array(predicted_coords)
        ate, rpe_trans, rpe_rot = float('nan'), float('nan'), float('nan')
        if len(pred_coords_arr) >= 2:
            try:
                pred_traj_evo = coords_to_evo_traj(pred_coords_arr)
                gt_traj_evo = coords_to_evo_traj(gt_coords_arr)
                ate, rpe_trans, rpe_rot = eval_ate_rpe(gt_traj_evo, pred_traj_evo)
            except Exception as e:
                print(f"Warning: ATE/RPE calculation failed for {traj_dirname} with error: {e}")
        
        if self.is_world_process_zero():
            print(f"--- Trajectory {traj_dirname} finished. ATE: {ate:.3f}, RPE_trans: {rpe_trans:.3f}, RPE_rot: {rpe_rot:.3f} ---")

        # --- 4. Save and Return ---
        final_eval_record = {
            "final_position": final_position.tolist(),
            "goal_position": gt_goal_pos.tolist() if gt_goal_pos is not None else None,
            "distance_to_goal": distance_to_goal,
            "ate": ate,
            "rpe_trans": rpe_trans,
            "rpe_rot": rpe_rot,
            "trajectory_actions": all_actions,
            "trajectory_decoded": all_decoded,
            "reference_coords": gt_coords_arr.tolist(),      # reference poses (x, y, yaw) of the test trajectory
            "predicted_coords": pred_coords_arr.tolist(),    # integrated predicted poses, start pose first
            "steps": current_trajectory_steps,               # per-step log (candidates / disagreement when --action_samples > 1)
        }
        
        eval_file_path = os.path.join(traj_output_dir, "eval_result.json")
        with open(eval_file_path, "w") as f_eval:
            json.dump(final_eval_record, f_eval, indent=4)

        summary_log = {
            "trajectory_id": traj_dirname,
            "input_start_img_path": raw_item['input_img_paths'][0],
            "input_goal_img_path": raw_item['input_img_paths'][1],
            "ground_truth_goal_path": raw_item['label_img_paths'][0],
            "steps": current_trajectory_steps
        }
        
        return {**final_eval_record, "summary_log": summary_log}


    def task_level_evaluation_loop(self, test_dataset: Dataset, test_examples: Dataset, metric_key_prefix: str = "eval") -> Dict[str, float]:
        logger.info("\n***** Running Task-Level Trajectory Generation *****")

        # --- 1. Setup ---
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        manager = MemoryBankManager(
            model=model, 
            trainer=self, 
            use_memory_bank_inference=self.args.use_memory_bank_inference
        )
        # Enable memory bank if the model supports it
        if manager.is_enabled and hasattr(model, 'enable_memory_bank'):
            try:
                model.enable_memory_bank()
                if self.is_world_process_zero():
                    vprint("  Memory bank functionality enabled for task-level evaluation")
                    vprint("  Memory bank configuration:")
                    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
                        memory_bank_layers = 0
                        for layer_idx, layer in enumerate(model.model.model.layers):
                            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'use_memory_bank'):
                                if layer.self_attn.use_memory_bank:
                                    memory_bank_layers += 1
                                    vprint(f"    - Layer {layer_idx}: Memory bank enabled")
                        vprint(f"    - Total layers with memory bank: {memory_bank_layers}")
            except Exception as e:
                if self.is_world_process_zero():
                    print(f"  Warning: Failed to enable memory bank: {e}")
        
        output_dir = os.path.join(self.args.output_dir, f"{metric_key_prefix}_task_level_results")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving task-level results to: {output_dir}")
            
        dataloader = self.get_eval_dataloader(test_dataset)

        viz_generation_kwargs = {
            "num_beams": 1,
            "max_new_tokens": self.model.image_token_num + 20, 
            "temperature": 0.7, 
            "top_p": 0.9,      
        }
        action_generation_kwargs = self._gen_kwargs if hasattr(self, '_gen_kwargs') else {}


        # --- 2. Main Loop ---
        all_trajectories_log = []
        all_trajectory_distances = []
        all_trajectory_ate = []
        all_trajectory_rpe = []

        world_size, rank = max(1, self.args.world_size), max(0, self.args.process_index)
        for item_idx, inputs in enumerate(dataloader):
            # accelerate shards the (batch-1) eval loader round-robin over processes: global index = local * world + rank
            global_idx = item_idx * world_size + rank if world_size > 1 else item_idx
            if global_idx >= len(test_examples):
                break
            raw_item = test_examples[global_idx]

            # --skip_existing: re-use an episode that an earlier (interrupted) run already completed
            existing = os.path.join(output_dir, os.path.basename(os.path.dirname(raw_item['input_img_paths'][0])), "eval_result.json")
            if getattr(self.args, "skip_existing", False) and os.path.isfile(existing):
                rec = json.load(open(existing))
                result = {"distance_to_goal": rec.get("distance_to_goal"), "ate": rec.get("ate", float('nan')), "rpe_trans": rec.get("rpe_trans", float('nan')),
                          "summary_log": {"trajectory_id": os.path.basename(os.path.dirname(existing)), "steps": rec.get("steps", []), "reused": True}}
            else:
                result = self._process_single_trajectory(
                    raw_item, 
                    output_dir,
                    model,
                    action_generation_kwargs,
                    viz_generation_kwargs,
                    manager
                )
            
            # Aggregate results
            if result.get("distance_to_goal") is not None:
                all_trajectory_distances.append(result["distance_to_goal"])
            all_trajectory_ate.append(result.get("ate", float('nan')))
            all_trajectory_rpe.append(result.get("rpe_trans", float('nan')))
            all_trajectories_log.append(result["summary_log"])

            # Print running averages
            if self.is_world_process_zero() and all_trajectory_distances:
                current_avg_distance = np.mean(all_trajectory_distances)
                current_avg_ate = np.nanmean(all_trajectory_ate)
                current_avg_rpe = np.nanmean(all_trajectory_rpe)
                print(f"--- Running Avg Distance: {current_avg_distance:.3f}, Avg ATE: {current_avg_ate:.3f}, Avg RPE: {current_avg_rpe:.3f} ---")

        # --- 3. Final Logging and Return ---
        log_file_path = os.path.join(output_dir, "predictions_log_by_name.json")
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(all_trajectories_log, f, indent=4, ensure_ascii=False)
        print(f"\nFull trajectory log saved to: {log_file_path}")
        
        # Print final memory bank statistics
        if manager.is_enabled and hasattr(model, 'memory_bank_initialized') and model.memory_bank_initialized:
            if self.is_world_process_zero():
                vprint("\n=== Memory Bank Final Statistics ===")
                total_stored_elements = 0
                if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
                    for layer_idx, layer in enumerate(model.model.model.layers):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'stored_keys'):
                            if layer.self_attn.stored_keys is not None:
                                layer_elements = layer.self_attn.stored_keys.numel() + layer.self_attn.stored_values.numel()
                                total_stored_elements += layer_elements
                                vprint(f"  Layer {layer_idx}: {layer_elements} elements stored")
                vprint(f"  Total memory bank storage: {total_stored_elements} elements")
                vprint(f"  Memory bank was used across {len(all_trajectories_log)} trajectories")
                vprint("=" * 40)

        # gather per-trajectory results from every rank before averaging
        if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, (all_trajectory_distances, all_trajectory_ate, all_trajectory_rpe))
            all_trajectory_distances = [d for g in gathered for d in g[0]]
            all_trajectory_ate = [a for g in gathered for a in g[1]]
            all_trajectory_rpe = [r for g in gathered for r in g[2]]
        metrics = {
            f"{metric_key_prefix}_mean_distance": float(np.nanmean(all_trajectory_distances)) if all_trajectory_distances else float('nan'),
            f"{metric_key_prefix}_mean_ate": float(np.nanmean(all_trajectory_ate)) if all_trajectory_ate else float('nan'),
            f"{metric_key_prefix}_mean_rpe": float(np.nanmean(all_trajectory_rpe)) if all_trajectory_rpe else float('nan'),
            f"{metric_key_prefix}_num_trajectories": len(all_trajectory_ate),
            f"{metric_key_prefix}_samples_processed": len(test_dataset)
        }
        self.log(metrics)
        return metrics

    def _stitch_and_save_comparison_image(self, gt_img: Image.Image, pred_img: Image.Image, save_path: str):
        gt_img = gt_img.resize((256, 256))
        pred_img = pred_img.resize((256, 256))
        
        comparison_img = Image.new('RGB', (256 * 2, 256))
        comparison_img.paste(gt_img, (0, 0))
        comparison_img.paste(pred_img, (256, 0))
        comparison_img.save(save_path)

    # ------------------------------------------------------------------ action-conditioning probe
    PROBE_CONDITIONS = [  # name -> action (dx, dy, dyaw) or "recorded" / "mirrored" (recorded with dyaw negated)
        ("recorded", None), ("mirrored", None), ("zero", (0.0, 0.0, 0.0)), ("forward", (0.15, 0.0, 0.0)),
        ("left", (0.10, 0.0, 0.20)), ("right", (0.10, 0.0, -0.20)),
    ]

    def run_action_probe(self, test_dataset, test_examples, metric_key_prefix="eval"):
        """Single-step imagination under counterfactual actions.

        For each test trajectory, one 'straight' frame (|dyaw| < 0.02) and one 'turn' frame (|dyaw| >= 0.05) are
        taken from the real sequence; from the same (start, goal, current) images the next frame is imagined under the
        recorded action, the mirrored action, a zero action, and fixed forward / left / right actions, `--probe_repeats`
        times each. Frames and a manifest are written for tools/action_probe_analysis.py (LPIPS between conditions vs
        between repeats; optical-flow yaw proxy of each frame against the commanded turn).
        """
        model = self.model; model.eval()
        output_dir = os.path.join(self.args.output_dir, "action_probe_results"); os.makedirs(output_dir, exist_ok=True)
        viz_kwargs = {"num_beams": 1, "max_new_tokens": self.model.image_token_num + 20, "temperature": 0.7, "top_p": 0.9}
        repeats = int(getattr(self.args, "probe_repeats", 2) or 2)
        world_size, rank = max(1, self.args.world_size), max(0, self.args.process_index)
        manifest = []
        for idx in range(rank, len(test_examples), world_size):
            raw_item = test_examples[idx]
            traj_dir = os.path.dirname(raw_item['input_img_paths'][0]); traj = os.path.basename(traj_dir)
            with open(os.path.join(traj_dir, "traj_data.pkl"), "rb") as f:
                pkl = pickle.load(f)
            delta = np.asarray(pkl["delta"]); n = len(delta) + 1
            frames = []
            straight = [k for k in range(1, n - 1) if abs(delta[k][2]) < 0.02 and delta[k][0] > 0.03]
            turn = [k for k in range(1, n - 1) if abs(delta[k][2]) >= 0.05]
            if straight: frames.append(("straight", straight[len(straight) // 2]))
            if turn: frames.append(("turn", turn[0]))
            coords = np.asarray(raw_item['coords']); start_pose = coords[0]
            start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"
            start_img = Image.open(os.path.join(traj_dir, "0.jpg")).convert("RGB").resize((256, 256))
            goal_img = Image.open(os.path.join(traj_dir, f"{n - 1}.jpg")).convert("RGB").resize((256, 256))
            for kind, k in frames:
                cur = self._sanitize_generated_image(Image.open(os.path.join(traj_dir, f"{k}.jpg")).convert("RGB").resize((256, 256)))
                rec = [float(v) for v in delta[k][:3]]
                out_k = os.path.join(output_dir, traj, f"k{k:02d}"); os.makedirs(out_k, exist_ok=True)
                cur.save(os.path.join(out_k, "current.png"))
                entry = {"traj": traj, "k": int(k), "kind": kind, "recorded": rec, "real_next": os.path.join(traj_dir, f"{k + 1}.jpg"),
                         "current": os.path.join(out_k, "current.png"), "generated": {}}
                for name, act in self.PROBE_CONDITIONS:
                    action = rec if name == "recorded" else ([rec[0], rec[1], -rec[2]] if name == "mirrored" else list(act))
                    prompt = build_viz_prompt(decoded_action=action_to_text(action), start_pose_str=start_pose_str, memory_block="")
                    inputs = self._prompt_inputs(prompt, [start_img, goal_img, cur], [])
                    paths = []
                    for r in range(repeats):
                        with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                            out = model.generate(**inputs, **viz_kwargs)
                        sp = os.path.join(out_k, f"{name}_{r}.png")
                        img = self._parse_and_save_image(out[0], model, self.tokenizer, sp)
                        paths.append(sp if img is not None else None)
                    entry["generated"][name] = {"action": action, "paths": paths}
                manifest.append(entry)
                if self.is_world_process_zero():
                    print(f"[probe] {traj} k={k} ({kind}) done")
        with open(os.path.join(output_dir, f"manifest_rank{rank}.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        if world_size > 1:
            torch.distributed.barrier()
        return {f"{metric_key_prefix}_probe_frames": len(manifest)}


    # ------------------------------------------------------------------ shared helpers for the diagnostics
    # The three diagnostics below build prompts WITHOUT the memory block, so they apply to no-memory checkpoints only.
    STOP_DIST_RULE = (0.2, 0.3)   # distance-heading Stop rule of the `stop` / data-efficiency models (Section 3.4)

    def _hf(self):
        return self.model.model if hasattr(self.model, "model") else self.model

    def _check_no_memory(self, task):
        if getattr(self.args, "memory", "none") != "none":
            raise ValueError(f"{task} builds prompts without the memory block; run it on a --memory none checkpoint")

    def _traj_meta(self, raw_item):
        """Trajectory directory, labels, task length, start-pose string, ranges and goal image of a task item.

        Follows the task truncation of --test_max_frames (n and the goal come from the task item, as in the closed loop).
        """
        traj_dir = os.path.dirname(raw_item['input_img_paths'][0])
        with open(os.path.join(traj_dir, "traj_data.pkl"), "rb") as f:
            pkl = pickle.load(f)
        delta_full = np.asarray(pkl["delta"], dtype=np.float64)
        n = int(min(len(raw_item['coords']), len(delta_full) + 1))
        delta = delta_full[:n - 1]
        coords = np.asarray(raw_item['coords']); sp = coords[0]
        start_pose_str = f"Starting Point Coordinate: x={sp[0]:.3f}, y={sp[1]:.3f}, yaw={sp[2]:.3f}\n"
        ranges = DEFAULT_RANGES
        for name, r in DATASET_RANGES.items():
            if f"/{name}/" in raw_item['input_img_paths'][0].replace("\\", "/"):
                ranges = r
        goal = raw_item['input_imgs'][1].convert("RGB").resize((256, 256))
        return traj_dir, pkl, delta, n, start_pose_str, ranges, goal

    @staticmethod
    def _frame(traj_dir, k):
        return Image.open(os.path.join(traj_dir, f"{k}.jpg")).convert("RGB").resize((256, 256))

    def _stop_labels(self, pkl, k, n):
        """Both Stop rules of scripts/navigation.py for frame k of a task of length n: (released, distance-heading)."""
        pos = np.asarray(pkl["position"], dtype=np.float64)[:n, :2]; yaw = np.asarray(pkl["yaw"], dtype=np.float64).ravel()[:n]
        d = float(np.linalg.norm(pos[n - 1] - pos[k])); dyaw = abs((yaw[k] - yaw[n - 1] + np.pi) % (2 * np.pi) - np.pi)
        sd, sy = self.STOP_DIST_RULE
        released = bool(k >= n - 1)
        return released, bool(released or (d < sd and dyaw < sy))

    def _sample_actions(self, images, start_pose_str, ranges, n_samples, greedy=True):
        """Action-head predictions for one observation: [greedy] + n sampled; each 'stop', [dx, dy, dyaw] or None."""
        prompt = build_action_prompt(start_pose_str=start_pose_str, dxy_range=ranges['dxy'], dyaw_range=ranges['dyaw'], memory_block="")
        inputs = self._prompt_inputs(prompt, images, [])
        kw = dict(self._gen_kwargs) if hasattr(self, "_gen_kwargs") else {}
        bin_step = self.action_cfg['bin_step']
        out = []
        runs = ([{**kw, "do_sample": False}] if greedy else []) + [kw] * int(n_samples)
        for g in runs:
            with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                o = self.model.generate(**inputs, **g)
            dec = self.tokenizer.batch_decode(o[0], skip_special_tokens=False)[0].strip()
            txt = re.sub(r"<reserved\d+>|</s>|<pad>", "", dec).strip()
            if txt.lower().startswith("stop"):
                out.append("stop"); continue
            dec = re.sub(r'(<d[^>]+>)+(<d[^>]+>)', r'\2', dec)
            ok = all(re.search(rf"<{c}_(pos|neg)_bin_\d+>", dec) for c in ("dx", "dy", "dyaw"))
            out.append([float(extract_bin_values(dec, c, bin_step)) for c in ("dx", "dy", "dyaw")] if ok else None)
        return out

    def _imagine(self, images, action, start_pose_str, save_path=None):
        """One imagined next frame (sanitised like the closed loop) for an action; None if generation fails."""
        viz_kwargs = {"num_beams": 1, "max_new_tokens": self.model.image_token_num + 20, "temperature": 0.7, "top_p": 0.9}
        prompt = build_viz_prompt(decoded_action=action_to_text([float(v) for v in action]), start_pose_str=start_pose_str, memory_block="")
        inputs = self._prompt_inputs(prompt, images, [])
        with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
            out = self.model.generate(**inputs, **viz_kwargs)
        img = self._parse_and_save_image(out[0], self.model, self.tokenizer, save_path)
        return self._sanitize_generated_image(img) if img is not None else None

    def _encode_ids(self, text, images, drop_bos=False):
        """Token ids with image placeholders replaced by VQ tokens, exactly as scripts/interleaved_tokenized_dataset.py."""
        t = self.tokenizer(text=[text], images=images if images else None, return_tensors="pt", padding=False, truncation=False)
        ids = t['input_ids']
        if images:
            hf = self._hf(); core = hf.model
            vq_dtype = next(core.vqmodel.parameters()).dtype
            pv = t['pixel_values'].to(self.args.device).to(vq_dtype)
            with torch.no_grad():
                img_tokens = core.get_image_tokens(pv).to(torch.int64).cpu().reshape(-1)
            mask = ids == hf.config.image_token_id
            assert int(mask.sum()) == img_tokens.shape[0], (int(mask.sum()), img_tokens.shape)
            ids = ids.clone(); ids[mask] = img_tokens
        if drop_bos:
            ids = ids[:, 1:]
        return ids

    def _target_logprobs(self, src_ids, tgt_ids):
        """Per-token log-probabilities of tgt_ids given src_ids (teacher forcing, logits only at target positions)."""
        ids = torch.cat([src_ids, tgt_ids], 1).to(self.args.device)
        labels = torch.cat([torch.full_like(src_ids, -100), tgt_ids], 1).to(self.args.device)
        shift = labels[:, 1:]; keep = shift != -100
        loss_positions = torch.cat([keep, torch.zeros_like(keep[:, :1])], dim=1)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), loss_positions=loss_positions)
        lp = torch.log_softmax(out.logits.float(), dim=-1)
        return lp.gather(-1, shift[keep][:, None]).squeeze(-1).cpu().numpy()

    # ------------------------------------------------------------------ direction 1: teacher-forced likelihood
    def run_action_likelihood(self, test_dataset, test_examples, metric_key_prefix="eval"):
        """Log-likelihood of the REAL next frame under different action texts (no generation, no optical flow).

        For every frame k of every test trajectory (stride --al_stride), the visualisation prompt is built exactly as
        in training with (start, goal, real frame k) and one of the candidate actions below, and the real frame k+1
        is scored token by token. Candidates: recorded delta[k]; heading-negated; zero; an action drawn from another
        trajectory; the earlier labels delta[k-1] and delta[k-2] and the later delta[k+1] (label-timing test).
        Candidates whose binned action text equals an earlier candidate's are not re-scored: they are stored as
        aliases ("same_text_as") with identical scores, so ties are explicit. The last frame of each task, whose
        target is the goal image already present in the prompt, is flagged "target_is_goal" (a copy control).
        """
        self._check_no_memory("action_likelihood")
        self.model.eval()
        out_dir = os.path.join(self.args.output_dir, "action_likelihood_results"); os.makedirs(out_dir, exist_ok=True)
        world, rank = max(1, self.args.world_size), max(0, self.args.process_index)
        stride = max(1, int(getattr(self.args, "al_stride", 1) or 1))
        pool = []
        for ex in test_examples:
            td = os.path.dirname(ex['input_img_paths'][0])
            with open(os.path.join(td, "traj_data.pkl"), "rb") as f:
                pool += [(os.path.basename(td), [float(v) for v in a[:3]]) for a in np.asarray(pickle.load(f)["delta"])]
        hf = self._hf(); boi, eoi = hf.config.boi_token_id, hf.config.eoi_token_id
        n_img = int(self.model.image_token_num)
        records = []
        for idx in range(rank, len(test_examples), world):
            raw = test_examples[idx]
            traj_dir, pkl, delta, n, sps, ranges, goal = self._traj_meta(raw); traj = os.path.basename(traj_dir)
            start = self._frame(traj_dir, 0)
            others = [a for t, a in pool if t != traj]
            for k in range(0, n - 1, stride):
                rng = random.Random(f"{traj}-{k}")
                tgt = self._encode_ids("<image>", [self._frame(traj_dir, k + 1)], drop_bos=True)
                t0 = tgt[0].tolist()
                assert t0[0] == boi and t0[n_img + 1] == eoi, (t0[:2], t0[n_img:n_img + 3])
                rec = [float(v) for v in delta[k][:3]]
                cands = {"recorded": rec, "negated": [rec[0], rec[1], -rec[2]], "zero": [0.0, 0.0, 0.0]}
                if others:
                    cands["other_traj"] = rng.choice(others)
                if k >= 1: cands["prev1"] = [float(v) for v in delta[k - 1][:3]]
                if k >= 2: cands["prev2"] = [float(v) for v in delta[k - 2][:3]]
                if k + 1 < len(delta): cands["next1"] = [float(v) for v in delta[k + 1][:3]]
                cur = self._frame(traj_dir, k)
                scores, seen = {}, {}
                for name, a in cands.items():
                    txt = action_to_text(a)
                    if txt in seen:
                        scores[name] = {**scores[seen[txt]], "action": a, "same_text_as": seen[txt]}
                        continue
                    seen[txt] = name
                    prompt = build_viz_prompt(decoded_action=txt, start_pose_str=sps, memory_block="")
                    src = self._encode_ids(prompt, [start, goal, cur])
                    lp = self._target_logprobs(src, tgt)
                    scores[name] = {"action": a, "action_text": txt, "nll_img": float(-lp[1:n_img + 1].sum()),
                                    "nll_all": float(-lp.sum()), "n_tgt": int(len(lp))}
                records.append({"traj": traj, "k": int(k), "n": int(n), "dyaw_k": rec[2],
                                "dyaw_km1": float(delta[k - 1][2]) if k >= 1 else None,
                                "target_is_goal": bool(k + 1 == n - 1), "scores": scores})
            if self.is_world_process_zero():
                print(f"[likelihood] {traj} done ({len(records)} frames on rank 0)", flush=True)
        with open(os.path.join(out_dir, f"likelihood_rank{rank}.json"), "w") as f:
            json.dump(records, f)
        if world > 1:
            torch.distributed.barrier()
        return {f"{metric_key_prefix}_likelihood_frames": len(records)}

    # ------------------------------------------------------------------ direction 2: controlled intervention
    def run_action_intervention(self, test_dataset, test_examples, metric_key_prefix="eval"):
        """How much do imagined observations change the action head's decision, and does it grow with depth?

        Along each test trajectory the imagination is driven by the RECORDED actions (so the reference stays valid).
        At every step s (including the goal frame s = n-1, which is Stop-labelled under both rules) the action head
        is queried, without executing its output, on three observations of the same moment: (a) the real frame s,
        (b) the real frame s after a VQ round trip, (c) the frame imagined after s recorded-action steps. Each query
        returns a greedy action and --ai_samples sampled actions. --ai_max_steps caps the steps (0 = whole task).
        """
        self._check_no_memory("action_intervention")
        self.model.eval()
        out_dir = os.path.join(self.args.output_dir, "action_intervention_results"); os.makedirs(out_dir, exist_ok=True)
        world, rank = max(1, self.args.world_size), max(0, self.args.process_index)
        n_samp = int(getattr(self.args, "ai_samples", 5) or 5)
        cap = int(getattr(self.args, "ai_max_steps", 0) or 0)
        records = []
        for idx in range(rank, len(test_examples), world):
            raw = test_examples[idx]
            traj_dir, pkl, delta, n, sps, ranges, goal = self._traj_meta(raw); traj = os.path.basename(traj_dir)
            start = self._frame(traj_dir, 0)
            gen = self._sanitize_generated_image(start)
            n_q = n if cap <= 0 else min(n, cap)
            steps = []
            for s in range(n_q):
                real = self._frame(traj_dir, s)
                vq = self._sanitize_generated_image(real)
                released, dist = self._stop_labels(pkl, s, n)
                row = {"s": int(s), "recorded": [float(v) for v in delta[s][:3]] if s < n - 1 else None,
                       "stop_label_released": released, "stop_label_dist": dist, "gen_available": gen is not None}
                for name, obs in (("real", real), ("vq", vq), ("gen", gen)):
                    row[name] = self._sample_actions([start, goal, obs], sps, ranges, n_samp, greedy=True) if obs is not None else None
                steps.append(row)
                if gen is not None and s + 1 < n_q:
                    gen = self._imagine([start, goal, gen], delta[s][:3], sps, os.path.join(out_dir, traj, f"gen_{s + 1}.png"))
            records.append({"traj": traj, "n": int(n), "steps": steps})
            if self.is_world_process_zero():
                print(f"[intervention] {traj} done", flush=True)
        with open(os.path.join(out_dir, f"intervention_rank{rank}.json"), "w") as f:
            json.dump(records, f)
        if world > 1:
            torch.distributed.barrier()
        return {f"{metric_key_prefix}_intervention_trajs": len(records)}

    # ------------------------------------------------------------------ direction 3: variance decomposition
    def run_variance_decomp(self, test_dataset, test_examples, metric_key_prefix="eval"):
        """Split the variability of the next decision into an image part and an action-sampling part.

        At --vd_states frames per test trajectory, the next frame is imagined --vd_frames times with the recorded
        action from the VQ round trip of the real frame (the input the closed loop imagines from). On each imagined
        frame, on the real next frame and on its VQ round trip, the action head is sampled --vd_actions times.
        """
        self._check_no_memory("variance_decomp")
        self.model.eval()
        out_dir = os.path.join(self.args.output_dir, "variance_decomp_results"); os.makedirs(out_dir, exist_ok=True)
        world, rank = max(1, self.args.world_size), max(0, self.args.process_index)
        n_states = int(getattr(self.args, "vd_states", 5) or 5)
        n_frames = int(getattr(self.args, "vd_frames", 4) or 4)
        n_act = int(getattr(self.args, "vd_actions", 5) or 5)
        records = []
        for idx in range(rank, len(test_examples), world):
            raw = test_examples[idx]
            traj_dir, pkl, delta, n, sps, ranges, goal = self._traj_meta(raw); traj = os.path.basename(traj_dir)
            if n < 3:
                continue
            start = self._frame(traj_dir, 0)
            ks = sorted(set(int(round(x)) for x in np.linspace(1, n - 2, min(n_states, n - 2))))
            for k in ks:
                cur = self._sanitize_generated_image(self._frame(traj_dir, k))
                frames = []
                for j in range(n_frames):
                    img = self._imagine([start, goal, cur], delta[k][:3], sps, os.path.join(out_dir, traj, f"k{k:02d}_img{j}.png"))
                    frames.append(self._sample_actions([start, goal, img], sps, ranges, n_act, greedy=False) if img is not None else None)
                real_next = self._frame(traj_dir, k + 1)
                ref = self._sample_actions([start, goal, real_next], sps, ranges, n_act, greedy=False)
                ref_vq = self._sample_actions([start, goal, self._sanitize_generated_image(real_next)], sps, ranges, n_act, greedy=False)
                released, dist = self._stop_labels(pkl, k + 1, n)
                records.append({"traj": traj, "k": int(k), "n": int(n), "recorded_k": [float(v) for v in delta[k][:3]],
                                "recorded_next": [float(v) for v in delta[k + 1][:3]] if k + 1 < len(delta) else None,
                                "stop_next_released": released, "stop_next_dist": dist,
                                "imagined": frames, "real_next": ref, "real_next_vq": ref_vq})
            if self.is_world_process_zero():
                print(f"[variance] {traj} done", flush=True)
        with open(os.path.join(out_dir, f"variance_rank{rank}.json"), "w") as f:
            json.dump(records, f)
        if world > 1:
            torch.distributed.barrier()
        return {f"{metric_key_prefix}_variance_states": len(records)}

    def _process_gt_action_rollout_trajectory(self, raw_item: Dict, output_dir: str, model: Any, viz_gen_kwargs: Dict, image_metrics_calculator) -> Dict[str, Any]:
        start_path = raw_item['input_img_paths'][0]
        traj_dir = os.path.dirname(start_path)
        traj_dirname = os.path.basename(traj_dir)
        traj_output_dir = os.path.join(output_dir, traj_dirname)
        os.makedirs(traj_output_dir, exist_ok=True)

        if self.is_world_process_zero():
            vprint(f"--- Processing Trajectory {traj_dirname} ---")

        # --- 1. Load all required data from files ---
        try:
            # Load GT actions from pickle file
            with open(os.path.join(traj_dir, 'traj_data.pkl'), 'rb') as f:
                gt_actions = pickle.load(f)['delta']
            if hasattr(gt_actions, 'tolist'):
                gt_actions = gt_actions.tolist()

            # Reconstruct GT image paths
            img_files = sorted(
                [f for f in os.listdir(traj_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                key=lambda f: int(os.path.splitext(f)[0])
            )
            gt_img_paths = [os.path.join(traj_dir, f) for f in img_files]
        except Exception as e:
            logger.error(f"  Failed to load data for {traj_dirname}. Error: {e}")
            return {} # Return empty dict on failure

        # --- 2. Initialize rollout state ---
        start_img = self._sanitize_generated_image(raw_item['input_imgs'][0].convert("RGB").resize((256, 256)))
        goal_img = self._sanitize_generated_image(raw_item['input_imgs'][1].convert("RGB").resize((256, 256)))
        start_pose_str = f"Start Pose: x={raw_item['coords'][0][0]:.3f}, y={raw_item['coords'][0][1]:.3f}, yaw={raw_item['coords'][0][2]:.3f}\n"
        
        current_observation = start_img
        num_steps = 5
        pose_mem = self._pose_memory(model)
        coords = [list(c) for c in raw_item['coords']]
        if pose_mem is not None:
            pose_mem.reset(); pose_mem.add(coords[0], current_observation)

        # --- 3. The 5-Step Rollout Loop ---
        for step in range(num_steps):
            if not (step < len(gt_actions) and (step + 1) < len(gt_img_paths)):
                logger.warning(f"  Trajectory {traj_dirname} too short. Stopping.")
                return {} # Skip this trajectory

            gt_action_vec = gt_actions[step]
            action_text = action_to_text(gt_action_vec)
            
            mem_text, mem_tokens = pose_mem.query(coords[step]) if pose_mem is not None else ("", [])
            viz_prompt = build_viz_prompt(action_text, start_pose_str, memory_block=mem_text)
            viz_inputs = self._prompt_inputs(viz_prompt, [start_img, goal_img, current_observation], mem_tokens)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                viz_outputs = model.generate(**viz_inputs, **viz_gen_kwargs)

            save_path = os.path.join(traj_output_dir, f"step_{step+1}_observation.png")
            new_obs_img = self._parse_and_save_image(viz_outputs[0], self.model, self.tokenizer, save_path)

            if not new_obs_img:
                logger.warning(f"  Visualization failed at step {step + 1}. Stopping.")
                return {} # Skip this trajectory
            
            # Stitch GT and Pred images and save
            gt_current_img = Image.open(gt_img_paths[step + 1])
            self._stitch_and_save_comparison_image(gt_current_img, new_obs_img, save_path)

            current_observation = self._sanitize_generated_image(new_obs_img)
            if pose_mem is not None and step + 1 < len(coords):
                pose_mem.add(coords[step + 1], current_observation)

        # --- 4. Final Evaluation and Saving ---
        final_predicted_img = current_observation
        gt_final_img = Image.open(gt_img_paths[num_steps]).convert("RGB")
        
        final_viz_metrics = image_metrics_calculator.calculate(final_predicted_img, gt_final_img)
        
        eval_file_path = os.path.join(traj_output_dir, "eval_result.json")
        with open(eval_file_path, "w") as f_eval:
            json.dump(final_viz_metrics, f_eval, indent=4)
        
        # Return only the metrics for aggregation
        return final_viz_metrics


    def run_gt_action_rollout(self, test_dataset: Dataset, test_examples: Dataset, metric_key_prefix: str = "eval") -> Dict[str, float]:
        logger.info("\n***** Running GT Action Viz Rollout *****")

        # --- 1. Setup ---
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        output_dir = os.path.join(self.args.output_dir, f"{metric_key_prefix}_gt_action_rollout_results")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving task-level results to: {output_dir}")
            
        dataloader = self.get_eval_dataloader(test_dataset)

        viz_generation_kwargs = {
            "num_beams": 1,
            "max_new_tokens": self.model.image_token_num + 20,
            "temperature": 0.7, 
            "top_p": 0.9,
        }
        action_generation_kwargs = self._gen_kwargs if hasattr(self, '_gen_kwargs') else {}
        
        image_metrics_calculator = ImageMetricsCalculator(device=self.args.device)

        # --- 2. Main Loop ---
        all_viz_metrics = defaultdict(list)

        world_size, rank = max(1, self.args.world_size), max(0, self.args.process_index)
        for item_idx, inputs in enumerate(dataloader):
            global_idx = item_idx * world_size + rank if world_size > 1 else item_idx
            if global_idx >= len(test_examples):
                break
            raw_item = test_examples[global_idx]
            
            metrics = self._process_gt_action_rollout_trajectory(
                raw_item, 
                output_dir,
                model,
                viz_generation_kwargs,
                image_metrics_calculator
            )
            
            if metrics:
                for key, value in metrics.items():
                    all_viz_metrics[key].append(value)
        
        # --- 3. Final Logging and Return ---
        if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, dict(all_viz_metrics))
            merged = defaultdict(list)
            for g in gathered:
                for k, v in g.items():
                    merged[k].extend(v)
            all_viz_metrics = merged
        final_summary = {}
        print("\n--- Final Aggregated Metrics ---")
        for key, values in all_viz_metrics.items():
            if values:
                mean_value = np.mean(values)
                final_summary[f"{metric_key_prefix}_{key}_mean"] = mean_value
                print(f"  Average {key}: {mean_value:.4f}")

        summary_path = os.path.join(output_dir, "final_summary.json")
        with open(summary_path, "w") as f:
            json.dump(final_summary, f, indent=4)
        print(f"Final summary saved to {summary_path}")

        self.log(final_summary)
        return final_summary
