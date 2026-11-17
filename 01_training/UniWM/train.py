import os
import torch
import logging
import argparse
import yaml
import copy
import math
import configparser
import transformers
import torch.distributed as dist

from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, StopStringCriteria, set_seed, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import get_last_checkpoint
from transformers.generation import StoppingCriteriaList

from scripts.run_config import create_run_name
from scripts.training_arguments import WrappedSeq2SeqTrainingArguments
from scripts.load_data import load_data, tokenize_dataset
from scripts.load_model import load_model
from scripts.evaluator import VisualizationEvaluator
from scripts.action_utils import generate_bin_tokens

from transformers.utils import logging as hf_logging 

hf_logging.set_verbosity_error()  

logger = logging.getLogger(__name__)

WANDB_API_KEY = "<YOUR_WANDB_KEY_API>"
WANDB_ENTITY = "<YOUR_WANDB_ENTITY>"
PROJECT_NAME = "<YOUR_PROJECT_NAME>"

import os
import re

def find_latest_valid_checkpoint(ckpt_dir):
    """Find the latest valid checkpoint (contains trainer_state.json) in the directory."""
    if not os.path.exists(ckpt_dir):
        return None

    checkpoint_dirs = [
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d)) and re.match(r"checkpoint-\d+", d)
    ]

    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]), reverse=True)

    for ckpt in checkpoint_dirs:
        trainer_state_path = os.path.join(ckpt_dir, ckpt, "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            print(f"[INFO] Found valid checkpoint: {ckpt}")
            return os.path.join(ckpt_dir, ckpt)

    print("[INFO] No valid checkpoint found.")
    return None

def init(args):
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    # Read in the training arguments
    setting_type = "interleaved"
    with open(os.path.join(args.cfg_path, setting_type + '.yaml')) as f:
        file = f.read()
        training_cfg = yaml.safe_load(file)

    if args.train_bz:
        training_cfg['hyper']['train_batch_size'] = args.train_bz
    if args.val_bz:
        training_cfg['hyper']['val_batch_size'] = args.val_bz
    if args.grad_acc:
        training_cfg['hyper']['grad_accumulation'] = args.grad_acc

    sup_hyper = training_cfg["hyper"]

    # Construct the run_name of the task
    args.run_name = create_run_name(args, training_cfg)

    args.run_name = args.note + args.run_name

    training_args = WrappedSeq2SeqTrainingArguments(
        output_dir=os.path.join(args.output, args.run_name),
        remove_unused_columns=False,
        evaluation_strategy=training_cfg['eval']['eval_strategy'],
        eval_steps=training_cfg['eval']['eval_steps'] if training_cfg['eval']['eval_strategy'] == "steps" else None,
        save_strategy=training_cfg['save']['save_strategy'],
        save_steps=args.save_steps or (training_cfg['save']['save_steps'] if training_cfg['save']['save_strategy'] == "steps" else 2000),
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        # note: for supervised tuning
        #############################
        learning_rate=args.lr or (sup_hyper['lr'] if sup_hyper else 0),
        per_device_train_batch_size=sup_hyper['train_batch_size'] if sup_hyper else 0,
        gradient_accumulation_steps=sup_hyper['grad_accumulation'] if sup_hyper else 0,
        per_device_eval_batch_size=sup_hyper['val_batch_size'] if sup_hyper else training_cfg['hyper']['val_batch_size'],
        num_train_epochs=args.epochs or (sup_hyper['epochs'] if sup_hyper else 0),
        #############################
        # warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        disable_tqdm=False,
        push_to_hub=False,
        # customize
        predict_with_generate=training_cfg['model']['predict_with_generate'],
        generation_max_new_tokens=training_cfg['model']['generation_max_new_tokens'],
        generation_num_beams=training_cfg['model']['generation_num_beams'],
        use_memory_bank_inference=args.use_memory_bank_inference,
        eval_max_steps=args.eval_max_steps,
        action_samples=args.action_samples,
        gate_std=args.gate_std,
        gate_scale=args.gate_scale,
        mem_text=args.mem_text,
        skip_existing=args.skip_existing,
        action_greedy=args.action_greedy,
        mem_ablate_image=args.mem_ablate_image,
        frame_samples=args.frame_samples,
        probe_repeats=args.probe_repeats,
        al_stride=args.al_stride,
        ai_samples=args.ai_samples,
        ai_max_steps=args.ai_max_steps,
        vd_states=args.vd_states,
        vd_frames=args.vd_frames,
        vd_actions=args.vd_actions,
        memory=args.memory,
        mem_k=args.mem_k,
        mem_res=args.mem_res,
        # precision / memory (V100: fp16 AMP + sdpa; A100: bf16)
        fp16=(args.precision == 'fp16'),
        bf16=(args.precision == 'bf16'),
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        ddp_timeout=args.ddp_timeout,
        max_steps=args.max_steps,
        dataloader_num_workers=0,   # VQ tokenization runs on the GPU inside __getitem__
        warmup_ratio=args.warmup_ratio,
        save_only_model=args.save_only_model,
    )

    # Initialize the wandb logger if specified
    try:
        rank = dist.get_rank()
    except:
        rank = 0
    args.local_rank = rank

    if args.report_to == "wandb" and rank == 0:
        import wandb
        init_args = {}

        # note: my new wandb api key
        wandb.login(key=WANDB_API_KEY)

        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        
        if args.local_rank == 0 or args.local_rank is None:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", PROJECT_NAME),
                name=args.run_name,
                entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
                **init_args,
            )
            wandb.config.update(training_args, allow_val_change=True)
    else:
        training_args.report_to = []

    # if os.path.exists(training_args.output_dir):
    #     args.model_ckpt = training_args.output_dir

    # # Detect the checkpoint
    # if args.model_ckpt is not None:
    #     training_args.load_weights_from = get_last_checkpoint(args.model_ckpt)
    # else:
    #     training_args.load_weights_from = None

    # Detect the checkpoint

    # Detect the checkpoint
    if args.model_ckpt is not None:
        trainer_state_file = os.path.join(args.model_ckpt, "trainer_state.json")
        if os.path.isfile(trainer_state_file):
            training_args.load_weights_from = args.model_ckpt
        else:
            latest_ckpt = find_latest_valid_checkpoint(args.model_ckpt)
            if latest_ckpt:
                print(f"[INFO] Found latest valid checkpoint: {latest_ckpt}")
                training_args.load_weights_from = latest_ckpt
            else:
                print(f"[INFO] No valid checkpoint found in directory: {args.model_ckpt}")
                training_args.load_weights_from = None
    else:
        training_args.load_weights_from = None


    return training_cfg, training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="anole")
    parser.add_argument("--data", type=str, nargs="+")
    parser.add_argument("--data_dir", type=str, default="data_samples")
    parser.add_argument("--decoder_type", type=str, default='anole')
    parser.add_argument('--note', type=str, default="debug")
    parser.add_argument('--image_seq_length', type=int, default=784)
    parser.add_argument('--no_perceptual_loss', action="store_true")

    # model argument
    parser.add_argument('--model_ckpt', type=str, default=None, help='path of the checkpoint')
    parser.add_argument('--load_last_checkpoint', action='store_true')

    # training arguments
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_single_step_eval', action='store_true')
    parser.add_argument('--do_task_level_eval', action='store_true')
    parser.add_argument('--do_action_probe', action='store_true', help='single-step imagination under counterfactual actions (see run_action_probe)')
    parser.add_argument('--do_action_likelihood', action='store_true', help='teacher-forced log-likelihood of the real next frame under candidate actions')
    parser.add_argument('--do_action_intervention', action='store_true', help='action head on real / VQ / imagined observations along recorded-action rollouts')
    parser.add_argument('--do_variance_decomp', action='store_true', help='image-sampling vs action-sampling variability of the next decision')
    parser.add_argument('--al_stride', type=int, default=1, help='likelihood: score every n-th frame')
    parser.add_argument('--ai_samples', type=int, default=5, help='intervention: sampled actions per query (plus one greedy)')
    parser.add_argument('--ai_max_steps', type=int, default=0, help='intervention: cap on steps per task (0 = the whole task)')
    parser.add_argument('--vd_states', type=int, default=5)
    parser.add_argument('--vd_frames', type=int, default=4)
    parser.add_argument('--vd_actions', type=int, default=5)
    parser.add_argument('--probe_repeats', type=int, default=2)
    parser.add_argument('--do_rollout_eval', action='store_true')
    parser.add_argument('--cfg_path', type=str, default='cfg')
    parser.add_argument('--patience', type=int, default=5)

    # input format argument
    parser.add_argument('--input_format', type=str, default="anole")

    # output configuration
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--report_to', type=str, default="wandb")
    parser.add_argument('--cache_dir', type=str, default=None)

    # randomness
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int)

    # debug
    parser.add_argument('--toy', action='store_true')

    # shortcut customization
    parser.add_argument('--train_bz', type=int, default=None)
    parser.add_argument('--val_bz', type=int, default=None)
    parser.add_argument('--grad_acc', type=int, default=None)
    
    # precision / attention / LoRA / schedule overrides
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                        help='fp16 for V100 (no bf16 tensor cores), bf16 for A100/H100')
    parser.add_argument('--bfloat16', action='store_true', help='(legacy) same as --precision bf16')
    parser.add_argument('--attn_impl', type=str, default='sdpa', choices=['sdpa', 'eager', 'flash_attention_2'])
    parser.add_argument('--base_model', type=str, default='leloy/Anole-7b-v0.1-hf')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_targets', type=str, default='q_proj,k_proj,v_proj,o_proj',
                        help='comma-separated; e.g. q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    parser.add_argument('--train_action_embeddings', action='store_true', help='full mode: also train the INPUT-embedding rows of the action-bin tokens (frozen at random init in the released recipe)')
    parser.add_argument('--lm_head_mode', type=str, default='new_rows', choices=['new_rows', 'full'],
                        help='new_rows: train only the embed/lm_head rows of the added action tokens (default); full: whole lm_head via modules_to_save')
    parser.add_argument('--no_grad_ckpt', action='store_true')
    parser.add_argument('--no_lora_fp32', action='store_true', help='compute the LoRA branch in fp16 (default: fp32, avoids fp16 gradient overflow)')
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--nan_guard_min_scale', type=float, default=4.0, help='stop training if the fp16 loss scale falls below this')
    parser.add_argument('--grad_downscale_layer', type=int, default=4, help='backward gradient into layers 0..N is divided by --grad_downscale (fp16 only)')
    parser.add_argument('--grad_downscale', type=float, default=64.0, help='1 disables')
    parser.add_argument('--fp32_mlp_layers', type=str, default='auto', help="decoder layers whose MLP runs in fp32 ('auto' = 31 for fp16, none otherwise)")
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--save_only_model', action='store_true', help='checkpoints without optimizer/scheduler state (3.4 GB instead of 5.4 GB; cannot resume from them)')
    parser.add_argument('--epochs', type=float, default=None, help='override cfg epochs')
    parser.add_argument('--lr', type=float, default=None, help='override cfg lr')
    parser.add_argument('--max_steps', type=int, default=-1, help='>0 overrides epochs (step budget)')
    parser.add_argument('--split_ratios', type=str, default='0.9,0.05,0.05', help='train,dev,test fractions per dataset (by trajectory)')
    parser.add_argument('--split_by', type=str, default='traj', choices=['traj', 'segment'], help="split unit: 'segment' keeps all chunks of one recording in one split (no train/test leakage)")
    parser.add_argument('--max_train_trajs', type=int, default=0, help='>0: keep only the first N training trajectories of each dataset')
    parser.add_argument('--eval_max_steps', type=int, default=10, help='max closed-loop steps per trajectory in task-level eval (paper: 10)')
    parser.add_argument('--action_samples', type=int, default=1, help='closed loop: sample N candidate actions per step, execute the component-wise median, Stop on majority (1 = single sample)')
    parser.add_argument('--gate_std', type=float, default=0.0, help='closed loop with --action_samples > 1: when the largest per-component std of the candidates exceeds this, scale the executed action by --gate_scale (0 = off)')
    parser.add_argument('--gate_scale', type=float, default=0.5, help='step scale applied when the disagreement gate fires')
    parser.add_argument('--mem_text', type=str, default='pose', choices=['pose', 'none'], help='memory block text: pose = relative pose of each frame (default); none = frames only (control)')
    parser.add_argument('--closed_loop_extra', type=str, default='none', choices=['none', 'dev'], help='dev: also run the closed-loop evaluation on the dev trajectories (about 50 episodes)')
    parser.add_argument('--skip_existing', action='store_true', help='closed loop: re-use episodes whose eval_result.json already exists (resume an interrupted evaluation)')
    parser.add_argument('--action_greedy', action='store_true', help='closed loop: greedy action decoding (deterministic given the imagined frames)')
    parser.add_argument('--mem_ablate_image', type=str, default='', help='closed loop: replace every memory frame by this image file (inference-time ablation of the memory frames)')
    parser.add_argument('--frame_samples', type=int, default=1, help='closed loop: sample N imagined frames per step and execute on the LPIPS medoid; logs the frame disagreement (1 = single sample)')
    parser.add_argument('--test_max_frames', type=int, default=0, help='>0: truncate each closed-loop task to its first N frames (goal = frame N-1); 15 reproduces the length of the released 1X trajectories')
    parser.add_argument('--ddp_timeout', type=int, default=21600, help='process-group timeout in seconds; the closed-loop ranks finish at very different times, so the default 1800 s can kill a long evaluation at the final gather')
    parser.add_argument('--stop_dist', type=float, default=0.0, help="label the action as 'Stop' within this distance of the goal (0 = last frame only)")
    parser.add_argument('--stop_yaw', type=float, default=0.3, help='heading tolerance (rad) for the Stop label')
    parser.add_argument('--memory', type=str, default='none', choices=['none', 'recent', 'pose'], help='pose-indexed memory frames in the prompts (I3)')
    parser.add_argument('--mem_k', type=int, default=2, help='number of memory frames')
    parser.add_argument('--mem_res', type=int, default=160, help='memory frame resolution (160 -> 100 tokens, 224 -> 196 tokens)')
    parser.add_argument('--max_source_length', type=int, default=0, help='0 = 3050 without memory, 3050 + mem_k*(tokens+2) + 90 with memory')
    parser.add_argument('--max_target_length', type=int, default=0, help='0 = 850 without memory, 800 with memory')
    parser.add_argument('--init_adapter', type=str, default=None, help='initialise LoRA + lm_head from this trained adapter dir and train a fresh schedule (fine-tuning)')
    parser.add_argument('--eval_samples', type=int, default=0, help='>0: cap the eval split to N samples (single-step eval; 0 = all)')
    parser.add_argument('--test_trajs', type=int, default=0, help='>0: cap the test split to the first N trajectories (closed-loop / rollout eval; 0 = all)')

    parser.add_argument('--use_memory_bank_inference', action='store_true')
    args = parser.parse_args()
    if args.bfloat16:
        args.precision = 'bf16'
    args.lora_fp32 = not args.no_lora_fp32

    if args.model in ['anole']:
        args.decoder_type = args.model
        assert args.input_format == "anole"

    if args.decoder_type in ['anole']:
        args.note = args.note + f"image_seq_len-{str(args.image_seq_length)}-"

    training_cfg, training_args = init(args)

    print(f'Preparing the {args.data} dataset... ')
    data = load_data(dataset=args.data, data_dir=args.data_dir,
                     split_ratios=tuple(float(x) for x in args.split_ratios.split(',')), max_train_trajs=args.max_train_trajs, stop_dist=args.stop_dist, stop_yaw=args.stop_yaw,
                     memory=args.memory, mem_k=args.mem_k, split_by=args.split_by, mem_text=args.mem_text, closed_loop_extra=args.closed_loop_extra, test_max_frames=args.test_max_frames)

    if len(data) == 2:
        train_split, eval_split, test_split = data['train'], None, data['test']
    else:
        try:
            train_split, eval_split, test_split = data['train'], data['dev'], data['test']
        except:
            train_split, eval_split, test_split = data['train'], data['validation'], data['test']

    if args.toy:
        print('Only using toy examples for debugging...')
        train_split = train_split.select(list(range(min(100, len(train_split)))))
        if eval_split:
            eval_split = eval_split.select(list(range(min(10, len(eval_split)))))
        test_split = test_split.select(list(range(min(10, len(test_split)))))

    model_processor = load_model(args, training_cfg)
    model, processor = model_processor['model'], model_processor["processor"]

    # def generate_bin_tokens(prefix, vmin, vmax, step):
    #     nbins = int((vmax - vmin) / step) + 1
    #     return [f"<{prefix}_bin_{i:02d}>" for i in range(nbins)]
    
    bin_tokens = []
    if args.model_ckpt is None:
        # --- Read action token settings from the YAML config ---
        print("Reading action token generation settings from config...")
        action_cfg = training_cfg['action_token_generation']
        MIN_DXY = action_cfg['min_dxy']
        MAX_DXY = action_cfg['max_dxy']
        MIN_DYAW = action_cfg['min_dyaw']
        MAX_DYAW = action_cfg['max_dyaw']
        BIN_STEP = action_cfg['bin_step']

        print(f"DX/DY Range for Vocabulary: [{MIN_DXY}, {MAX_DXY}]")
        print(f"DYAW Range for Vocabulary: [{MIN_DYAW}, {MAX_DYAW}]")
        
        # --- Generate vocabulary of action tokens ---
        print("Generating vocabulary of action tokens...")
        
        bin_tokens += generate_bin_tokens("dx", MIN_DXY, MAX_DXY, BIN_STEP)
        bin_tokens += generate_bin_tokens("dy", MIN_DXY, MAX_DXY, BIN_STEP)
        bin_tokens += generate_bin_tokens("dyaw", MIN_DYAW, MAX_DYAW, BIN_STEP)
        
        print(f"Generated a total of {len(bin_tokens)} action tokens.")


    existing_vocab = set(processor.tokenizer.get_vocab().keys())
    new_tokens = [t for t in bin_tokens if t not in existing_vocab]

    # new_tokens.append("<IMS>")
    # new_tokens.append("<IME>")

    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens, special_tokens=True)

        # Resize embed_tokens / lm_head for the added action tokens. In "full" mode lm_head is a PEFT
        # ModulesToSaveWrapper and must be unwrapped, resized and re-wrapped; otherwise resize directly.
        lm_head = model.model.lm_head
        if hasattr(lm_head, 'base_layer') or 'ModulesToSaveWrapper' in type(lm_head).__name__:
            from peft.utils.other import ModulesToSaveWrapper
            model.model.lm_head = lm_head.base_layer if hasattr(lm_head, 'base_layer') else lm_head.original_module
            model.model.resize_token_embeddings(len(processor.tokenizer))
            model.model.lm_head = ModulesToSaveWrapper(model.model.lm_head, "default")
        else:
            model.model.resize_token_embeddings(len(processor.tokenizer))
        print(f"[train] resized embeddings to {model.get_input_embeddings().weight.shape[0]} rows")

    extra_tokens_enabled = args.lm_head_mode == 'new_rows' and len(new_tokens) > 0
    if extra_tokens_enabled:
        # train only the new-token rows of embed_tokens / lm_head (fp32), base rows stay frozen fp16
        from uniwm import extra_tokens
        extra_tokens.install(model, n_new=len(new_tokens))
        # PEFT would otherwise dump the (frozen, resized) embed_tokens/lm_head into the adapter (~1.1 GB per save)
        _orig_save = model.save_pretrained
        model.save_pretrained = lambda *a, **k: _orig_save(*a, **{**k, 'save_embedding_layers': False})
    elif args.lm_head_mode == 'full':
        # Always store the (resized, untrained) embed_tokens rows with the adapter: PEFT's 'auto' rule needs a hub
        # lookup of the base config, which fails on machines without huggingface.co access, and eval would then
        # re-initialise the action-token embeddings randomly.
        _orig_save = model.save_pretrained
        model.save_pretrained = lambda *a, **k: _orig_save(*a, **{**k, 'save_embedding_layers': True})
    if args.train_action_embeddings:
        if args.lm_head_mode != 'full' or extra_tokens_enabled:
            raise SystemExit('--train_action_embeddings requires --lm_head_mode full')
        from uniwm import action_embed
        _n_old, _n_new = action_embed.action_rows(processor.tokenizer, model)
        _emb = action_embed.install(model, _n_new)
        print(f"[train] action-token input embeddings trainable: rows {_n_old}..{_n_old + _n_new - 1} ({_n_new} x {model.get_input_embeddings().weight.shape[1]}), "
              f"initial md5 {action_embed._digest(_emb.new_weight)}, mean row norm {_emb.new_weight.detach().norm(dim=1).mean().item():.3f}")
    # resize_token_embeddings above re-created lm_head/embed_tokens in the base dtype and re-wrapped lm_head:
    # every trainable parameter must be fp32 for fp16 AMP (GradScaler), so cast again here.
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    _n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _n_all = sum(p.numel() for p in model.parameters())
    print(f"[train] new action tokens: {len(new_tokens)} | trainable params: {_n_train / 1e6:.1f}M / {_n_all / 1e6:.0f}M | precision={args.precision} attn={args.attn_impl}")
    eval_data_num = (len(eval_split) // (training_args.per_device_eval_batch_size * torch.cuda.device_count())) * (training_args.per_device_eval_batch_size * torch.cuda.device_count())
    eval_split = eval_split.select(list(range(eval_data_num)))
    # eval_split = eval_split.filter(lambda ex: len(ex['label_imgs']) > 0 and ex['train_task'] == 'single_step_visualization')
    # eval_split = eval_split.filter(lambda ex: len(ex['label_imgs']) > 0)
    eval_split = eval_split.filter(
    lambda ex: ex['train_task'] == 'action_reasoning' or len(ex['label_imgs']) > 0
    )
    if args.eval_samples and len(eval_split) > args.eval_samples:
        # keep both tasks balanced: the split is already shuffled, so an interleaved take is representative
        n = args.eval_samples // 2
        viz = [i for i, t in enumerate(eval_split['train_task']) if t == 'single_step_visualization'][:n]
        act = [i for i, t in enumerate(eval_split['train_task']) if t == 'action_reasoning'][:args.eval_samples - n]
        eval_split = eval_split.select(sorted(viz + act))
        print(f"[eval] capped eval split to {len(eval_split)} samples ({len(viz)} visualization, {len(act)} action)")
    if args.test_trajs and len(test_split) > args.test_trajs:
        test_split = test_split.select(list(range(args.test_trajs)))   # quick checks of the closed loop
    test_data_num = (len(test_split) // (training_args.per_device_eval_batch_size * torch.cuda.device_count())) * (training_args.per_device_eval_batch_size * torch.cuda.device_count())
    test_split = test_split.select(list(range(test_data_num)))

    print(f"Eval Num: {eval_data_num}")

    tokenized_data, max_source_length, max_target_length = tokenize_dataset(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        model=model,
        processor=processor,
        input_format=args.input_format,
        interleave=True,
        data_name = "-".join(args.data),
        # measured on 1x_nav: max input content is 2826 tokens without memory and 3106 with 2 memory frames @160 px;
        # keep input + target <= 4096 so that no padded position exceeds the trained RoPE range
        max_source_length=args.max_source_length or (3050 if args.memory == 'none' else 2836 + 90 + args.mem_k * ((args.mem_res // 16) ** 2 + 2)),
        max_target_length=args.max_target_length or (800 if args.memory != 'none' else 850),
        mem_res=args.mem_res,
    )

    training_args.generation_max_new_tokens = max_target_length + 100
    print(f"generation_max_new_tokens: {training_args.generation_max_new_tokens}")

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience)
    label_pad_token_id = -100
    
    # Data collator: 
    from scripts.data_collator import customize_data_collator
    data_collator = customize_data_collator
    
    from scripts.customize_trainer import CustomizeSeq2SeqTrainer
    trainer_type = CustomizeSeq2SeqTrainer

    # fixme:
    training_args.label_smoothing_factor = 0.1

    if args.model in ['anole']:
        # used in evaluation when do_eval
        kwargs = dict()
        kwargs['multimodal_generation_mode'] = "interleaved-text-image"     # see L217 in wrapped_visualizer.py
        kwargs['stopping_criteria'] = StoppingCriteriaList([StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)])
        # used in evaluation during training
        training_args.customize_gen_stopping_criteria = StoppingCriteriaList([StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)])

    trainer = trainer_type(
        args=training_args,
        model=model,
        evaluator=VisualizationEvaluator(args=args, bin_step=training_cfg.get('action_token_generation', {}).get('bin_step', 0.01)),
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=processor,
        data_collator=data_collator,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['eval'] if 'eval' in tokenized_data.keys() else tokenized_data['test'],
        eval_examples=eval_split if 'eval' in tokenized_data.keys() else test_split,
        wandb_run_dir=(__import__('wandb').run.dir if ("wandb" in training_args.report_to and training_args.local_rank <= 0) else None),
        # callbacks=[early_stopping_callback],  # currently disabled early stopping for now
        callbacks=([__import__('uniwm.extra_tokens', fromlist=['ExtraTokensCallback']).ExtraTokensCallback(model)] if extra_tokens_enabled else [])
                  + ([__import__('uniwm.action_embed', fromlist=['Callback']).Callback(model)] if args.train_action_embeddings else [])
                 + [__import__('uniwm.callbacks', fromlist=['NanGuardCallback']).NanGuardCallback(model, min_scale=args.nan_guard_min_scale)],
        image_loss_func=not args.no_perceptual_loss, 
        action_cfg=training_cfg.get('action_token_generation'),
    )

    # if not dist.is_initialized() or dist.get_rank() == 0:
    #     print("DEBUG SAMPLE:", eval_dataset[0])

    for cb in trainer.callback_handler.callbacks:
        if cb.__class__.__name__ == 'NanGuardCallback':
            cb.scaler = getattr(trainer.accelerator, 'scaler', None)
    print('Trainer build successfully.')

    # for anole, there would be different inference mode. We use kwargs to pass these settings into the inference process.
    checkpoint = None
    if training_args.load_weights_from is not None:
        checkpoint = training_args.load_weights_from

    # NOTE: train the model with supervision
    if args.do_train:
        if checkpoint is not None and extra_tokens_enabled:
            from uniwm import extra_tokens
            print(f"[train] extra token rows loaded from {checkpoint}: {extra_tokens.load(model, checkpoint)}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        if args.train_action_embeddings and trainer.is_world_process_zero():
            from uniwm import action_embed
            print(f"[train] action-embedding rows saved to {training_args.output_dir} (md5 {action_embed.save(model, training_args.output_dir)})")
        if extra_tokens_enabled and trainer.is_world_process_zero():
            from uniwm import extra_tokens
            extra_tokens.save(model, training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = len(tokenized_data['train'])
        metrics["train_samples"] = min(max_train_samples, len(tokenized_data['train']))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.do_single_step_eval:
        logger.info("*** Single Step Eval ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            **kwargs
        )
        max_eval_samples = len(tokenized_data['eval'])
        metrics["eval_samples"] = min(max_eval_samples, len(tokenized_data['eval']))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_task_level_eval:
        logger.info("*** Task Level Eval ***")

        predict_results = trainer.predict(
            test_dataset=tokenized_data['test'],
            test_examples=tokenized_data['test'].dataset,
            metric_key_prefix="predict",
            predict_task = "task_level_evaluation",
            **kwargs
        )
        metrics = predict_results.metrics
        max_predict_samples = len(tokenized_data['test'])
        metrics["predict_samples"] = min(max_predict_samples, len(tokenized_data['test']))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
    if (args.do_action_likelihood or args.do_action_intervention or args.do_variance_decomp) and args.memory != 'none':
        raise SystemExit('action diagnostics build prompts without the memory block; run them only on --memory none checkpoints')
    for flag, task in (("do_action_likelihood", "action_likelihood"), ("do_action_intervention", "action_intervention"), ("do_variance_decomp", "variance_decomp")):
        if getattr(args, flag):
            logger.info(f"*** {task} ***")
            trainer.predict(test_dataset=tokenized_data['test'], test_examples=tokenized_data['test'].dataset,
                            metric_key_prefix="predict", predict_task=task, **kwargs)

    if args.do_action_probe:
        logger.info("*** Action-conditioning probe ***")
        trainer.predict(test_dataset=tokenized_data['test'], test_examples=tokenized_data['test'].dataset,
                        metric_key_prefix="predict", predict_task="action_probe", **kwargs)

    if args.do_rollout_eval:
        logger.info("*** Rollout ***")

        predict_results = trainer.predict(
            test_dataset=tokenized_data['test'],
            test_examples=tokenized_data['test'].dataset,
            metric_key_prefix="predict",
            predict_task = "rollout_evaluation",
            **kwargs
        )
        metrics = predict_results.metrics
        max_predict_samples = len(tokenized_data['test'])
        metrics["predict_samples"] = min(max_predict_samples, len(tokenized_data['test']))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)