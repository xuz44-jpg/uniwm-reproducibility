import os

import torch

from transformers import AutoProcessor

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def load_model(args, training_cfg):
    if args.model not in ['anole']:
        raise ValueError("Unsupported model type. ")

    model_ckpt_path = args.model_ckpt
    image_token_num = args.image_seq_length
    dtype = DTYPES[getattr(args, "precision", "bf16")]
    attn_impl = getattr(args, "attn_impl", "sdpa")           # V100 (sm70) has no flash-attn; sdpa picks the mem-efficient kernel
    base = getattr(args, "base_model", "leloy/Anole-7b-v0.1-hf")
    # Bind to this process's GPU explicitly: torchrun only exports LOCAL_RANK (it no longer passes --local_rank),
    # so device_map="cuda" would put every rank on the current default device (GPU 0) and OOM.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        # PEFT's infer_device() returns bare "cuda", which safetensors maps to cuda:0; every adapter load
        # (from_pretrained, load_adapter during Trainer resume) would then land on GPU 0 for all ranks.
        import peft.peft_model as _peft_model
        _peft_model.infer_device = lambda: f"cuda:{local_rank}"
    load_kwargs = dict(device_map={"": local_rank}, torch_dtype=dtype, attn_implementation=attn_impl, codebook_sim="mse")

    # Use MemoryBankAnoleForConditionalGeneration for memory-bank inference only
    if args.use_memory_bank_inference and args.do_task_level_eval and not args.do_train:
        from uniwm.memory_bank import MemoryBankAnoleForConditionalGeneration
        model = MemoryBankAnoleForConditionalGeneration.from_pretrained(base, **load_kwargs)
    else:
        from uniwm.wrapped_visualizer import AnoleforConditionalGeneration
        model = AnoleforConditionalGeneration.from_pretrained(base, **load_kwargs)

    # In inference the processor comes from the checkpoint so that the extended (action-token) vocab matches
    is_inference_only = (args.do_single_step_eval or args.do_task_level_eval or args.do_rollout_eval or getattr(args, "do_action_probe", False) or getattr(args, "do_action_likelihood", False) or getattr(args, "do_action_intervention", False) or getattr(args, "do_variance_decomp", False)) and not args.do_train and model_ckpt_path
    init_adapter = getattr(args, 'init_adapter', None)
    proc_src = model_ckpt_path if is_inference_only else (init_adapter if init_adapter else base)
    processor = AutoProcessor.from_pretrained(proc_src, image_seq_length=image_token_num)

    tokenizer_size = len(processor.tokenizer)
    if model.get_input_embeddings().weight.shape[0] != tokenizer_size:
        model.resize_token_embeddings(tokenizer_size)

    processor.tokenizer.padding_side = "left"

    # Monkey patch to ignore 'padding_side' kwarg in tokenizer (workaround for library bug)
    def patched_batch_encode_plus(self, *a, **kw):
        kw.pop('padding_side', None)
        return self.__original_batch_encode_plus(*a, **kw)

    if not hasattr(processor.tokenizer, '__original_batch_encode_plus'):
        processor.tokenizer.__original_batch_encode_plus = processor.tokenizer._batch_encode_plus
        processor.tokenizer._batch_encode_plus = patched_batch_encode_plus.__get__(processor.tokenizer)

    resolution = 448
    processor.image_processor.size = {"shortest_edge": resolution}
    processor.image_processor.crop_size = {"height": resolution, "width": resolution}

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.model.vqmodel.config.resolution = resolution
    model.model.vqmodel.quantize.quant_state_dims = [
        resolution // 2 ** (len(model.model.vqmodel.config.channel_multiplier) - 1)
    ] * 2

    args.sketch_resolution = resolution
    model.sketch_resolution = (resolution, resolution)
    model.image_token_num = image_token_num
    # keep the frozen VQ-VAE in fp32: fp16 encoding flips ~3% of the tokens vs fp32 (measured on V100);
    # get_image_tokens casts pixel_values to the VQ dtype, and autocast handles the eval-time decode.
    model.model.vqmodel.float()
    model.get_vis_codebook_sim()

    # fp16 overflow guard (measured on V100): only the last decoder layer's MLP exceeds the fp16 range
    # (intermediate ~1.4e5, output ~2.5e5 vs. 65504). Run those MLPs in fp32; everything else stays fp16.
    spec = str(getattr(args, "fp32_mlp_layers", "auto"))
    if spec == "auto":
        spec = "31" if dtype == torch.float16 else ""
    fp32_layers = [int(x) for x in spec.split(",") if x.strip() != ""]
    for li in fp32_layers:
        mlp = model.model.layers[li].mlp
        mlp.float()
        mlp._orig_forward = mlp.forward
        def _fp32_forward(x, _f=mlp._orig_forward):
            with torch.autocast(device_type="cuda", enabled=False):
                return _f(x.float())
        mlp.forward = _fp32_forward
    if fp32_layers:
        print(f"[load_model] MLP of decoder layers {fp32_layers} kept in fp32 (fp16 overflow guard)")
    gd_layer, gd_factor = int(getattr(args, 'grad_downscale_layer', 4)), float(getattr(args, 'grad_downscale', 64.0))
    if dtype == torch.float16 and gd_factor > 1.0 and getattr(args, 'do_train', False):
        from uniwm.grad_scale import install_grad_downscale
        install_grad_downscale(model, gd_layer, gd_factor)
        print(f"[load_model] gradient into decoder layers 0..{gd_layer} scaled by 1/{gd_factor:g} in backward (fp16 headroom)")

    from peft import LoraConfig, get_peft_model
    from peft.peft_model import PeftModel

    targets = [t.strip() for t in getattr(args, "lora_targets", "q_proj,k_proj,v_proj,o_proj").split(",") if t.strip()]
    lm_head_mode = getattr(args, "lm_head_mode", "new_rows")   # new_rows: train only the added-token rows; full: modules_to_save
    if is_inference_only:
        from uniwm import extra_tokens
        n_new = extra_tokens.n_new_in(model_ckpt_path)
        if n_new > 0:   # checkpoint trained in new_rows mode: wrap embed/lm_head and load the extra rows
            extra_tokens.install(model, n_new)
            extra_tokens.load(model, model_ckpt_path)
        # torch_device must carry the index: PEFT's infer_device() returns bare "cuda", and safetensors then
        # loads the (multi-GB) adapter onto GPU 0 for every rank.
        lora_model = PeftModel.from_pretrained(model, model_ckpt_path, is_trainable=False, torch_device=f"cuda:{local_rank}")
        from uniwm import action_embed
        _dg = action_embed.load_into_plain(lora_model, model_ckpt_path)
        if _dg:
            print(f"[load_model] action-token input embeddings loaded from {model_ckpt_path} (md5 {_dg})")
    elif init_adapter:
        # fine-tuning: start from a trained adapter (LoRA + lm_head copy) with a fresh optimizer / schedule
        lora_model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True, torch_device=f"cuda:{local_rank}")
        from uniwm import action_embed
        _dg = action_embed.load_into_plain(lora_model, init_adapter)
        if _dg:
            print(f"[load_model] action-token input embeddings loaded from {init_adapter} (md5 {_dg})")
        print(f"[load_model] initialised trainable adapter from {init_adapter}")
    else:
        config = LoraConfig(
            r=getattr(args, "lora_r", 16),
            lora_alpha=getattr(args, "lora_alpha", 32),
            target_modules=targets,
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            bias="none",
            modules_to_save=["lm_head"] if lm_head_mode == "full" else None,
        )
        lora_model = get_peft_model(model, config)
    if not is_inference_only:
        if getattr(args, 'lora_fp32', True) and dtype == torch.float16:
            # Compute the LoRA branch (A, B) in fp32 regardless of autocast: with fp16 matmuls the adapter
            # weight-gradients overflow at loss scale ~64 on action samples (measured), which pinned the
            # GradScaler at 32 and eventually collapsed it. PEFT casts the branch output back to fp16, so the
            # rest of the network is unchanged; the cost is negligible (rank 16).
            import torch.nn.functional as _F
            from peft.tuners.lora import LoraLayer
            def _fp32_forward(self, x):
                with torch.autocast(device_type='cuda', enabled=False):
                    return _F.linear(x.float(), self.weight, self.bias)
            n_patched = 0
            for m in lora_model.modules():
                if isinstance(m, LoraLayer):
                    for name in ('lora_A', 'lora_B'):
                        for lin in getattr(m, name).values():
                            lin.forward = _fp32_forward.__get__(lin); n_patched += 1
            print(f'[load_model] LoRA branch computed in fp32 ({n_patched} linears patched)')
        # fp16 AMP (GradScaler) needs fp32 master weights for every trainable parameter
        for p in lora_model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()

    n_train = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"[load_model] rank{local_rank} base={base} dtype={dtype} attn={attn_impl} vocab={tokenizer_size} "
          f"lora_r={getattr(args, 'lora_r', 16)} targets={targets} lm_head_mode={lm_head_mode} trainable_params={n_train / 1e6:.1f}M")

    return {'processor': processor, 'model': lora_model}
