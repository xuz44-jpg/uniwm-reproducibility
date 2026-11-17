import torch
import os
import torch.nn as nn

from typing import Optional, Literal, List, Tuple

from PIL import Image
import numpy as np
import torch.nn.functional as F

from .custom_chameleon import ChameleonForConditionalGeneration

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor


def pairwise_euclidean_distance(tensor):
    # tensor: (token_num, embedding_dim)
    # Calculate squared norms of each row (token)
    squared_norms = torch.sum(tensor**2, dim=1, keepdim=True)  # (token_num, 1)

    # Use broadcasting to calculate pairwise squared Euclidean distances
    distances_squared = squared_norms + squared_norms.T - 2 * torch.matmul(tensor, tensor.T)

    # Due to possible floating-point precision issues, clamp to avoid negative values
    distances_squared = torch.clamp(distances_squared, min=0.0)

    # Calculate Euclidean distance
    distances = torch.sqrt(distances_squared)
    
    return distances

class GeneratedImageStructureLogitsProcessor(LogitsProcessor):
    """Enforce `boi + image_seq_length image tokens + eoi` for images generated after the prompt.

    Unlike the fork's relative-offset/window processors this never inspects the prompt, so prompts may contain
    images of any length (e.g. reduced-resolution memory frames). Outside a generated image: image tokens and
    eoi are suppressed (boi is suppressed too when fewer than image_seq_length + 1 positions remain). Inside a
    generated image: only image tokens until image_seq_length have been produced, then only eoi.
    """

    def __init__(self, boi, eoi, image_token_ids, image_seq_length, prompt_len, max_length=None, device="cpu"):
        self.boi, self.eoi, self.n, self.prompt_len, self.max_length = int(boi), int(eoi), int(image_seq_length), int(prompt_len), max_length
        self.image_token_ids = torch.tensor(list(image_token_ids), device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen = input_ids[:, self.prompt_len:]
        neg = torch.finfo(scores.dtype).min
        out = scores.clone()
        for b in range(gen.shape[0]):
            row = gen[b]
            boi_pos = (row == self.boi).nonzero().flatten()
            inside = False
            if boi_pos.numel() > 0:
                last = int(boi_pos[-1])
                inside = not bool((row[last + 1:] == self.eoi).any())
            if inside:
                produced = int(row.shape[0] - last - 1)
                mask = torch.ones_like(out[b], dtype=torch.bool)
                if produced < self.n:
                    mask[self.image_token_ids] = False   # only image tokens
                else:
                    mask[self.eoi] = False               # exactly image_seq_length tokens: close the image
                out[b] = out[b].masked_fill(mask, neg)
            else:
                out[b, self.image_token_ids] = neg
                out[b, self.eoi] = neg
                if self.max_length is not None and input_ids.shape[1] + self.n + 1 >= self.max_length:
                    out[b, self.boi] = neg               # no room left for a complete image
        return out


class AnoleforConditionalGeneration(ChameleonForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.image_decoder = None   # for having the image_decoder property, L516 in customize_trainer.py
        self.generate_with_embeds = False

        self.image_postprocess = True   # for postprocessing the pixel value with processor

        self.sketch_resolution = (self.model.vqmodel.config.resolution, self.model.vqmodel.config.resolution) # fixme
        
        self.image_token_num = 784

        self.bpe_indices = self.model.vocabulary_mapping.image_token_ids
        self.img_indices = [self.model.vocabulary_mapping.bpe2img[i] for i in self.bpe_indices]

        if "codebook_sim" in kwargs:
            self.codebook_sim = kwargs['codebook_sim']
        else:
            self.codebook_sim = None
        
        self.global_memory_manager = None
        self.current_step = 0
    
    def get_vis_codebook_sim(self):
        if self.codebook_sim == "mse":
            self.codebook_sim_matrix = pairwise_euclidean_distance(self.model.vqmodel.quantize.embedding.weight.data.to(torch.float64)).to(torch.float32)
        else:
            self.codebook_sim_matrix = None
    
    def set_global_memory_manager(self, global_memory_manager):

        self.global_memory_manager = global_memory_manager
 
        super().set_global_memory_manager(global_memory_manager)
    
    def update_step(self, step):

        self.current_step = step

        super().update_step(step)
    
    def reset_memory_bank(self):

        if hasattr(self, 'memory_bank_initialized'):
            self.memory_bank_initialized = False
        
        if hasattr(super(), 'reset_memory_bank'):
            super().reset_memory_bank()
        
        if hasattr(self, 'model') and hasattr(self.model, 'reset_memory_bank'):
            self.model.reset_memory_bank()
    
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            multimodal_generation_mode: Optional[
                Literal["text-only", "image-only", "interleaved-text-image", "unrestricted"]
            ] = "interleaved-text-image",
            **kwargs,
    ):
        # The fork's "interleaved-text-image" constraints look for a begin-image token in the last
        # image_seq_length tokens of the *whole* sequence, i.e. also inside the prompt. Any image in the prompt
        # that is not exactly image_seq_length tokens long (the 100-token memory frames of --memory pose/recent)
        # or that ends within that window breaks them: the sampler is forced onto image tokens without a
        # begin-image token and can never close the image. We therefore run the fork in "unrestricted" mode and
        # enforce the same structure ourselves, tracking only the images the model generates after the prompt.
        if multimodal_generation_mode == "interleaved-text-image" and "input_ids" in kwargs:
            prompt_len = kwargs["input_ids"].shape[-1]
            max_new = kwargs.get("max_new_tokens")
            proc = GeneratedImageStructureLogitsProcessor(
                boi=self.vocabulary_mapping.boi_token_id, eoi=self.vocabulary_mapping.eoi_token_id,
                image_token_ids=self.vocabulary_mapping.image_token_ids, image_seq_length=self.model.image_seq_length,
                prompt_len=prompt_len, max_length=(prompt_len + max_new) if max_new else None,
                device=kwargs["input_ids"].device)
            logits_processor = LogitsProcessorList(list(logits_processor or []) + [proc])
            multimodal_generation_mode = "unrestricted"
        generate_ids = super().generate(
            inputs=inputs, 
            generation_config=generation_config, 
            logits_processor=logits_processor, 
            multimodal_generation_mode=multimodal_generation_mode,
            do_sample=kwargs.pop("do_sample", True),   # greedy action decoding (--action_greedy) passes False
            **kwargs
        )

        if multimodal_generation_mode == "text-only":
            return generate_ids[:, kwargs["input_ids"].shape[-1]:], None
        
        elif multimodal_generation_mode == "image-only":
            response_ids = generate_ids[:, kwargs["input_ids"].shape[-1]:]
            return response_ids, None
        
        elif multimodal_generation_mode in ["interleaved-text-image", "unrestricted"]:
            response_ids = generate_ids[:, kwargs["input_ids"].shape[-1]:]
            return response_ids, None
    

def split_token_sequence(
    tokens: torch.LongTensor,
    image_seq_length: int, 
    boi: int,
    eoi: int,
    max_length: int,
    pad_token_id: int
) -> List[Tuple[str, torch.LongTensor]]:
    """
    Split a sequence of tokens into text and image segments.
    
    Args:
        tokens (torch.LongTensor): The token sequence.
        boi (int): Begin of image token.
        eoi (int): End of image token.
    
    Returns:
        List[Tuple[str, torch.LongTensor]]: List of tuples indicating segment type and tokens.
    """
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"
    
    device = tokens.device
    tokens = tokens[0]  # remove batch dimension
    tokens = tokens.to(device)
    segments = []
    current_segment = []
    in_image_seg = False

    for token in tokens:
        if token == boi:
            # if entering an image segment, save the current text segment (if any)
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
            in_image_seg = True
        elif token == eoi and in_image_seg:
            # if exiting an image segment, save the current image segment
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)
    # save any remaining tokens
    if current_segment:
        if in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))

    generated_imgs = []
    generated_texts = []
    for seg_id, (seg_type, seg_tokens) in enumerate(segments):
        if seg_type == "image_seg":
            assert seg_tokens.shape[1] == image_seq_length
            generated_imgs.append(seg_tokens)
        else:
            assert seg_type == "text_seg"
            generated_texts.append(seg_tokens.view(-1))

    text_tokens = torch.cat(generated_texts)
    if max_length > text_tokens.shape[-1]:
        text_tokens = torch.cat((text_tokens, torch.full((max_length-text_tokens.shape[-1],), fill_value=pad_token_id, device=text_tokens.device))).unsqueeze(0)
    elif max_length < text_tokens.shape[-1]:
        text_tokens = text_tokens.unsqueeze(0)[:, :max_length]
    else:
        text_tokens = text_tokens.unsqueeze(0)
    return {
        "texts": text_tokens,
        "images": generated_imgs if len(generated_imgs) != 0 else None
    }