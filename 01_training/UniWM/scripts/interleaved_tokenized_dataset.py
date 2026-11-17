import os
import json
import torch
import random
import copy

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from uniwm.pose_memory import MEM_PLACEHOLDER, MemoryEncoder, splice_memory

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class InterleaveAnoleTokenizedDataset(Dataset):
    def __init__(self,
                 dataset,
                 model,
                 processor,
                 split,
                 input_max_length,
                 label_max_length,
                 input_format,
                 **kwargs):
        self.model = model
        self.processor = processor
        self.split = split
        self.dataset = dataset

        self.input_max_length = input_max_length
        self.label_max_length = label_max_length
        
        self.label_processor = copy.deepcopy(self.processor)
        self.label_processor.tokenizer.padding_side = "right"

        self.processor.tokenizer.padding_side = "left"

        # dtype of the (frozen) VQ-VAE; fp16 on V100, bf16 on A100
        self.vq_dtype = next(self.model.model.model.vqmodel.parameters()).dtype

        # pose-indexed memory (I3): frames are VQ-encoded at a low resolution and spliced in at the placeholder token
        self.mem_res = int(kwargs.get("mem_res", 160) or 160)
        self.mem_encoder = None
        self.marker_id = self.processor.tokenizer.convert_tokens_to_ids(MEM_PLACEHOLDER)
        self.pad_id = self.processor.tokenizer.pad_token_id
        self.image_token_id = self.model.config.image_token_id
        self.boi, self.eoi = self.model.config.boi_token_id, self.model.config.eoi_token_id

    def __len__(self):
        return len(self.dataset)

    def _encode_main_images(self, tokenized, images):
        """Replace the processor's <image> placeholders by the VQ tokens of the 448x448 images (in place)."""
        if not images:
            return tokenized
        mask = tokenized['input_ids'] == self.image_token_id
        pv = tokenized.pop('pixel_values').to(self.model.device).to(self.vq_dtype)
        img_tokens = self.model.model.model.get_image_tokens(pv).to(torch.int64).to(tokenized['input_ids'].device).reshape(-1)
        expected = int(mask.sum())
        if img_tokens.shape[0] != expected:
            print(f"Shape mismatch: img_tokens={img_tokens.shape}, expected={expected}")
            img_tokens = img_tokens[:expected]
        tokenized['input_ids'][mask] = img_tokens
        return tokenized

    def _memory_tokens(self, item):
        paths = item.get('mem_img_paths') or []
        if not paths:
            return []
        if self.mem_encoder is None:
            self.mem_encoder = MemoryEncoder(self.model, self.processor, res=self.mem_res)
        return self.mem_encoder.encode([Image.open(p).convert("RGB") for p in paths])

    def _pad(self, ids, length, side):
        """Pad a [1, L] id tensor to `length` on the given side; returns (ids, attention_mask). Fails on overflow."""
        n = ids.shape[1]
        if n > length:
            raise ValueError(f"sequence of {n} tokens exceeds max length {length} (raise --max_source_length/--max_target_length)")
        pad = torch.full((1, length - n), self.pad_id, dtype=ids.dtype)
        att = torch.cat([torch.zeros(1, length - n, dtype=torch.long), torch.ones(1, n, dtype=torch.long)], 1) if side == "left" \
            else torch.cat([torch.ones(1, n, dtype=torch.long), torch.zeros(1, length - n, dtype=torch.long)], 1)
        ids = torch.cat([pad, ids], 1) if side == "left" else torch.cat([ids, pad], 1)
        return ids, att

    def __getitem__(self, idx):
        item = self.dataset[idx]

        input_text = item['input_text']
        input_imgs = [img.convert("RGB").resize((256, 256)) for img in item['input_imgs']]
        label_text = item['label_text']
        label_imgs = [img.convert("RGB").resize((256, 256)) for img in item['label_imgs']]

        input_imgs = [self.augment_image(img, random.randint(0, 8)) for img in input_imgs]

        # --- input: tokenize without padding, fill in image tokens, splice memory frames, then left-pad ---
        tokenized_input = self.processor([input_text], images=input_imgs if input_imgs else None, padding=False, truncation=False, return_tensors="pt")
        tokenized_input = self._encode_main_images(tokenized_input, input_imgs)
        mem_tokens = self._memory_tokens(item)
        if mem_tokens or int((tokenized_input['input_ids'] == self.marker_id).sum()) > 0:
            tokenized_input['input_ids'] = splice_memory(tokenized_input['input_ids'], mem_tokens, self.marker_id, self.boi, self.eoi)
        input_ids, input_att = self._pad(tokenized_input['input_ids'], self.input_max_length, "left")

        if self.split not in ['train']:
            return {"input_ids": input_ids, "attention_mask": input_att}

        # --- label: tokenize without padding, drop <s>, fill in image tokens, right-pad ---
        tokenized_label = self.label_processor([label_text], images=label_imgs if label_imgs else None, padding=False, truncation=False, return_tensors="pt")
        tokenized_label['input_ids'] = tokenized_label['input_ids'][:, 1:]
        tokenized_label = self._encode_main_images(tokenized_label, label_imgs)
        label_ids, label_att = self._pad(tokenized_label['input_ids'], self.label_max_length, "right")

        labels = torch.cat((torch.full(input_ids.shape, -100), label_ids), 1)
        labels[labels == self.pad_id] = -100

        return {
            "input_ids": torch.cat((input_ids, label_ids), 1),
            "attention_mask": torch.cat([input_att, label_att], 1),
            "labels": labels,
        }

    def augment_image(self, reconstructed_img, k):
        for i in range(k):
            tokenized_image = self.processor(
                text="<image>", 
                images=reconstructed_img,
                return_tensors='pt'
            )['pixel_values'].to(self.vq_dtype).to(self.model.device)
            img_tokens = self.model.model.model.get_image_tokens(tokenized_image)
            reconstructed_img_pixels = self.model.model.decode_image_tokens(img_tokens)
            reconstructed_img = self.processor.postprocess_pixel_values(reconstructed_img_pixels).squeeze()
            reconstructed_img = Image.fromarray(reconstructed_img.permute(1, 2, 0).detach().cpu().numpy())
        return reconstructed_img
