import os
import torch
import json

from datasets import load_dataset, concatenate_datasets

from scripts.tokenized_dataset import AnoleTokenizedDataset
from scripts.interleaved_tokenized_dataset import InterleaveAnoleTokenizedDataset

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True


def load_data(dataset, data_dir, split_ratios=(0.9, 0.05, 0.05), max_train_trajs=0, stop_dist=0.0, stop_yaw=0.3, memory="none", mem_k=2, split_by="traj", mem_text="pose", closed_loop_extra="none", test_max_frames=0):
    data_list = []
    dataset_str = dataset[0]
    dataset_names = dataset_str.split(',')
    print(f"Attempting to load datasets: {dataset_names} from root: {data_dir}")

    for name in dataset_names:
        specific_data_path = os.path.join(data_dir, name)
        
        if not os.path.exists(specific_data_path):
            print(f"Warning: Path not found for dataset '{name}': {specific_data_path}. Skipping.")
            continue
        
        if os.environ.get('RANK', '0') == '0':
            print(f"--- Loading dataset: {name} ---")
        data = load_dataset(
            "scripts/navigation.py",
            trust_remote_code=True,
            tasks=['navigation_simulation'], 
            modes=['single_step_visualization', 'action_reasoning', 'task_level_evaluation'], 
            data_dir=specific_data_path,
            split_ratios=tuple(split_ratios),
            max_train_trajs=int(max_train_trajs),
            stop_dist=float(stop_dist),
            stop_yaw=float(stop_yaw),
            memory=str(memory),
            mem_k=int(mem_k),
            split_by=str(split_by),
            mem_text=str(mem_text),
            closed_loop_extra=str(closed_loop_extra),
            test_max_frames=int(test_max_frames),
        )
        if os.environ.get('RANK', '0') == '0':
            print(f"Loaded {name}: {len(data['train'])} training samples.")
        data_list.append(data)

    concatenate_data = dict()
    for k in data.keys():
        if k in ['train']:
            concatenate_data[k] = concatenate_datasets([i[k] for i in data_list])
        else:
            concatenate_data[k] = concatenate_datasets([
                i[k].shuffle(seed=42).select(range(min(800, len(i[k])))) for i in data_list
            ])
    
    from collections import Counter

    for split in ['train', 'validation', 'test']:
        if split in concatenate_data:
            if split == 'train':
                labs = concatenate_data[split]['label_text']
                n_stop = sum(1 for l in labs if l == 'Stop'); n_act = sum(1 for l in labs if l != '<image>')
                print(f"[data] Stop labels in train action samples: {n_stop}/{n_act} ({100.0 * n_stop / max(n_act, 1):.1f}%)")
            print(f"\n=== {split.upper()} split train_task distribution ===")
            task_counter = Counter(concatenate_data[split]['train_task'])
            for task, count in task_counter.items():
                print(f"{task}: {count}")

    return concatenate_data

def tokenize_dataset(train_split, eval_split, test_split, model, processor, **kwargs):
    tokenized_data = dict()

    data_name = kwargs.pop("data_name")

    max_source_length = int(kwargs.pop("max_source_length", 3050) or 3050)
    print(f"Max source length: {max_source_length}")

    max_target_length = int(kwargs.pop("max_target_length", 850) or 850)
    print(f"Max target length: {max_target_length}")

    if not kwargs["interleave"]:
        tokenized_dataset_type = AnoleTokenizedDataset
    else:
        tokenized_dataset_type = InterleaveAnoleTokenizedDataset

    if train_split:
        tokenized_train = tokenized_dataset_type(
            dataset=train_split,
            split='train',
            model=model,
            processor=processor,
            input_max_length=max_source_length, 
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['train'] = tokenized_train
    if eval_split:
        tokenized_eval = tokenized_dataset_type(
            dataset=eval_split,
            split='eval',
            model=model,
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['eval'] = tokenized_eval
    if test_split:
        tokenized_test = tokenized_dataset_type(
            dataset=test_split,
            split='test',
            model=model,
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['test'] = tokenized_test
    return tokenized_data, max_source_length, max_target_length


def get_image_token_num(model, processor, resolution):
    if hasattr(processor, 'image_seq_length'):
        return processor.image_seq_length
    elif hasattr(model, get_image_token_num):
        return model.get_image_token_num(resolution=resolution)
    else:
        raise NotImplementedError("Either model should have the get_image_token_num method or processor should have the iamge_seq_length property. ")