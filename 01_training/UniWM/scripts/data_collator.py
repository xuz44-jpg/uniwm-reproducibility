from typing import Any, Dict, List, Mapping, NewType
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from transformers import default_data_collator
from PIL import Image

from torchvision.transforms.functional import pil_to_tensor

from collections.abc import Mapping

InputDataClass = NewType("InputDataClass", Any)


def customize_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import torch
    
    if not features or len(features) == 0:
        pass
        return {}
    
    original_length = len(features)
    features = [f for f in features if f is not None]
    if len(features) != original_length:
        pass
    
    if not features:
        pass
        return {}
    
    if not isinstance(features[0], (dict, Mapping)) and hasattr(features[0], '__dict__'):
        pass
        features = [vars(f) for f in features if hasattr(f, '__dict__')]
        pass
    
    if not features or features[0] is None:
        pass
        return {}
        
    first = features[0]

    if not isinstance(first, (dict, Mapping)):
        pass
        return {}
        
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k in ("pixel_values"):
            if len(v.shape) == 4:
                batch[k] = torch.stack([f[k].squeeze() for f in features])
            else:
                batch[k] = torch.stack([f[k] for f in features])
        elif k not in ("label", "label_ids") and v is not None:
            if isinstance(v, torch.Tensor):
                from torch.nn.utils.rnn import pad_sequence
                tensors = [f[k].squeeze() for f in features]
                batch[k] = pad_sequence(tensors, batch_first=True)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            elif hasattr(v, "im"):
                batch_list = []
                for f in features:
                    temp = pil_to_tensor(f[k])
                    # mask = (temp.permute((1, 2, 0))[..., :3] == torch.tensor([0, 0, 0])).all(-1)
                    # temp[-1][mask] = 0
                    batch_list.append(temp.div(255))
                
                batch[k] = torch.stack(batch_list)
            elif k == "ranges":
                batch[k] = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    
    pass
    return batch