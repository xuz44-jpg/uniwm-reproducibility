from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments

# NOTE: The transformers package is updated during our work
# , our "WrappedSeq2SeqTrainingArguments" is now a feature named "Seq2seqTrainingArguments"

@dataclass
class WrappedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use a `sortish sampler` or not. Only possible if the underlying datasets are `Seq2SeqDataset` for
        now but will become generally available in the near future.

        It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness for
        the training set.
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    cfg: str = None
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_new_tokens: int = field(
        default=100, metadata={
            "help": "The max_length to use on each evaluation loop when predict_with_generate=True."
                    " Will default to the max_new_tokens value of the model configuration."}
    )
    generation_num_beams: int = field(
        default=4, metadata={
            "help": "The num_beams to use on each evaluation loop when predict_with_generate=True."
                    " Will default to the num_beams value of the model configuration."}
    )
    load_weights_from: Optional[str] = field(
        default=None, metadata={
            "help": "The directory to load the model weights from."}
    )
    customize_gen_stopping_criteria: Optional[list] = field(
        default=None, metadata={
            "help": "Stopping criteria for generation. "}
    )
    eval_max_steps: int = field(default=10, metadata={"help": "max closed-loop steps per trajectory in task-level eval"})
    action_samples: int = field(default=1, metadata={"help": "closed loop: sample N candidate actions per step and execute the component-wise median (1 = single sample)"})
    gate_std: float = field(default=0.0, metadata={"help": "disagreement gate threshold on the candidates' per-component std (0 = off)"})
    gate_scale: float = field(default=0.5, metadata={"help": "step scale when the disagreement gate fires"})
    mem_text: str = field(default="pose", metadata={"help": "memory block text: pose | none"})
    skip_existing: bool = field(default=False, metadata={"help": "closed loop: re-use episodes whose eval_result.json already exists in the output directory"})
    action_greedy: bool = field(default=False, metadata={"help": "closed loop: greedy (argmax) action decoding instead of sampling; imagined frames are still sampled"})
    mem_ablate_image: str = field(default="", metadata={"help": "closed loop: replace every memory frame by this image (inference-time ablation)"})
    frame_samples: int = field(default=1, metadata={"help": "closed loop: sample N imagined frames per step and keep the LPIPS medoid (1 = single sample)"})
    probe_repeats: int = field(default=2, metadata={"help": "action probe: generations per condition"})
    al_stride: int = field(default=1, metadata={"help": "likelihood: score every n-th frame"})
    ai_samples: int = field(default=5, metadata={"help": "intervention: sampled actions per query"})
    ai_max_steps: int = field(default=0, metadata={"help": "intervention: cap on steps per task (0 = whole task)"})
    vd_states: int = field(default=5, metadata={"help": "variance decomposition: states per trajectory"})
    vd_frames: int = field(default=4, metadata={"help": "variance decomposition: imagined frames per state"})
    vd_actions: int = field(default=5, metadata={"help": "variance decomposition: sampled actions per frame"})
    memory: str = field(default="none", metadata={"help": "none | recent | pose (I3 pose-indexed memory)"})
    mem_k: int = field(default=2, metadata={"help": "memory frames per prompt"})
    mem_res: int = field(default=160, metadata={"help": "memory frame resolution"})
    use_memory_bank_inference: bool = field(
        default=False,
        metadata={"help": "Enable memory bank during inference."}
    )
    