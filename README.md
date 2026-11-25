# UniWM reproducibility

Code for the capstone report *Adapting UniWM to Humanoid Navigation and Diagnosing Small-Angle Action Response* (Xu Zhu, University of Washington Tacoma, 2026). Repository: https://github.com/xuz44-jpg/uniwm-reproducibility

Clone this repository to use the source and configurations.

- `01_training/UniWM/` contains the UniWM model, training and evaluation entry points, requirements and analysis tools.
- `02_research/uniwm/uniwm_1x_pilot/` contains the 1X pilot code and checkpoint manifests.
- `02_research/uniwm/model_deployment/` contains the Habitat adapter training, generation and scoring tools, the NWM / RAE-NWM / UniWM backends and experiment configuration.
- `02_research/uniwm/output/aaai_workshop/` contains the Workshop sign analysis, its recorded score inputs and the checkpoint provenance.
- `02_research/uniwm/model_deployment/history_confirmation_v2/` (scripts `hc2_*.py`) contains the pre-registered NWM history replication at twelve new positions.
- `02_research/uniwm/model_deployment/uniwm_train_probe/` (scripts `uniwm_train_probe*.py`, `uniwm_multi_action_probe.py`) contains the training-input probe of the local UniWM reproduction.
- `02_research/uniwm/vlm_direction/` contains the Qwen3-VL-8B left/right judgment probe on the rendered Habitat references.

Run the recorded Workshop sign analysis from the repository root with `python3 02_research/uniwm/output/aaai_workshop/analyze_sign_breakdown.py`. For fresh inference, use the documented model versions and download the available checkpoints from [UW Google Drive](https://drive.google.com/drive/folders/1CcnmlKcC9r_SGBpFUr0XfGOUayTvr_Fx?usp=drive_link). Follow the component READMEs to install dependencies and point data, model and output paths to your environment. Fresh generation requires the original research datasets, pinned public base weights and a GPU environment. The A30 configs and protocols retain the recorded paths and hashes. All paths in configs and manifests are relative to the repository root; place the data and model directories under `external/` (or edit the relative paths) to rerun a frozen protocol.

## License and third-party code

The code in this repository is released under the MIT License (see `LICENSE`). Authors are listed in `AUTHORS`.

Some directories contain third-party code under their own terms, which are not changed by the MIT license above:

- `02_research/uniwm/model_deployment/repos/nwm/`: Navigation World Models, licensed under CC BY-NC 4.0 (non-commercial use only; see `LICENSE.md` in that directory).
- `02_research/uniwm/model_deployment/repos/raenwm/`: RAE-NWM, licensed under the MIT License (see its own `LICENSE`).
- `01_training/UniWM/uniwm/custom_chameleon.py` and `01_training/UniWM/uniwm/logging.py`: copyright notices in the files; licensed under the Apache License 2.0.
