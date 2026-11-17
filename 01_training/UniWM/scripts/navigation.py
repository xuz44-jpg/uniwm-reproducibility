import os
import sys
import pickle
import random
import numpy as np

# The released 1XHumanoid traj_data.pkl files were pickled with numpy 2 (they reference `numpy._core`);
# make them loadable on numpy 1.x environments (e.g. the pinned numpy==1.24.3).
if not hasattr(np, "_core"):
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    for _sub in ("multiarray", "numeric", "_multiarray_umath", "umath"):
        sys.modules.setdefault(f"numpy._core.{_sub}", getattr(_np_core, _sub, _np_core.multiarray))
import datasets
from PIL import Image
import math
from scripts.prompt_builder import build_action_prompt, build_viz_prompt

from uniwm.pose_memory import retrieve as mem_retrieve, relative_pose as mem_relative_pose, memory_block as mem_block
from scripts.action_utils import (
    DATASET_RANGES,
    DEFAULT_RANGES,
    calculate_action_delta,
    action_to_text
)


class NavigationConfig(datasets.BuilderConfig):
    def __init__(self, tasks, modes, data_dir, split_ratios=(0.9, 0.05, 0.05), max_train_trajs=0, stop_dist=0.0, stop_yaw=0.3, memory="none", mem_k=2, split_by="traj", mem_text="pose", closed_loop_extra="none", test_max_frames=0, **kwargs):
        super(NavigationConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir
        self.split_ratios = tuple(split_ratios)      # train / dev / test fractions of the (seeded) shuffled trajectory list
        self.max_train_trajs = int(max_train_trajs)  # >0: keep only the first N training trajectories (few-shot recipes)
        # action label becomes 'Stop' once the agent is within stop_dist of the goal (0 = only the last frame, the
        # released behaviour, which makes Stop 1/24 of the action labels and the model never learns to emit it)
        self.stop_dist = float(stop_dist)
        self.stop_yaw = float(stop_yaw)      # heading must also be within stop_yaw rad of the goal heading (in-place rotations)
        # split unit: 'traj' shuffles trajectories (released behaviour; chunks cut from the same recording can land in
        # train and test), 'segment' keeps all chunks of one (shard, segment) recording in the same split (no leakage)
        self.split_by = str(split_by)
        self.memory = str(memory)            # none | recent | pose : pose-indexed memory frames appended to the prompts (I3)
        self.mem_k = int(mem_k)
        self.mem_text = str(mem_text)        # pose | none : whether the memory block states each frame's relative pose
        self.closed_loop_extra = str(closed_loop_extra)   # none | dev : also emit task-level (closed-loop) items for the dev trajectories
        self.test_max_frames = int(test_max_frames)      # >0: truncate each closed-loop task to its first N frames (goal = frame N-1), e.g. 15 for the released 10-step protocol


class NavigationDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = NavigationConfig
    BUILDER_CONFIGS = [
        NavigationConfig(
            name="processed_navigation",
            version="1.0.0",
            description="Refactored navigation dataset.",
            tasks=["navigation_simulation"],
            modes=["single_step_visualization", "action_reasoning", "task_level_evaluation"],
            data_dir="go_stanford",
        )
    ]
    DEFAULT_CONFIG_NAME = "processed_navigation"

    def _info(self):
        features = datasets.Features({
            'idx': datasets.Value('int32'),
            "input_text": datasets.Value("string"),
            "input_imgs": datasets.Sequence(datasets.Image()),
            'gt_next_action': datasets.Value("string"),
            "label_text": datasets.Value("string"),
            "label_imgs": datasets.Sequence(datasets.Image()),
            "label_img_paths": datasets.Sequence(datasets.Value("string")),
            "input_img_paths": datasets.Sequence(datasets.Value("string")),
            'task': datasets.Value('string'),
            'train_task': datasets.Value("string"),
            'coords': datasets.Sequence(datasets.Sequence(datasets.Value("float32"))), # sequence of [x,y,yaw]
            "action_vector": datasets.Sequence(datasets.Value("float32")),
            "mem_img_paths": datasets.Sequence(datasets.Value("string")),           # memory frames (I3), most relevant first
            "mem_rel_poses": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),  # (dx, dy, dyaw) of each memory frame relative to the current pose
        })
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        data_root = self.config.data_dir
        all_traj_names = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        random.seed(42)
        if self.config.split_by == "segment":
            # group by source recording; a trajectory without shard/segment metadata is its own group
            groups = {}
            for name in all_traj_names:
                key = name
                try:
                    with open(os.path.join(data_root, name, "traj_data.pkl"), "rb") as f:
                        d = pickle.load(f)
                    if "shard" in d and "segment" in d:
                        key = (int(d["shard"]), int(d["segment"]))
                except Exception:
                    pass
                groups.setdefault(key, []).append(name)
            group_list = list(groups.values())
            random.shuffle(group_list)
            all_traj_names = [n for g in group_list for n in g]   # contiguous groups; the cut points below fall between groups
            print(f"[navigation] split_by=segment: {len(group_list)} recordings, {len(all_traj_names)} trajectories")
        else:
            random.shuffle(all_traj_names)

        total_used_ratio = 1  # Use 100% of the data, change to 0.5 to use 50%
        train_ratio, dev_ratio, test_ratio = self.config.split_ratios
        # total_used_ratio = 0.002  # Use 100% of the data, change to 0.5 to use 50%
        # train_ratio, dev_ratio, test_ratio = 0.5, 0.3, 0.2
        total_count = int(len(all_traj_names) * total_used_ratio)
        split_point1 = int(total_count * train_ratio)
        split_point2 = split_point1 + int(total_count * dev_ratio)
        if self.config.split_by == "segment":
            # move each cut point forward to the next group boundary so no recording straddles two splits
            bounds, pos = [], 0
            for g in group_list:
                pos += len(g); bounds.append(pos)
            split_point1 = min(b for b in bounds if b >= split_point1)
            split_point2 = min(b for b in bounds if b >= max(split_point2, split_point1 + 1))
        train_names = all_traj_names[:split_point1]
        if self.config.max_train_trajs > 0:
            train_names = train_names[:self.config.max_train_trajs]
        print(f"[navigation] {data_root}: train={len(train_names)} dev={split_point2 - split_point1} test={total_count - split_point2} trajectories")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    "traj_dirs": [os.path.join(data_root, name) for name in train_names],
                    "split": datasets.Split.TRAIN 
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={
                    "traj_dirs": [os.path.join(data_root, name) for name in all_traj_names[split_point1:split_point2]],
                    "split": datasets.Split.VALIDATION
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={
                    # closed_loop_extra=dev: the dev trajectories (also unseen in training) are appended to the test
                    # split so the closed-loop evaluation runs on ~50 episodes instead of ~25
                    "traj_dirs": [os.path.join(data_root, name) for name in all_traj_names[split_point2:total_count]]
                                 + ([os.path.join(data_root, name) for name in all_traj_names[split_point1:split_point2]] if self.config.closed_loop_extra == "dev" else []),
                    "split": datasets.Split.TEST
                }
            ),
        ]

    def _load_and_validate_trajectory(self, traj_dir):
        """Loads and validates data for a single trajectory."""
        pkl_path = os.path.join(traj_dir, "traj_data.pkl")
        if not os.path.exists(pkl_path): return None

        try:
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            
            positions = np.array(pkl_data['position'], dtype=float)
            yaws = np.array(pkl_data['yaw'], dtype=float).reshape(-1, 1)

            if not (positions.ndim == 2 and positions.shape[1] == 2 and yaws.ndim == 2 and yaws.shape[1] == 1 and positions.shape[0] == yaws.shape[0]):
                return None
            
            traj_len = positions.shape[0]

            image_files = sorted([f for f in os.listdir(traj_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))], key=lambda f: int(os.path.splitext(f)[0]))
            image_paths = [os.path.join(traj_dir, f) for f in image_files]

            if len(image_paths) != traj_len:
                return None

            states_xy_yaw = np.concatenate([positions, yaws], axis=1).tolist()

            return {
                "pkl_data": pkl_data,
                "image_paths": image_paths,
                "traj_len": traj_len,
                "states_xy_yaw": states_xy_yaw
            }
        except (IOError, pickle.UnpicklingError, KeyError, ValueError, ModuleNotFoundError, AttributeError):
            return None

    def _prepare_actions(self, pkl_data, states_xy_yaw):
        delta_action_key = "delta"
        
        delta_actions = pkl_data.get(delta_action_key)
        if delta_actions is not None:
            delta_actions = np.array(delta_actions)
            if delta_actions.ndim == 1:
                delta_actions = delta_actions.reshape(-1, 3)
            # Removed _MAX_STEPS_PER_TRAJ logic
            return ["input_img"] + delta_actions.tolist()

        numeric_actions = [calculate_action_delta(states_xy_yaw[i], states_xy_yaw[i+1]) for i in range(len(states_xy_yaw) - 1)]
        numeric_actions.append([0.0, 0.0, 0.0])
        return ["input_img"] + numeric_actions
    

    def _get_dataset_ranges(self, image_path):
        """ Determine the dataset ranges based on the image path."""
        for name, ranges in DATASET_RANGES.items():
            if f"/{name}/" in image_path.replace("\\", "/"):
                return ranges
        return DEFAULT_RANGES
    

    def _memory_for_step(self, k, all_image_paths, states_xy_yaw):
        """Memory frames for step k: retrieved among frames 0..k-1 by GT pose (training-time teacher forcing)."""
        if self.config.memory == "none" or k == 0:
            return "", [], []
        cur = states_xy_yaw[k]
        idx = mem_retrieve(cur, states_xy_yaw[:k], self.config.mem_k, self.config.memory)
        rel = [mem_relative_pose(cur, states_xy_yaw[i]) for i in idx]
        return mem_block(rel, self.config.mem_text), [all_image_paths[i] for i in idx], [list(r) for r in rel]

####Key 1
    def _prepare_visualization_sample(self, k, len_seq, all_images, all_image_paths, actions, states_xy_yaw, ranges):
        start_img = all_images[0]
        goal_img = all_images[len_seq - 1]
        current_img = all_images[k]
        next_img = all_images[k + 1] if k + 1 < len_seq else all_images[k]

        start_path = all_image_paths[0]
        goal_path = all_image_paths[len_seq - 1]
        current_path = all_image_paths[k]
        next_path = all_image_paths[k + 1] if k + 1 < len_seq else all_image_paths[k]

        current_action = action_to_text(actions[k+1]) if k < len_seq - 1 else "Stop"

        start_pose = states_xy_yaw[0]  # [x, y, yaw]
        start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"

        # history_action_seq_text = "\n".join([action_to_text(act) for act in actions[0:k-1]])

        # if k == len_seq - 1:
        #     next_action_text = "Stop"
        # else:
        #     next_action_text = action_to_text(actions[k + 1])
        
        mem_text, mem_paths, mem_rel = self._memory_for_step(k, all_image_paths, states_xy_yaw)
        input_text = build_viz_prompt(current_action, start_pose_str, memory_block=mem_text)

        return {
            "input_text": input_text,
            "input_imgs": [start_img, goal_img, current_img],
            "input_img_paths": [start_path, goal_path, current_path],
            "gt_next_action": f"Next Action: {current_action}",
            "label_text": "<image>",
            "label_imgs": [next_img],
            "label_img_paths": [next_path],
            "train_task": "single_step_visualization",
            "coords": states_xy_yaw[:k+1],
            "action_vector": [],
            "mem_img_paths": mem_paths,
            "mem_rel_poses": mem_rel,
        }

# ###Key 2
    def _prepare_reasoning_sample(self, k, len_seq, all_images, all_image_paths, actions, states_xy_yaw, ranges):
        start_img = all_images[0]
        goal_img = all_images[len_seq - 1]
        current_img = all_images[k]
        next_img = all_images[k + 1] if k + 1 < len_seq else all_images[k]

        start_path = all_image_paths[0]
        goal_path = all_image_paths[len_seq - 1]
        current_path = all_image_paths[k]
        next_path = all_image_paths[k + 1] if k + 1 < len_seq else all_image_paths[k]

        start_pose = states_xy_yaw[0]  # [x, y, yaw]
        start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"
        
        action_history_text = ""
        for i in range(k):
            action_history_text += f"{action_to_text(actions[i+1])}"
        
        dxy_range = ranges['dxy']
        dyaw_range = ranges['dyaw']
        mem_text, mem_paths, mem_rel = self._memory_for_step(k, all_image_paths, states_xy_yaw)
        input_text = build_action_prompt(start_pose_str, dxy_range, dyaw_range, memory_block=mem_text)
        
        goal_xy = np.asarray(states_xy_yaw[len_seq - 1][:2]); cur_xy = np.asarray(states_xy_yaw[k][:2])
        dyaw_goal = abs((states_xy_yaw[k][2] - states_xy_yaw[len_seq - 1][2] + math.pi) % (2 * math.pi) - math.pi)
        is_stop = (k >= len_seq - 1) or (self.config.stop_dist > 0 and float(np.linalg.norm(goal_xy - cur_xy)) < self.config.stop_dist
                                         and dyaw_goal < self.config.stop_yaw)
        return {
            "input_text": input_text,
            "input_imgs": [start_img, goal_img, current_img],
            "input_img_paths": [start_path, goal_path, current_path],
            "gt_next_action": "",
            "label_text": "Stop" if is_stop else action_to_text(actions[k+1]),
            "label_imgs": [],
            "label_img_paths": [],
            "train_task": "action_reasoning",
            "coords": states_xy_yaw[:k+1],
            "action_vector": actions[k+1] if k < len_seq - 1 else [0.0, 0.0, 0.0],
            "mem_img_paths": mem_paths,
            "mem_rel_poses": mem_rel,
        }
    
    def _prepare_task_level_sample(self, len_seq, all_images, all_image_paths, actions, states_xy_yaw, ranges):
        start_img = all_images[0]
        goal_img = all_images[len_seq - 1]
        start_path = all_image_paths[0]
        goal_path = all_image_paths[len_seq - 1]
        start_pose = states_xy_yaw[0]

        start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"

        input_text = (
            "Task: Full Navigation Task\n"
            "Description: Navigate from the start observation to the goal observation. "
            "Start Observation: <image>\nGoal Observation: <image>\n"
            f"{start_pose_str}"
        )

        # The 'label' for this task is the final goal image, to be compared against the final predicted image.
        # We also pass the ground truth trajectory for more detailed analysis if needed.
        return {
            "input_text": input_text,
            "input_imgs": [start_img, goal_img],
            "input_img_paths": [start_path, goal_path],
            "label_text": "",
            "label_imgs": [goal_img],
            "label_img_paths": [goal_path],
            "train_task": "task_level_evaluation",
            "coords": states_xy_yaw,
            "action_vector": [],
            "ranges": ranges,
            "mem_img_paths": [],
            "mem_rel_poses": [],
        }

    def _generate_examples(self, traj_dirs, split):
        global_idx = 0
        for traj_dir in traj_dirs:
            # 1. Load and validate data for one trajectory
            traj_data = self._load_and_validate_trajectory(traj_dir)
            if not traj_data:
                continue
            trajectory_ranges = self._get_dataset_ranges(traj_data["image_paths"][0])

            # 2. Prepare the definitive list of actions for this trajectory
            actions = self._prepare_actions(
                traj_data["pkl_data"],
                traj_data["states_xy_yaw"],
            )

            # 3. Reference images by path. datasets.Image() decodes them lazily on access, so the Arrow cache stays
            #    tiny (the old version re-encoded every 256x256 frame as PNG bytes, ~170 KB per sample); the 256x256
            #    squash-resize now happens in the tokenized dataset / evaluation code.
            all_images = list(traj_data["image_paths"])
            
            if "task_level_evaluation" in self.config.modes and split == datasets.Split.TEST:
                n_task = min(len(all_images), self.config.test_max_frames) if self.config.test_max_frames > 0 else len(all_images)
                sample = self._prepare_task_level_sample(n_task, all_images[:n_task],
                                                         traj_data["image_paths"][:n_task], actions[:n_task],
                                                         traj_data["states_xy_yaw"][:n_task], trajectory_ranges)
                sample['task'] = self.config.tasks[0]
                sample['idx'] = global_idx
                yield global_idx, sample
                global_idx += 1

            # Use the full length of all_images directly
            # 4. Generate samples
            for k in range(len(all_images)):
                # Generate a sample for the reasoning task
                if "action_reasoning" in self.config.modes and split != datasets.Split.TEST:
                    sample = self._prepare_reasoning_sample(k, len(all_images), all_images,
                                                            traj_data["image_paths"], actions,
                                                            traj_data["states_xy_yaw"], trajectory_ranges)

                    sample['task'] = self.config.tasks[0]
                    sample['idx'] = global_idx
                    yield global_idx, sample
                    global_idx += 1
                    

                # Generate a sample for the visualization task
                if "single_step_visualization" in self.config.modes and split != datasets.Split.TEST:
                    sample = self._prepare_visualization_sample(k, len(all_images), all_images,
                                                                traj_data["image_paths"], actions,
                                                                traj_data["states_xy_yaw"], trajectory_ranges)
                    sample['task'] = self.config.tasks[0]
                    sample['idx'] = global_idx
                    yield global_idx, sample
                    global_idx += 1
