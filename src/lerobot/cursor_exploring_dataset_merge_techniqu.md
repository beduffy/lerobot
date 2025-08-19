# Exploring dataset merge techniques
_Exported on 8/17/2025 at 15:22:26 GMT+2 from Cursor (1.4.5)_

---

**User**

Given everything we've learned in @tests/ and @analyse_dataset.py and more in this repo. How can we merge two datasets? Apparently, phospho did it in their platform, with merge and repair etc. Study how they did it and then let's think how we can do it.

@https://blog.phospho.ai/repair-merge-and-split-lerobot-datasets/ 
@https://github.com/phospho-app/phosphobot/pull/94/files 

---

**Cursor**

I'll scan the codebase to see existing dataset structures and any merge-related utilities, and search tests for references to merging or repair to guide a design. Then I'll review the local `LeRobotDataset` implementation and analysis utilities to understand the expected metadata format.

I scanned the repo for any existing merge/repair utilities and reviewed how `LeRobotDataset` stores metadata and episodes. I didn’t find an existing merge tool here; we’ll need to design one.

### What Phospho did (essentials)
- Added a backend endpoint to merge datasets that accepts:
  - `first_dataset`, `second_dataset`, `new_dataset_name`
  - `image_key_mappings` to reconcile camera key names between datasets
  - A dataset info response used to inspect compatibility
- Fixed task indexes during merge (deduplicate tasks across both datasets and remap `task_index` in episodes).
- Exposed an info endpoint to return robot type, DOF, episode count, image keys, and preview frames for UI checks.

Refs: [Blog: Repair, merge and split LeRobot datasets](https://blog.phospho.ai/repair-merge-and-split-lerobot-datasets/), [PR #94 (files)](https://github.com/phospho-app/phosphobot/pull/94/files)

### How to merge two LeRobot datasets safely
- Compatibility checks (fail fast if violated):
  - fps must match.
  - action dimensionality and dtype must match.
  - storage types reconciled: either both video or convert images→video for the output.
  - camera keys harmonized via a mapping; image shapes per camera should match after mapping.
  - same codebase version (v2.1) preferred; otherwise upgrade older dataset first.
  - robot type/DOF should match; if not, clearly mark the merged dataset or avoid merging.

- Schema decision:
  - Use the first dataset’s `features` as the canonical schema.
  - Apply `image_key_mappings` to the second dataset to rename its camera keys into the first’s naming.
  - Pick a conservative policy: keep only keys present in both after mapping (“intersection”); optionally allow “union” but that requires careful handling at training time.

- Task deduplication and index repair:
  - Load both `meta/tasks.jsonl`; build a union of distinct task strings.
  - Create a new contiguous `task_index` mapping for the merged dataset.
  - For every episode parquet to be merged, rewrite the `task_index` column using:
    - first dataset’s old_task_idx → new_task_idx
    - second dataset’s old_task_idx (post-map) → new_task_idx

- Episode and file renumbering:
  - Create the merged dataset root with a fresh `repo_id`/name.
  - Choose a `chunks_size` (use the first dataset’s).
  - Append all episodes from the first dataset as-is but ensure `episode_index` and on-disk filenames match; then append episodes from the second dataset with new `episode_index` values continuing after the first.
  - For each second-dataset episode:
    - Copy or rewrite parquet to the new `data/chunk-XXX/episode_YYYYYY.parquet` path.
    - Replace columns `episode_index` and `index` appropriately.
    - Remap `task_index` using the merged mapping.
  - Video files: copy `videos/chunk-XXX/<camera>/episode_YYYYYY.mp4` and rename to new episode indices. If the second dataset is images-only, encode videos from embedded images for the merged output, or decide to keep “images” for both (less preferred).

- Metadata and stats:
  - Build `meta/info.json` from the first dataset and update:
    - `total_episodes`, `total_frames`, `total_chunks`
    - `splits` (e.g., `"train": f"0:{total_episodes}"`)
    - `features` (schema decided above)
    - `video_path`/`data_path` templates (unchanged)
    - video `features[key]["info"]` (recompute from first episode in the merged set to be safe)
  - Merge `meta/episodes.jsonl`:
    - Write each episode dict with updated `episode_index`, `length`, and `tasks` (strings) for the merged indices.
  - Merge `episodes_stats`:
    - Copy per-episode stats files but rename indices; aggregate into `stats.json` if needed (v2.1 typically uses per-episode stats + aggregate).
  - Validate timestamps per-episode using the same sync check `check_timestamps_sync` invariant (merging should not break per-episode timestamp spacing at fps).

- Camera key reconciliation:
  - Use `image_key_mappings` such as:
    - `{"observation.images.wrist": "observation.images.wrist_2", "observation.images.context": "observation.images.scene_2"}`
  - For images-in-parquet, rename the image columns when rewriting the parquet.
  - For videos, just place/copy mp4s in the correct camera subfolders and ensure `meta.info.features` lists the canonical camera keys.

- Storage reconciliation:
  - Prefer output as video-backed datasets for performance.
  - If one source is images:
    - Read embedded images from parquet and encode to mp4 per episode (you can reuse the same fps).
    - Remove image columns from the rewritten parquet if moving to pure-video storage (video frames are not stored in parquet).
  - Otherwise, if you must keep images, keep both as images and ensure the merged `features` dtype for cameras is “image” consistently.

### Minimal algorithmic outline
- Load both datasets’ `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl` (+ per-episode stats).
- Validate compatibility and build:
  - `task_str -> new_task_idx`
  - `first_old_idx -> new_task_idx` and `second_old_idx -> new_task_idx`
- Create merged root and initialize `meta/info.json` (from first, then update totals later).
- Copy/transform episodes:
  - For each episode in dataset A:
    - If keeping video: copy mp4s; parquet may be reused; ensure filename/index consistency.
    - If rewriting: rewrite parquet to ensure consistency of `episode_index`, `index`, and `task_index`.
  - For each episode in dataset B:
    - Rewrite parquet with remapped `task_index`, new `episode_index`, fixed `index`, and any image column renames per camera mapping.
    - If images→video: encode mp4s, drop image columns; else copy images or keep embedded.
- Recompute/aggregate stats, write merged `meta/*`.
- Sanity checks: counts, ability to load with `LeRobotDataset`, iterate first few items, and decode videos.

### Practical options for you
- Quick alternative (no merge): use `MultiLeRobotDataset` during training across two datasets without modifying files, at the cost of only using the intersection of features at load time. This is already available in your code under `lerobot_some_original_code/lerobot_dataset.py` and is the fastest path for experiments.
- Batch merge tool: we can add a small CLI script that follows the steps above. Suggested interface:
  - `--first /path/to/dsA --second /path/to/dsB --out /path/to/merged`
  - `--image-key-mapping '{"observation.images.wrist":"observation.images.wrist2","observation.images.context":"observation.images.context2"}'`
  - `--keep intersection|union` (default: intersection)
  - `--storage video|image` (default: video)
  - `--force` to overwrite output folder

If you want, I can implement a minimal `tools/merge_lerobot_datasets.py` that:
- Reads/writes parquet with pyarrow, remaps `task_index`, `episode_index`, and camera columns.
- Copies/encodes videos.
- Writes merged `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, and per-episode stats.

Summary:
- Phospho merges by reconciling camera keys, fixing task indices, and writing a new dataset with consistent schema and metadata; they expose this via an API. We can replicate this locally with a CLI that validates fps/action dims, builds a unified task map, renumbers episodes, rewrites parquets (including optional images→video conversion), copies/encodes mp4s, and updates meta and stats. Refs: blog and PR linked above.

---

**User**

i didn't know about @MultiLeRobotDataset ... Hmmm Is this easy to train on as normal? What is the co-training mixture? Everything is the same proportion? Is this what you recommend for training on 5+ datasets?

---

**Cursor**

### Short answer
- Yes, you can train on `MultiLeRobotDataset` like a normal dataset. It returns the same item dict as `LeRobotDataset` plus a `dataset_index` tensor.
- The default sampling mixture is proportional to each dataset’s number of frames. If you want balanced co-training, use a custom/balanced sampler.
- For 5+ datasets, I recommend starting with `MultiLeRobotDataset` + a balanced sampler. Consider full merge later if you need one on-disk dataset or unified metadata.

### Why this works
`MultiLeRobotDataset` concatenates multiple `LeRobotDataset`s and automatically keeps only the intersection of feature keys across them. It also exposes which dataset an item came from via `dataset_index`.

```1046:1065:lerobot_some_original_code/lerobot_dataset.py
        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)
```

```1156:1176:lerobot_some_original_code/lerobot_dataset.py
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item
```

Note: It aggregates normalization stats across sub-datasets, which may be suboptimal if robots differ (the code even calls this out).

```1068:1072:lerobot_some_original_code/lerobot_dataset.py
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
```

### Co-training mixture (default vs balanced)
- Default mixture: With `shuffle=True`, sampling is uniform over the concatenated index space, so probability is proportional to each dataset’s `num_frames`.
- Balanced options:
  - Use `WeightedRandomSampler` with per-sample weights inversely proportional to the source dataset size.
  - Implement a simple round-robin or interleaving `BatchSampler` that draws an equal number of batches from each dataset.
  - Run multiple DataLoaders (one per dataset) and alternate batches in the training loop.
  - Downsample large datasets by passing `episodes` per repo to control exposure.

### When to prefer merging
- If you need a single on-disk dataset with unified tasks, repaired `task_index`, harmonized camera keys, and consolidated metadata for sharing/publishing, do an offline merge (as Phospho does).
- If you just want to co-train quickly on many datasets locally, `MultiLeRobotDataset` is simpler and avoids heavy I/O.

### Practical recommendations for 5+ datasets
- Start with `MultiLeRobotDataset` for speed of iteration.
- Add a balanced sampler if you want equal per-dataset exposure.
- Ensure compatibility up front: same fps, same action dimensionality/dtype, and compatible camera shapes. Remember only common features are kept.
- If robot ranges differ, consider per-dataset normalization or domain conditioning; the current aggregate normalization may not be ideal.

References:
- Phospho’s merge/repair/split overview: [blog post](https://blog.phospho.ai/repair-merge-and-split-lerobot-datasets/)
- Dataset merge API and task index fixes: [PR #94 files](https://github.com/phospho-app/phosphobot/pull/94/files)

- Implemented intersection-of-features and dataset-index exposure in `MultiLeRobotDataset` (see cited code in `lerobot_some_original_code/lerobot_dataset.py`).

- For quick co-training, using `MultiLeRobotDataset` is fine; mixture is size-proportional by default; use a weighted or round-robin sampler for balancing.

```python
        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)
```

```python
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item
```

```python
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
```

---

**User**

ok, i will record 2-7 more datasets, and then how do i make multilerobot dataset work with a train.py command like this: 

run_count=1
task_name="pick_place_one_white_sock_black_out_blinds"
python src/lerobot/scripts/train.py \                                             
  --dataset.repo_id=bearlover365/${task_name} \
  --policy.type=act \
  --policy.repo_id=bearlover365/${task_name}_${run_count}_act_model \
  --output_dir=outputs/train/${task_name}_testesttest_numpy_less_than_2 \
  --job_name=${task_name} \
  --policy.device=cuda \
  --wandb.enable=true \
  --dataset.video_backend=pyav

---

**Cursor**

I'll check how your current training script constructs the dataset, so I can show exactly what to change to support multiple repo_ids and instantiate `MultiLeRobotDataset`. Then I'll provide the command-line pattern to pass multiple datasets and an optional balanced sampler.

```python
--dataset._target_=lerobot.datasets.lerobot_dataset.MultiLeRobotDataset --dataset.repo_ids='[...]'
```

### Short answer
- Yes: you can train “as normal” with `MultiLeRobotDataset`. It yields the same item dict as a single dataset plus `dataset_index`.
- Default mixture is proportional to each dataset’s number of frames. Use a balanced sampler if you want equal exposure.
- For 5+ datasets, start with `MultiLeRobotDataset` + balanced sampling; only do an on-disk merge later if you need a single unified dataset.

### How to use it with your train command

You have two practical paths:

- If you’re using the upstream Hydra training script (your example: `src/lerobot/scripts/train.py`):
  - Some Hydra configs allow swapping the dataset target and passing a list of repo IDs. If supported in your LeRobot version, you’d pass something like:
    - set the dataset target to the multi-dataset class
    - pass `dataset.repo_ids` as a list
  - Example pattern (adjust to your config keys; Hydra list values must be quoted):
    - `--dataset._target_=lerobot.datasets.lerobot_dataset.MultiLeRobotDataset`
    - `--dataset.repo_ids='["bearlover365/ds1","bearlover365/ds2","bearlover365/ds3"]'`
    - Keep the rest (policy, wandb, video backend) the same as your single-dataset run.
  - If your Hydra config does not expose a multi-dataset target or `repo_ids`, use the local script path below.

- If you’re using the local `train.py` in this repo:
  - Add a CLI flag that takes multiple datasets (e.g., `--datasets 'bearlover365/ds1,bearlover365/ds2,...'`).
  - In `load_dataset`, if multiple repo IDs are provided, instantiate `MultiLeRobotDataset(repo_ids=[...], root=..., delta_timestamps=..., image_transforms=..., video_backend=...)` instead of `LeRobotDataset`.
  - Dataloader stays the same and will iterate over the concatenation of all datasets.

Command example with the local script:
- `python train.py --datasets 'bearlover365/ds1,bearlover365/ds2,bearlover365/ds3' --video-backend pyav --batch-size 64 --steps 3000 --output-dir outputs/train/my_multi_run`

### Co-training mixture and balancing
- Default sampling is size-proportional (uniform over the concatenated index space).
- For balanced exposure across N datasets:
  - Use a `WeightedRandomSampler` with per-sample weights ∝ 1/size(dataset_of_sample).
  - Or implement a simple round-robin batch sampler that draws equally from each dataset.
  - Optionally, predefine per-dataset probabilities (e.g., 50/30/20) and set weights accordingly.

### Tips for 5+ datasets
- Ensure compatibility before co-training: same fps, same action dimensionality/dtype, compatible camera shapes. `MultiLeRobotDataset` automatically keeps only the intersection of feature keys.
- Normalization: the current multi-dataset implementation aggregates stats across sub-datasets; if robots differ significantly, consider per-dataset normalization or adding domain-conditioning features.
- If later you need a single artifact, do an offline merge (as in Phospho’s approach) to harmonize camera keys and repair task indexes, then publish one merged dataset.
  - See: Phospho’s overview on merge/repair/split and their PR adding merge endpoints and task-index fixes: [blog](https://blog.phospho.ai/repair-merge-and-split-lerobot-datasets/), [PR #94 files](https://github.com/phospho-app/phosphobot/pull/94/files)

- For quick iteration now, `MultiLeRobotDataset` + balanced sampler is the most direct path.

