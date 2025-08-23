import os, json, shutil
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import ast

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
  INFO_PATH, EPISODES_PATH, EPISODES_STATS_PATH, STATS_PATH, TASKS_PATH,
  write_json, write_jsonlines, write_episode_stats, load_stats, write_stats,
  get_hf_features_from_features
)
from lerobot.datasets.compute_stats import aggregate_stats


def read_tasks(meta_root: Path) -> Tuple[List[str], Dict[int, str]]:
    tasks_fp = meta_root / TASKS_PATH
    tasks: List[str] = []
    if tasks_fp.is_file():
        for line in tasks_fp.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            tasks.append(obj["task"]) if isinstance(obj, dict) and "task" in obj else tasks.append(str(obj))
    idx_to_task = {i: t for i, t in enumerate(tasks)}
    return tasks, idx_to_task


def build_merged_task_map(first_root: Path, second_root: Path) -> Tuple[Dict[int, int], Dict[int, int], List[str]]:
    first_tasks, first_idx_to_task = read_tasks(first_root)
    second_tasks, second_idx_to_task = read_tasks(second_root)

    merged: List[str] = []
    task_to_new: Dict[str, int] = {}

    def ensure_task(task_str: str) -> int:
        if task_str not in task_to_new:
            task_to_new[task_str] = len(merged)
            merged.append(task_str)
        return task_to_new[task_str]

    first_map: Dict[int, int] = {}
    for old_idx, task_str in first_idx_to_task.items():
        first_map[old_idx] = ensure_task(task_str)

    second_map: Dict[int, int] = {}
    for old_idx, task_str in second_idx_to_task.items():
        second_map[old_idx] = ensure_task(task_str)

    return first_map, second_map, merged


def verify_merged(out_dir: Path):
    info = json.loads((out_dir / INFO_PATH).read_text())
    total_frames_from_parquet = 0
    for ep_idx in range(info["total_episodes"]):
        chunk = ep_idx // info["chunks_size"]
        ep_path = out_dir / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        ep_ds = load_dataset("parquet", data_files=str(ep_path), split="train")
        n = len(ep_ds)
        total_frames_from_parquet += n
        if "episode_index" in ep_ds.column_names:
            if set(ep_ds["episode_index"]) != {ep_idx}:
                raise RuntimeError(f"episode_index mismatch in {ep_path}")
        if "index" in ep_ds.column_names and n > 0:
            idxs = ep_ds["index"]
            if any(b - a != 1 for a, b in zip(idxs, idxs[1:])):
                raise RuntimeError(f"index not continuous in {ep_path}")
    if total_frames_from_parquet != info["total_frames"]:
        raise RuntimeError("total_frames mismatch after merge")


def ensure_second_video_key_mapping(first_keys: List[str], second_keys: List[str], mapping: Dict[str, str] | None) -> Dict[str, str]:
    if mapping is None:
        if set(first_keys) != set(second_keys):
            raise RuntimeError("Video/camera keys differ and no mapping provided for the second dataset")
        return {k: k for k in second_keys}
    # validate mapping covers second keys and maps into first keys
    missing = [k for k in second_keys if k not in mapping]
    if missing:
        raise RuntimeError(f"Mapping missing for second dataset keys: {missing}")
    invalid_targets = [v for v in mapping.values() if v not in first_keys]
    if invalid_targets:
        raise RuntimeError(f"Mapping targets not in first dataset keys: {invalid_targets}")
    return mapping


class _LocalMeta:
    def __init__(self, root: Path, episodes: Dict[int, dict], episodes_stats: Dict[int, dict], video_keys: List[str]):
        self.root = root
        self.episodes = episodes
        self.episodes_stats = episodes_stats
        self.video_keys = video_keys


class _LocalDS:
    def __init__(self, root: Path):
        self.meta = self._load_meta(root)

    def _load_meta(self, root: Path) -> _LocalMeta:
        info = json.loads((root / INFO_PATH).read_text())
        # episodes.jsonl is optional; build from data directory if missing
        episodes: Dict[int, dict] = {}
        episodes_fp = root / EPISODES_PATH
        if episodes_fp.is_file():
            for line in episodes_fp.read_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                ep_idx = int(obj.get("episode_index", len(episodes)))
                episodes[ep_idx] = obj
        else:
            total = info.get("total_episodes", 0)
            for ep_idx in range(total):
                episodes[ep_idx] = {"episode_index": ep_idx}
        # episodes_stats not needed for local smoke
        episodes_stats: Dict[int, dict] = {}
        video_keys: List[str] = []
        return _LocalMeta(root, episodes, episodes_stats, video_keys)


def merge_datasets(first_repo: str, second_repo: str, out_repo: str, out_dir: Path, dry_run: bool, skip_videos: bool, second_image_key_mapping: Dict[str, str] | None, first_dir: Path | None = None, second_dir: Path | None = None):
    if first_dir and second_dir:
        first = _LocalDS(first_dir)
        second = _LocalDS(second_dir)
        local_mode = True
    else:
        first = LeRobotDataset(first_repo, download_videos=not skip_videos)
        second = LeRobotDataset(second_repo, download_videos=not skip_videos)
        local_mode = False

    out_dir.mkdir(parents=True, exist_ok=True)

    # Base info from first
    info_first = json.loads((first.meta.root / INFO_PATH).read_text())
    info_second = json.loads((second.meta.root / INFO_PATH).read_text())
    info = dict(info_first)

    # Features: use first as canonical if available (LeRobotDataset). In local mode, let HF infer features.
    hf_features = None
    if not local_mode:
        hf_features = get_hf_features_from_features(first.meta.features)

    # Compatibility checks (minimal)
    if info_first.get("fps") != info_second.get("fps"):
        raise RuntimeError(f"fps mismatch: {info_first.get('fps')} vs {info_second.get('fps')}")
    # Storage expectations
    if bool(info_first.get("video_path")) != bool(info_second.get("video_path")):
        raise RuntimeError("One dataset is video-backed and the other is not. Conversion not implemented in this tool.")

    # Camera keys reconciliation for videos
    video_key_map_second_to_first: Dict[str, str] | None = None
    if info.get("video_path") and not local_mode:
        video_key_map_second_to_first = ensure_second_video_key_mapping(first.meta.video_keys, second.meta.video_keys, second_image_key_mapping)

    # Build task remap
    first_map, second_map, merged_tasks = build_merged_task_map(first.meta.root, second.meta.root)

    # Write tasks.jsonl with required schema
    if merged_tasks:
        write_jsonlines([
            {"task_index": idx, "task": t} for idx, t in enumerate(merged_tasks)
        ], out_dir / TASKS_PATH)

    # Prepare episodes.jsonl and stats
    new_episodes = []
    running_index = 0
    total_frames = 0

    # Helper to copy/transform episodes from a dataset
    def append_from(ds: LeRobotDataset, start_ep_idx: int, task_map: Dict[int, int], is_second: bool = False):
        nonlocal running_index, total_frames, new_episodes
        base_len = len(new_episodes)
        # iterate in natural order of ds.meta.episodes keys
        for local_ep_idx in sorted(ds.meta.episodes.keys()):
            new_ep_idx = start_ep_idx + local_ep_idx
            epd = dict(ds.meta.episodes[local_ep_idx])
            epd["episode_index"] = new_ep_idx
            # Fix task_index in episode metadata if present
            if "task_index" in epd and isinstance(epd["task_index"], int) and task_map:
                epd["task_index"] = task_map.get(epd["task_index"], epd["task_index"])
            new_episodes.append(epd)

            # Load parquet, rewrite indices and task_index column if present
            chunk_old = local_ep_idx // info["chunks_size"]
            old_parquet = ds.meta.root / f"data/chunk-{chunk_old:03d}/episode_{local_ep_idx:06d}.parquet"
            ep_ds = load_dataset("parquet", data_files=str(old_parquet), split="train")
            data_dict = {k: ep_ds[k] for k in ep_ds.features}
            n = len(ep_ds)
            total_frames += n
            data_dict["episode_index"] = [new_ep_idx] * n
            data_dict["index"] = list(range(running_index, running_index + n))
            running_index += n
            if task_map and "task_index" in data_dict:
                data_dict["task_index"] = [task_map.get(i, i) for i in data_dict["task_index"]]

            out_chunk = new_ep_idx // info["chunks_size"]
            out_parquet = out_dir / f"data/chunk-{out_chunk:03d}/episode_{new_ep_idx:06d}.parquet"
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            if hf_features is not None:
                Dataset.from_dict(data_dict, features=hf_features, split="train").to_parquet(out_parquet)
            else:
                Dataset.from_dict(data_dict, split="train").to_parquet(out_parquet)

            # videos copy if present
            if not skip_videos and info.get("video_path") and not local_mode:
                for vid_key in ds.meta.video_keys:
                    src = ds.meta.root / ds.meta.get_video_file_path(local_ep_idx, vid_key)
                    target_key = vid_key
                    if is_second and video_key_map_second_to_first is not None:
                        target_key = video_key_map_second_to_first[vid_key]
                    rel = info["video_path"].format(episode_chunk=out_chunk, video_key=target_key, episode_index=new_ep_idx)
                    dst = out_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_file():
                        shutil.copyfile(src, dst)

    # First dataset episodes (indices start at 0)
    append_from(first, 0, first_map, is_second=False)
    # Second dataset episodes (continue numbering)
    start_for_second = len(new_episodes)
    append_from(second, start_for_second, second_map, is_second=True)

    # Write episodes.jsonl
    write_jsonlines(new_episodes, out_dir / EPISODES_PATH)

    # Stats: aggregate from both datasets if available for best normalization
    stats_first = load_stats(first.meta.root) or getattr(first.meta, "stats", None)
    stats_second = load_stats(second.meta.root) or getattr(second.meta, "stats", None)
    if stats_first is not None and stats_second is not None:
        stats = aggregate_stats([stats_first, stats_second])
    else:
        stats = stats_first or stats_second
    if stats is not None:
        write_stats(stats, out_dir)

    # Episodes stats: copy and rename where available
    for new_idx, epd in enumerate(new_episodes):
        # Determine origin ds/old idx by inverse mapping from order
        # For simplicity, try to copy from first if exists else second
        if new_idx in first.meta.episodes:
            src_stats = first.meta.episodes_stats.get(new_idx)
        else:
            local_idx = new_idx - len(first.meta.episodes)
            src_stats = second.meta.episodes_stats.get(local_idx)
        if src_stats is not None:
            write_episode_stats(new_idx, src_stats, out_dir)

    # Update and write info.json
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = int(total_frames)
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    if "chunks_size" in info:
        chunks_size = info["chunks_size"]
        info["total_chunks"] = (info["total_episodes"] + chunks_size - 1) // chunks_size
    write_json(info, out_dir / INFO_PATH)

    # Verify basic invariants
    verify_merged(out_dir)

    # Push
    if not dry_run:
        api = HfApi()
        api.create_repo(repo_id=out_repo, private=False, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=out_repo, folder_path=str(out_dir), repo_type="dataset",
                          allow_patterns=["**"], ignore_patterns=["*.tmp","*.log"])


def main():
    parser = argparse.ArgumentParser(description="Merge two LeRobot datasets into one")
    parser.add_argument("--first_repo", required=True, type=str)
    parser.add_argument("--second_repo", required=True, type=str)
    parser.add_argument("--out_repo", required=True, type=str)
    parser.add_argument("--out_dir", required=False, type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--second-image-key-mapping", type=str, default=None, help="JSON mapping from second dataset camera keys to first dataset keys")
    parser.add_argument("--first_dir", type=str, default=None, help="Local dataset root for first dataset (offline mode)")
    parser.add_argument("--second_dir", type=str, default=None, help="Local dataset root for second dataset (offline mode)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path.home() / ".cache/huggingface/lerobot" / args.out_repo
    if out_dir.exists():
        shutil.rmtree(out_dir)

    mapping = None
    if args.second_image_key_mapping:
        mapping = json.loads(args.second_image_key_mapping)

    first_dir = Path(args.first_dir) if args.first_dir else None
    second_dir = Path(args.second_dir) if args.second_dir else None
    merge_datasets(args.first_repo, args.second_repo, args.out_repo, out_dir, args.dry_run, args.skip_videos, mapping, first_dir, second_dir)
    print("Merge completed.")


if __name__ == "__main__":
    main()


