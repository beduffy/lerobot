import os, shutil, json, numpy as np
import argparse
from pathlib import Path
from huggingface_hub import HfApi
from datasets import load_dataset, Dataset, Features, Value, Array3D, Array2D, Sequence, Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import INFO_PATH, EPISODES_PATH, EPISODES_STATS_PATH, STATS_PATH, write_json, write_jsonlines, write_episode_stats, get_hf_features_from_features
from lerobot.datasets.utils import (
  INFO_PATH, EPISODES_PATH, EPISODES_STATS_PATH, STATS_PATH, TASKS_PATH,
  write_json, write_jsonlines, write_episode_stats,
  get_hf_features_from_features, load_stats, write_stats
)

SOURCE_REPO = "bearlover365/pick_place_one_white_sock_black_out_blinds"
HELDOUT = [50]  # 0-based index of the 51st episode
TRAIN_REPO = "bearlover365/pick_place_one_white_sock_black_out_blinds_all_except_one"
VAL_REPO = "bearlover365/pick_place_one_white_sock_black_out_blinds_51st_episode"

def load_src():
    ds = LeRobotDataset(SOURCE_REPO, download_videos=True)  # pulls meta + data
    return ds

def build_hf_features(meta_features):
    return get_hf_features_from_features(meta_features)

def extract_episode(ds, ep_idx):
    # slice rows belonging to episode ep_idx
    ep_info = ds.meta.episodes[ep_idx]
    # find contiguous rows with episode_index==ep_idx
    indices = np.nonzero(np.stack(ds.hf_dataset["episode_index"])==ep_idx)[0]
    return ds.hf_dataset.select(indices)

def verify_output(out_dir: Path):
    """Quick self-check that the split dataset is internally consistent."""
    info_path = out_dir / INFO_PATH
    episodes_path = out_dir / EPISODES_PATH
    if not info_path.is_file():
        raise RuntimeError(f"Missing info.json at {info_path}")
    if not episodes_path.is_file():
        raise RuntimeError(f"Missing episodes.jsonl at {episodes_path}")

    info = json.loads(info_path.read_text())
    total_frames_from_parquet = 0
    for ep_idx in range(info["total_episodes"]):
        chunk = ep_idx // info["chunks_size"]
        ep_path = out_dir / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        if not ep_path.is_file():
            raise RuntimeError(f"Missing parquet for episode {ep_idx}: {ep_path}")
        ep_ds = load_dataset("parquet", data_files=str(ep_path), split="train")
        n = len(ep_ds)
        total_frames_from_parquet += n
        if "episode_index" in ep_ds.column_names:
            unique_eps = set(ep_ds["episode_index"]) if n > 0 else {ep_idx}
            if unique_eps != {ep_idx}:
                raise RuntimeError(f"episode_index values mismatch in {ep_path}: {unique_eps}")
        if "index" in ep_ds.column_names and n > 0:
            idxs = ep_ds["index"]
            if any(b - a != 1 for a, b in zip(idxs, idxs[1:])):
                raise RuntimeError(f"index column not continuous in {ep_path}")

    if total_frames_from_parquet != info["total_frames"]:
        raise RuntimeError(
            f"total_frames mismatch: info={info['total_frames']} vs parquet={total_frames_from_parquet}"
        )


def assemble_split(ds, split_eps, out_dir: Path, out_repo, dry_run: bool = False, skip_videos: bool = False, verify: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write meta/info with updated counts
    info = json.loads((ds.meta.root / INFO_PATH).read_text())
    info["total_episodes"] = len(split_eps)
    info["total_frames"] = int(sum(ds.meta.episodes[e]["length"] for e in split_eps))
    info["splits"] = {"train": f"0:{len(split_eps)}"}
    write_json(info, out_dir / INFO_PATH)

    # tasks.jsonl: copy from source if present
    src_tasks = ds.meta.root / TASKS_PATH
    if src_tasks.is_file():
        dst_tasks = out_dir / TASKS_PATH
        dst_tasks.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_tasks, dst_tasks)

    # episodes.jsonl and episodes_stats.jsonl
    new_episodes = []
    for new_idx, old_idx in enumerate(split_eps):
        epd = dict(ds.meta.episodes[old_idx])
        epd["episode_index"] = new_idx  # reindex
        new_episodes.append(epd)
    write_jsonlines(new_episodes, out_dir / EPISODES_PATH)

    # stats.json: write (copy if present, otherwise compute from ds.meta.stats)
    stats = load_stats(ds.meta.root)
    if stats is None:
        stats = ds.meta.stats
    write_stats(stats, out_dir)

    # episodes_stats.jsonl with reindexed episode_index
    for new_idx, old_idx in enumerate(split_eps):
        write_episode_stats(new_idx, ds.meta.episodes_stats[old_idx], out_dir)

    # write parquet per episode with reindexed episode_index and continuous index
    hf_features = build_hf_features(ds.meta.features)
    running_index = 0
    for new_idx, old_idx in enumerate(split_eps):
        ep_ds = extract_episode(ds, old_idx)
        # materialize tensors -> numpy, adjust indices
        d = {k: ep_ds[k] for k in ep_ds.features}
        n = len(d["index"])
        d["episode_index"] = [new_idx]*n
        d["index"] = list(range(running_index, running_index+n))
        running_index += n
        # convert to HF dataset with new features schema
        ep_out = Dataset.from_dict(d, features=hf_features, split="train")
        # embed images already handled by loader when writing; here we only write parquet
        chunk = new_idx // info["chunks_size"]
        ep_path = out_dir / f"data/chunk-{chunk:03d}/episode_{new_idx:06d}.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        ep_out.to_parquet(ep_path)

    # videos: copy and rename per episode (optional; if you rely on videos)
    if info.get("video_path") and not skip_videos:
        for new_idx, old_idx in enumerate(split_eps):
            for vid_key in ds.meta.video_keys:
                src = ds.meta.root / ds.meta.get_video_file_path(old_idx, vid_key)
                chunk = new_idx // info["chunks_size"]
                rel = info["video_path"].format(episode_chunk=chunk, video_key=vid_key, episode_index=new_idx)
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_file():
                    shutil.copyfile(src, dst)

    if verify:
        verify_output(out_dir)

    # push to hub
    if not dry_run:
        api = HfApi()
        api.create_repo(repo_id=out_repo, private=True, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=out_repo, folder_path=str(out_dir), repo_type="dataset",
                          allow_patterns=["**"], ignore_patterns=["*.tmp","*.log"])

def main():
    global SOURCE_REPO, HELDOUT, TRAIN_REPO, VAL_REPO
    parser = argparse.ArgumentParser(description="Split a LeRobot dataset by episodes")
    parser.add_argument("--source_repo", type=str, default=SOURCE_REPO)
    parser.add_argument("--heldout", type=str, default=",".join(map(str, HELDOUT)), help="Comma-separated 0-based episode indices for validation")
    parser.add_argument("--train_repo", type=str, default=TRAIN_REPO)
    parser.add_argument("--val_repo", type=str, default=VAL_REPO)
    parser.add_argument("--dry-run", action="store_true", help="Do not push to HF; only write locally and verify")
    parser.add_argument("--skip-videos", action="store_true", help="Skip copying videos to speed up local checks")
    args = parser.parse_args()

    SOURCE_REPO = args.source_repo
    HELDOUT = [int(x) for x in args.heldout.split(",") if x != ""]
    TRAIN_REPO = args.train_repo
    VAL_REPO = args.val_repo

    ds = LeRobotDataset(SOURCE_REPO, download_videos=not args.skip_videos)
    all_eps = sorted(ds.meta.episodes.keys())
    val_eps = sorted(HELDOUT)
    train_eps = [e for e in all_eps if e not in val_eps]
    home = Path(ds.root).parent  # ~/.cache/huggingface/lerobot
    train_dir = home / TRAIN_REPO
    val_dir = home / VAL_REPO
    if train_dir.exists(): shutil.rmtree(train_dir)
    if val_dir.exists(): shutil.rmtree(val_dir)
    assemble_split(ds, train_eps, train_dir, TRAIN_REPO, dry_run=args.dry_run, skip_videos=args.skip_videos, verify=True)
    assemble_split(ds, val_eps, val_dir, VAL_REPO, dry_run=args.dry_run, skip_videos=args.skip_videos, verify=True)
    print("Done.")

if __name__ == "__main__":
    main()