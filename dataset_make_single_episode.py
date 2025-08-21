import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    INFO_PATH,
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    TASKS_PATH,
    write_json,
    write_jsonlines,
    write_episode_stats,
    get_hf_features_from_features,
    load_stats,
    write_stats,
)


def load_source_dataset(source_repo: str, download_videos: bool = True) -> LeRobotDataset:
    ds = LeRobotDataset(source_repo, download_videos=download_videos)
    return ds


def build_hf_features(meta_features):
    return get_hf_features_from_features(meta_features)


def extract_episode(ds: LeRobotDataset, episode_index: int):
    indices = np.nonzero(np.stack(ds.hf_dataset["episode_index"]) == episode_index)[0]
    return ds.hf_dataset.select(indices)


def assemble_split(ds: LeRobotDataset, split_eps: list[int], out_dir: Path, out_repo: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((ds.meta.root / INFO_PATH).read_text())
    info["total_episodes"] = len(split_eps)
    info["total_frames"] = int(sum(ds.meta.episodes[e]["length"] for e in split_eps))
    info["splits"] = {"train": f"0:{len(split_eps)}"}
    write_json(info, out_dir / INFO_PATH)

    src_tasks = ds.meta.root / TASKS_PATH
    if src_tasks.is_file():
        dst_tasks = out_dir / TASKS_PATH
        dst_tasks.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_tasks, dst_tasks)

    new_episodes = []
    for new_idx, old_idx in enumerate(split_eps):
        epd = dict(ds.meta.episodes[old_idx])
        epd["episode_index"] = new_idx
        new_episodes.append(epd)
    write_jsonlines(new_episodes, out_dir / EPISODES_PATH)

    stats = load_stats(ds.meta.root)
    if stats is None:
        stats = ds.meta.stats
    write_stats(stats, out_dir)

    for new_idx, old_idx in enumerate(split_eps):
        write_episode_stats(new_idx, ds.meta.episodes_stats[old_idx], out_dir)

    hf_features = build_hf_features(ds.meta.features)
    running_index = 0
    for new_idx, old_idx in enumerate(split_eps):
        ep_ds = extract_episode(ds, old_idx)
        d = {k: ep_ds[k] for k in ep_ds.features}
        n = len(d["index"])
        d["episode_index"] = [new_idx] * n
        d["index"] = list(range(running_index, running_index + n))
        running_index += n
        ep_out = Dataset.from_dict(d, features=hf_features, split="train")
        chunk = new_idx // info["chunks_size"]
        ep_path = out_dir / f"data/chunk-{chunk:03d}/episode_{new_idx:06d}.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        ep_out.to_parquet(ep_path)

    if info.get("video_path"):
        for new_idx, old_idx in enumerate(split_eps):
            for vid_key in ds.meta.video_keys:
                src = ds.meta.root / ds.meta.get_video_file_path(old_idx, vid_key)
                chunk = new_idx // info["chunks_size"]
                rel = info["video_path"].format(
                    episode_chunk=chunk, video_key=vid_key, episode_index=new_idx
                )
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_file():
                    shutil.copyfile(src, dst)

    api = HfApi()
    api.create_repo(repo_id=out_repo, private=True, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=out_repo,
        folder_path=str(out_dir),
        repo_type="dataset",
        allow_patterns=["**"],
        ignore_patterns=["*.tmp", "*.log"],
    )


def main():
    parser = argparse.ArgumentParser(description="Create single-episode dataset from a source repo")
    parser.add_argument(
        "--source-repo",
        type=str,
        required=True,
        help="Source HF dataset repo id (e.g. bearlover365/pick_place_up_to_four_white_socks_black_out_blinds)",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to keep in the single-episode dataset (0-based)",
    )
    parser.add_argument(
        "--train-repo",
        type=str,
        required=True,
        help="Destination HF dataset repo id for TRAIN split (single episode)",
    )
    parser.add_argument(
        "--val-repo",
        type=str,
        required=True,
        help="Destination HF dataset repo id for VAL split (single episode)",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download and copy videos along with parquet data",
    )
    args = parser.parse_args()

    ds = load_source_dataset(args.source_repo, download_videos=args.download_videos)
    selected_ep = int(args.episode_index)

    all_eps = sorted(ds.meta.episodes.keys())
    if selected_ep not in all_eps:
        raise ValueError(f"Episode {selected_ep} not in source dataset (available: {all_eps[:8]} ...)")

    home = Path(ds.root).parent
    train_dir = home / args.train_repo
    val_dir = home / args.val_repo
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)

    assemble_split(ds, [selected_ep], train_dir, args.train_repo)
    assemble_split(ds, [selected_ep], val_dir, args.val_repo)

    print("Created and uploaded single-episode datasets:")
    print(f"  train_repo={args.train_repo}")
    print(f"  val_repo  ={args.val_repo}")


if __name__ == "__main__":
    main()



