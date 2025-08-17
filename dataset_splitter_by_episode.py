import os, shutil, json, numpy as np
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

def assemble_split(ds, split_eps, out_dir: Path, out_repo):
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
    if info.get("video_path"):
        for new_idx, old_idx in enumerate(split_eps):
            for vid_key in ds.meta.video_keys:
                src = ds.meta.root / ds.meta.get_video_file_path(old_idx, vid_key)
                chunk = new_idx // info["chunks_size"]
                rel = info["video_path"].format(episode_chunk=chunk, video_key=vid_key, episode_index=new_idx)
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_file():
                    shutil.copyfile(src, dst)

    # push to hub
    api = HfApi()
    api.create_repo(repo_id=out_repo, private=True, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=out_repo, folder_path=str(out_dir), repo_type="dataset",
                      allow_patterns=["**"], ignore_patterns=["*.tmp","*.log"])

def main():
    ds = load_src()
    all_eps = sorted(ds.meta.episodes.keys())
    val_eps = sorted(HELDOUT)
    train_eps = [e for e in all_eps if e not in val_eps]
    home = Path(ds.root).parent  # ~/.cache/huggingface/lerobot
    train_dir = home / TRAIN_REPO
    val_dir = home / VAL_REPO
    if train_dir.exists(): shutil.rmtree(train_dir)
    if val_dir.exists(): shutil.rmtree(val_dir)
    assemble_split(ds, train_eps, train_dir, TRAIN_REPO)
    assemble_split(ds, val_eps, val_dir, VAL_REPO)
    print("Done.")
main()