import json, shutil
from pathlib import Path
from datasets import Dataset

from lerobot.merge_lerobot_datasets import merge_datasets, verify_merged
from lerobot.datasets.utils import INFO_PATH


def make_local_ds(root: Path, ep_lengths: list[int]):
    if root.exists():
        shutil.rmtree(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    chunks_size = 100
    total_frames = sum(ep_lengths)
    info = {
        "chunks_size": chunks_size,
        "total_episodes": len(ep_lengths),
        "total_frames": total_frames,
    }
    (root / INFO_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / INFO_PATH).write_text(json.dumps(info))

    running_index = 0
    for ep_idx, n in enumerate(ep_lengths):
        data = {
            "index": list(range(running_index, running_index + n)),
            "episode_index": [ep_idx] * n,
        }
        running_index += n
        ds = Dataset.from_dict(data)
        out = root / f"data/chunk-{ep_idx // chunks_size:03d}/episode_{ep_idx:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(out)


def main():
    base = Path.home() / ".cache/huggingface/lerobot/_smoke_merge"
    a = base / "a"
    b = base / "b"
    out = base / "merged"
    make_local_ds(a, [3, 4])
    make_local_ds(b, [2])
    if out.exists():
        shutil.rmtree(out)
    merge_datasets("first", "second", "merged", out, dry_run=True, skip_videos=True, second_image_key_mapping=None, first_dir=a, second_dir=b)
    verify_merged(out)
    print("smoke merge verify: OK")


if __name__ == "__main__":
    main()


