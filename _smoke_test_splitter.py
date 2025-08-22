import json
from pathlib import Path
from datasets import Dataset

from lerobot.dataset_splitter_by_episode import verify_output
from lerobot.datasets.utils import INFO_PATH


def make_fake_split(out_dir: Path, num_frames: int = 5):
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "chunks_size": 100,
        "total_episodes": 1,
        "total_frames": num_frames,
    }
    (out_dir / INFO_PATH).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / INFO_PATH).write_text(json.dumps(info))

    # Build one episode parquet
    data = {
        "index": list(range(num_frames)),
        "episode_index": [0] * num_frames,
    }
    ds = Dataset.from_dict(data)
    ep_path = out_dir / f"data/chunk-000/episode_{0:06d}.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(ep_path)


def main():
    tmp = Path.home() / ".cache/huggingface/lerobot/_smoke_split"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    make_fake_split(tmp, num_frames=7)
    verify_output(tmp)
    print("smoke splitter verify: OK")


if __name__ == "__main__":
    main()


