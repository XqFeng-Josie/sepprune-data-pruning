"""Download the public 16.5 GB LRS2-2Mix archive into this workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/LRS2-2Mix")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="JusperLee/LRS2-2Mix",
        repo_type="dataset",
        filename="lrs2.tar.gz",
        revision="cab998836cfa9a28a4048c54809885811984370a",
        local_dir=output_dir,
    )
    print(downloaded, flush=True)


if __name__ == "__main__":
    main()
