from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dataset.json"
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
info = HfApi().repo_info(repo_id=config["repo_id"], repo_type="dataset")
config["revision"] = info.sha
CONFIG_PATH.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(info.sha)
