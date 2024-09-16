# Author: Harsh Kohli
# Date Created: 02-09-2024

import os
from huggingface_hub import HfApi
from constants import LORA_MODULE_NAMES

api = HfApi()

for model in LORA_MODULE_NAMES:
    print("Downloading: " + model)
    api.snapshot_download(repo_id=model, repo_type="model", local_dir=os.path.join("resources", model))

api.snapshot_download(repo_id="lorahub/flanv2", repo_type="dataset", local_dir="resources/flan_datasets",
                      local_dir_use_symlinks=False)
