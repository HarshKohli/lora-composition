# Author: Harsh Kohli
# Date Created: 07-09-2024

import json
import os
import gc
import torch
from peft import PeftModel, PeftConfig
from utils import get_model_ability
from constants import BASE_MODEL, MODELS_DIR, DIFFICULTY_DIR, MAPPINGS_FILE, ABILITIES_FILE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

dataset_files = os.listdir(DIFFICULTY_DIR)
dataset_files = [f for f in dataset_files if not f.startswith('.')]

model_dir = MODELS_DIR
models = os.listdir(model_dir)
models = [f for f in models if not f.startswith('.')]

all_models, all_matching_datasets = [], []
for model in models:
    task = model.split("-")[1]
    matching_dataset = [s for s in dataset_files if s.startswith(task)]
    count = len(matching_dataset)
    if count != 1 or not matching_dataset[0].endswith(".json"):
        continue
    all_models.append(model)
    all_matching_datasets.append(matching_dataset[0])


all_mappings, idx = {}, 0
all_mappings['adapter_2_idx'] = {}
all_mappings['task_2_idx'] = {}
all_mappings['adapter_2_task'] = {}
all_mappings['task_2_adapter'] = {}
all_mappings['idx_2_adapter'] = {}
all_mappings['idx_2_task'] = {}


for adapter, task in zip(all_models, all_matching_datasets):
    all_mappings['adapter_2_idx'][adapter] = idx
    all_mappings['task_2_idx'][task] = idx
    all_mappings['adapter_2_task'][adapter] = task
    all_mappings['task_2_adapter'][task] = adapter
    all_mappings['idx_2_adapter'][idx] = adapter
    all_mappings['idx_2_task'][idx] = task
    idx = idx + 1

with open(MAPPINGS_FILE, 'w') as json_file:
    json.dump(all_mappings, json_file, indent=4)

abilities = {}
for adapter in all_models:
    config = PeftConfig.from_pretrained("lorahub/" + adapter)
    lora_model = PeftModel.from_pretrained(base_model, "lorahub/" + adapter).to(device)

    abilities[adapter] = {}
    for dataset in all_matching_datasets:
        difficulty_file = open(os.path.join(DIFFICULTY_DIR, dataset), 'r', encoding='utf8')
        data = json.load(difficulty_file)
        difficulty_file.close()
        ability, se = get_model_ability(lora_model, data, tokenizer, device)
        print("Adapter " + adapter + " ability on task " + dataset + " is " + str(ability))
        abilities[adapter][dataset] = {"ability": ability, "se": se}
        with open(ABILITIES_FILE, 'w') as json_file:
            json.dump(abilities, json_file, indent=4)

    del lora_model, config
    torch.cuda.empty_cache()
    gc.collect()
