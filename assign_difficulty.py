# Author: Harsh Kohli
# Date Created: 03-09-2024

import os
import json
import torch
import gc
from tqdm import tqdm
from constants import BASE_MODEL, DATASETS_DIR, MODELS_DIR, DIFFICULTY_DIR
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

dataset_dir = DATASETS_DIR
dataset_files = os.listdir(dataset_dir)
dataset_files = [f for f in dataset_files if not f.startswith('.')]

model_dir = MODELS_DIR
models = os.listdir(model_dir)
models = [f for f in models if not f.startswith('.')]

task_num = 0
for model in models:
    task = model.split("-")[1]
    matching_dataset = [s for s in dataset_files if s.startswith(task)]
    count = len(matching_dataset)
    if count != 1 or not matching_dataset[0].endswith(".json"):
        continue
    print("Evaluating task " + str(task_num) + ": " + task)
    task_num = task_num + 1
    datafile = open(os.path.join(DATASETS_DIR, matching_dataset[0]), 'r', encoding='utf8')
    task_data = []

    config = PeftConfig.from_pretrained("lorahub/" + model)
    lora_model = PeftModel.from_pretrained(base_model, "lorahub/" + model).to(device)
    for line in tqdm(datafile.readlines()):
        sample = json.loads(line)
        input_ids = tokenizer(sample['inputs'], return_tensors='pt').input_ids.to(device)
        target_ids = tokenizer(sample['targets'], return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            outputs = lora_model(input_ids=input_ids, decoder_input_ids=target_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        log_likelihood = target_log_probs.sum()

        avg_log_likelihood = log_likelihood / shift_labels.size(1)

        confidence = torch.exp(log_likelihood)

        sample["log_likelihood"] = float(log_likelihood.cpu().numpy())
        sample["avg_log_likelihood"] = float(avg_log_likelihood.cpu().numpy())
        sample["confidence"] = float(confidence.cpu().numpy())
        task_data.append(sample)


    datafile.close()
    with open(os.path.join(DIFFICULTY_DIR, matching_dataset[0]), 'w', encoding='utf8') as outfile:
        json.dump(task_data, outfile, indent=4)


    del lora_model, config
    torch.cuda.empty_cache()
    gc.collect()

print('Done')
