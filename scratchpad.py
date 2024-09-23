# Author: Harsh Kohli
# Date Created: 15-09-2024

import os
import gc
import json
import torch
import evaluate
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from constants import ABILITIES_FILE, MAPPINGS_FILE, BASE_MODEL, DATASETS_DIR
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
metric = evaluate.load('squad')

f = open(ABILITIES_FILE, 'r', encoding='utf8')
abilities_info = json.load(f)
f.close()

f = open(MAPPINGS_FILE, 'r', encoding='utf8')
mappings_info = json.load(f)
f.close()

to_evaluate = [1, 14, 25]
all_models, all_tasks = [], []
for model, v in abilities_info.items():
    task = mappings_info['adapter_2_task'][model]
    all_models.append(model)
    all_tasks.append(task)

select_models = [all_models[x] for x in to_evaluate]
select_tasks = [all_tasks[x] for x in to_evaluate]

abilities, errors = [], []

for model in select_models:
    one_ability, one_error = [], []
    for task in select_tasks:
        info = abilities_info[model][task]
        one_ability.append(info['ability'])
        one_error.append(info['se'])
    abilities.append(one_ability)
    errors.append(one_error)

print('Abilities: ')
print()
for a in abilities:
    print([f"{x:.2f}" for x in a])

print()

print('Errors: ')
print()

for a in errors:
    print([f"{x:.2f}" for x in a])

abilities, errors = [], []

for model in all_models:
    one_ability, one_error = [], []
    for task in all_tasks:
        info = abilities_info[model][task]
        one_ability.append(info['ability'])
        one_error.append(info['se'])
    abilities.append(one_ability)
    errors.append(one_error)

print('Abilities: ')
print()
for a in abilities:
    print([f"{x:.2f}" for x in a])

print()

print('Errors: ')
print()

for a in errors:
    print([f"{x:.2f}" for x in a])

result_matrix = {}
for adapter in select_models:
    result_matrix[adapter] = {}

    config = PeftConfig.from_pretrained("lorahub/" + adapter)
    lora_model = PeftModel.from_pretrained(base_model, "lorahub/" + adapter).to(device)
    for dataset in select_tasks:
        result_matrix[adapter][dataset] = {}
        datafile = open(os.path.join(DATASETS_DIR, dataset), 'r', encoding='utf8')
        all_f1, all_em = [], []
        for line in tqdm(datafile.readlines()):
            sample = json.loads(line)
            input_ids = tokenizer(sample['inputs'], return_tensors='pt').input_ids.to(device)
            with torch.no_grad():
                outputs = lora_model.generate(input_ids=input_ids)
                predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions = [{'id': '1', 'prediction_text': predicted_answer}]
                references = [{'id': '1', 'answers': {'text': [sample['targets']], 'answer_start': [0]}}]
                results = metric.compute(predictions=predictions, references=references)
                all_f1.append(results['f1'])
                all_em.append(results['exact_match'])
        result_matrix[adapter][dataset]['em'] = sum(all_em) / len(all_em)
        result_matrix[adapter][dataset]['f1'] = sum(all_f1) / len(all_f1)
        with open("result_matrx.json", 'w') as json_file:
            json.dump(result_matrix, json_file, indent=4)

    del lora_model, config
    torch.cuda.empty_cache()
    gc.collect()
