# Author: Harsh Kohli
# Date Created: 15-09-2024

import torch
import numpy as np
from scipy.optimize import minimize
import evaluate

# Adaptive Item Selection using Fisher Information
def item_information(difficulty, ability_estimate):
    # For Rasch model, the item information function is:
    # I(theta) = P(theta) * [1 - P(theta)]
    P = 1 / (1 + np.exp(-(ability_estimate - difficulty)))
    information = P * (1 - P)
    return information

def select_item(task_dataset, ability_estimate, administered_items):
    # Exclude items that have already been administered
    remaining_items = [item for item in task_dataset if item not in administered_items]
    # Compute information for each remaining item
    for item in remaining_items:
        item['information'] = item_information(item['difficulty'], ability_estimate)
    # Select the item with the highest information
    selected_item = max(remaining_items, key=lambda x: x['information'])
    return selected_item

def estimate_ability(item_difficulties, item_responses):
    # Negative log-likelihood function for a single examinee
    def neg_log_likelihood(theta, item_difficulties, item_responses):
        P = 1 / (1 + np.exp(-(theta - np.array(item_difficulties))))
        # Clip P to avoid log(0)
        epsilon = 1e-10
        P = np.clip(P, epsilon, 1 - epsilon)
        Y = np.array(item_responses)
        nll = -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
        return nll

    # Initial guess for theta
    initial_theta = 0.0
    # Estimate ability (theta) by minimizing negative log-likelihood
    result = minimize(
        neg_log_likelihood,
        initial_theta,
        args=(item_difficulties, item_responses),
        method='BFGS'
    )
    if not result.success:
        print("Ability estimation failed:", result.message)
    theta_estimate = result.x[0]
    # Compute standard error
    P = 1 / (1 + np.exp(-(theta_estimate - np.array(item_difficulties))))
    I = P * (1 - P)
    current_se = 1 / np.sqrt(np.sum(I))
    return theta_estimate, current_se

def get_model_ability(adapter, data, tokenizer, device):
    metric = evaluate.load('squad')
    diffs = [(0-x['avg_log_likelihood']) for x in data]
    min_val, max_val = min(diffs), max(diffs)
    for item in data:
        all = (0-item['avg_log_likelihood'])
        item['difficulty'] = ((all - min_val) / (max_val - min_val)) * 6 - 3
    ability = 0.0
    administered_items = []
    item_responses = []
    item_difficulties = []
    current_se, desired_se = np.inf, 0.45
    max_items = 50

    while current_se > desired_se and len(administered_items) < max_items:
        next_item = select_item(data, ability, administered_items)
        input_ids = tokenizer(next_item['inputs'], return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            outputs = adapter.generate(input_ids=input_ids)
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions = [{'id': '1', 'prediction_text': predicted_answer}]
        references = [{'id': '1', 'answers': {'text': [next_item['targets']], 'answer_start': [0]}}]
        results = metric.compute(predictions=predictions, references=references)
        if results['f1'] > 0.5:
            response = 1
        else:
            response = 0

        administered_items.append(next_item)
        item_responses.append(response)
        item_difficulties.append(next_item['difficulty'])

        ability, current_se = estimate_ability(item_difficulties, item_responses)

    print("SE: " + str(current_se) + " num items tested " + str(len(administered_items)))
    return ability, current_se
