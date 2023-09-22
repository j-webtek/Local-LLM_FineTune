import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from data_utils import load_validation_data
import pickle
from datasets import load_metric

# Paths
MODEL_PATH = r"C:\Users\Jack\Desktop\Model_Training\models\final"
VALIDATION_DATA_PATH = r"C:\Users\Jack\Desktop\Model_Training\data\validation_data.txt"
RESULTS_OUTPUT_PATH = r"C:\Users\Jack\Desktop\Model_Training\evaluation_results.txt"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load validation data
validation_data = load_validation_data(VALIDATION_DATA_PATH)

# Tokenize validation data (or load tokenized data if it exists)
tokenized_validation_data_file = VALIDATION_DATA_PATH.replace(".txt", "_tokenized.pkl")
if os.path.exists(tokenized_validation_data_file):
    with open(tokenized_validation_data_file, "rb") as file:
        tokenized_validation_data = pickle.load(file)
else:
    tokenized_validation_data = tokenizer(validation_data, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    with open(tokenized_validation_data_file, "wb") as file:
        pickle.dump(tokenized_validation_data, file)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Standard accuracy metric
    metric_accuracy = load_metric("accuracy")
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    
    # Precision, Recall and F1 Score
    metric_seqeval = load_metric("seqeval")
    results_seqeval = metric_seqeval.compute(predictions=predictions, references=labels)
    precision = results_seqeval['precision']
    recall = results_seqeval['recall']
    f1 = results_seqeval['f1']

    # Perplexity
    loss = eval_pred[0]
    perplexity = np.exp(loss)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'perplexity': perplexity
    }

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    eval_dataset=tokenized_validation_data
)

# Evaluate the model
results = trainer.evaluate()

# Save the results to a plain text file
with open(RESULTS_OUTPUT_PATH, 'w') as file:
    for metric, value in results.items():
        file.write(f"{metric}: {value}\n")

print(f"Results saved to {RESULTS_OUTPUT_PATH}")
