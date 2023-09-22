from database_utils import batch_load_data_from_db
from data_utils import prepare_data_for_training
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_log.log")
    ]
)

import pickle
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
from torch.utils.data import Dataset
from transformers import (
    AdamW, 
    AutoTokenizer, 
    AutoModelForCausalLM, #auto-regressive language ex. GPT models
    Trainer,
    TrainingArguments,
    AutoModelForSeq2SeqLM #encoder-decoder ex. T5 and BART
)

def load_data_from_db(db_path: str) -> list:
    """
    Load email bodies from the SQLite database.
    """
    from db_extraction import extract_email_content_from_db
    return extract_email_content_from_db(db_path)

def initialize_tokenizer_and_model(model_name: str):
    """
    Initialize and return the tokenizer and model.
    """
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("Tokenizer loaded.")

    logging.info("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # may need to adjust based on model
    logging.info("Model loaded.")

    return tokenizer, model

def prepare_data_for_training(tokenizer, email_bodies: list, max_length: int = 128):
    """
    Tokenize the data and prepare it for training.
    """
    logging.info("Tokenizing data...")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_data = tokenizer(email_bodies, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    
    # Set the labels to be the input_ids. This is typical for causal language modeling.
    tokenized_data["labels"] = tokenized_data["input_ids"].detach().clone()

    logging.info("Data tokenized.")

    class TokenizedEmailDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings['input_ids'])

        def __getitem__(self, idx):
            return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
    
    return TokenizedEmailDataset(tokenized_data)

def train_model(model, tokenizer, train_dataset, epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5):
    """
    Handle the training of the model.
    """
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_dir='./logs',
        save_strategy="epoch",
        output_dir="./models/checkpoints",
        logging_steps=10, 
        log_level="info",
        fp16=True,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0  # Gradient clipping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed.")

def save_model(model, tokenizer, save_path: str = "./models/final"):
    """
    Save the trained model and tokenizer.
    """
    logging.info("Saving the final model...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info("Final model saved.")

# The main execution script
if __name__ == "__main__":
    DB_PATH = r"PATH"
    MODEL_NAME = "gpt2-medium" # needs to be changed if switching models
    
    email_bodies = load_data_from_db(DB_PATH)
    tokenizer, model = initialize_tokenizer_and_model(MODEL_NAME)
    train_dataset = prepare_data_for_training(tokenizer, email_bodies)
    train_model(model, tokenizer, train_dataset)
    save_model(model, tokenizer)
