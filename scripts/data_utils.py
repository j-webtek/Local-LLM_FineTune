from transformers import AutoTokenizer
import logging

def prepare_data_for_training(tokenizer, email_bodies: list, max_length: int = 128, batch_size: int = 1000):
    """
    Tokenize email bodies in batches and return the tokenized data.

    Args:
    - tokenizer: The tokenizer object used for tokenization.
    - email_bodies (list): List of email bodies to tokenize.
    - max_length (int): Maximum length for tokenization.
    - batch_size (int): Number of email bodies to tokenize in each batch.

    Yields:
    - The tokenized representations of email bodies in the current batch.
    """
    tokenizer.pad_token = tokenizer.eos_token
    
    for i in range(0, len(email_bodies), batch_size):
        batch = email_bodies[i: i + batch_size]
        tokenized_data = tokenizer(batch, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        logging.info(f"Tokenized {len(batch)} email bodies.")
        yield tokenized_data
