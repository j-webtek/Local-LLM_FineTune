import pandas as pd
import sqlite3
import csv
import os

# Define paths based on the provided information
CSV_FILE_PATH = "C:/Users/Jack/Desktop/PDF Extractor Basic/master-gdc-gdcdatasets-2020445568-2020445568/Emails_Training/emails.csv"
TRAIN_DB_PATH = "C:/Users/Jack/Desktop/Model_Training/training_data.db"
VALID_DB_PATH = "C:/Users/Jack/Desktop/Model_Training/validation_data.db"

# Create the SQLite databases for training and validation data
conn_train = sqlite3.connect(TRAIN_DB_PATH)
conn_valid = sqlite3.connect(VALID_DB_PATH)

# Set chunk size
CHUNKSIZE = 10**4

# Iterate over the dataset in smaller chunks, process them, and write to SQLite databases
for chunk in pd.read_csv(CSV_FILE_PATH, chunksize=CHUNKSIZE):
    # Reconstruct the content using a more efficient method
    chunk['content'] = chunk['tokens'].str.replace("[", "").str.replace("]", "").str.replace("'", "").str.replace(",", " ")
    
    # Split the chunk into training and validation sets
    train_chunk = chunk['content'].sample(frac=0.8, random_state=42)
    valid_chunk = chunk['content'].drop(train_chunk.index)
    
    # Write the data immediately to the SQLite databases
    train_chunk.to_sql('content', conn_train, if_exists='append', index=False)
    valid_chunk.to_sql('content', conn_valid, if_exists='append', index=False)

# Close the database connections
conn_train.close()
conn_valid.close()

print("Data extraction and split completed!")
