import sqlite3
import logging

def batch_load_data_from_db(db_path: str, batch_size: int = 10000):
    """
    Load email bodies from a SQLite database in batches.

    Args:
    - db_path (str): Path to the SQLite database.
    - batch_size (int): Number of records to fetch in each batch.

    Yields:
    - List[str]: List of email bodies in the current batch.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    offset = 0
    while True:
        cursor.execute(f"SELECT email_body FROM emails LIMIT {batch_size} OFFSET {offset}")
        email_bodies = [item[0] for item in cursor.fetchall()]
        
        if not email_bodies:
            break

        offset += batch_size
        logging.info(f"Loaded {len(email_bodies)} email bodies from the database from offset {offset}.")
        yield email_bodies

    conn.close()
