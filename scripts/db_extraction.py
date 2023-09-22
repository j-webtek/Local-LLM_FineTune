# db_extraction.py

import sqlite3

def extract_email_content_from_db(db_path):
    """
    Extracts the email content from the 'Content' column of the SQLite database.
    
    :param db_path: Path to the SQLite database file.
    :return: List containing email body content from the database.
    """
    
    # Connect to the SQLite database.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute a query to fetch the email content from the 'Content' column.
    # Here, we assume the table's name is 'emails'. Modify if it has a different name.
    cursor.execute("SELECT Content FROM content")
    
    # Convert the result into a list of email bodies.
    email_bodies = [row[0] for row in cursor.fetchall()]

    # Always remember to close the database connection to free up resources.
    conn.close()

    return email_bodies
