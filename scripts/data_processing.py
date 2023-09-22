# data_processing.py

import torch
from torch.utils.data import Dataset

class EmailDataset(Dataset):
    """
    PyTorch Dataset for handling email body data.
    """

    def __init__(self, email_bodies):
        """
        Constructor for the EmailDataset class.

        :param email_bodies: List of email body content.
        """
        self.texts = email_bodies

    def __len__(self):
        """
        Returns the number of emails in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Allows the dataset to be indexed, so that it can work with the DataLoader.

        :param idx: Index of the desired email body.
        :return: Dictionary with a single key 'text' containing the desired email body.
        """
        return {'text': self.texts[idx]}
