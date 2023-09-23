# Local-LLM_FineTune

Local-LLM_FineTune permits local finetuning of you're pre-trained LLM choice. 
The purpose of this repository is to teach the process of LLM finetuning. 

Please review the docs folder for more information on each script.

I wrote this codebase by research of LLMs and used GPT4, Advanced Data Analysis, to help me write the code. 
You'll find that you'll need a 3090 (24GB VRAM) or higher to finetune in any realistic timeframe.
Please check out TheBloke ( https://huggingface.co/TheBloke ) for GPTQ/GGUFs to train time efficiently. 

Data Documentation
- data/: Contains datasets used for model training and evaluation.
  - train_data.csv: The training dataset containing input-output pairs.
  - valid_data.csv: The validation dataset used during model evaluation.
  - test_data.csv: The testing dataset used to test the model's performance.

---

Logs Documentation
- logs/: Contains logs generated during model training and evaluation.
  - training_logs.txt: Logs generated during model training.
  - evaluation_logs.txt: Logs generated during model evaluation.

---

Models Documentation
- models/: Contains trained model checkpoints.
  - best_model.pt: The best-performing model checkpoint saved during training.

---

Pretrained Models Documentation
- pretrained_models/: Contains pretrained models that can be fine-tuned on specific tasks.
  - language_model.pt: A general-purpose language model.
  - classifier_model.pt: A model pretrained for classification tasks.

---

Scripts Documentation

- Database Utilities (database_utils.py):
  - Contains helper functions to interact with databases.

- Data Processing (data_processing.py):
  - Provides functionality to preprocess and prepare data for training and evaluation.

- DB Extraction (db_extraction.py):
  - Extracts relevant data from the database and saves it to the appropriate location.

- Evaluate Model (evaluate_model.py):
  - Evaluates a trained model on a validation dataset.
  - Outputs evaluation metrics to a file (evaluation_results.txt).

- Training Extract (training_extract.py):
  - Extracts training data and prepares it for model training.

- Train Model (train_model.py):
  - Trains a model using the provided training data.
  - Saves the best-performing model checkpoint.

---

Environment and Requirements

- Python Version: Python 3.10.10 64-bit
- Operating System: Microsoft Windows 11 Home
- System: x64-based PC
- Processor: 13th Gen Intel(R) Core(TM) i7-13700K, 3400 Mhz, 16 Core(s), 24 Logical Processor(s)
- GPU: NVIDIA GeForce RTX 3090 (at least 24GB VRAM)

Required Python Packages:
transformers==4.33.0
torch==2.0.0
torchvision==0.15.1
torchaudio==2.0.1
scikit-learn==1.3.0
numpy==1.24.4
pandas==2.0.3

---
