## Hotel Sentiment Analysis - Script Explaination
The script is designed to fine-tune a RoBERTa model for sentiment classification and includes training, evaluation, and deployment functions.

**Important Note:** This is self-note in purpose to better understanding what is going on in the script.

---

### 1. **Importing Libraries**
```python
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
```
- **Libraries:**
  - `os`: To manage system environment variables.
  - `re`: Regular expressions for text preprocessing.
  - `numpy`: Numerical computations.
  - `pandas`: For loading and preprocessing the dataset.
  - `scikit-learn`: For splitting data and calculating metrics.
  - `torch`: PyTorch library for handling datasets and computations.
  - `transformers`: Hugging Face library for pre-trained models and tokenizers.

---

### 2. **Text Preprocessing**
```python
def preprocess_text(text):
    text = re.sub(r'[“”]', '', text)  # Remove quotes
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'\$', ' dollar ', text)
    text = re.sub(r'%', ' percent ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = text.lower() 
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'!!+', '!', text)
    return text
```
- **Purpose:** Clean and normalize text for better performance.
- **Examples:**
  - `$50` -> `dollar 50`
  - `!!!` -> `!`
  - Removes special characters like quotes and extra spaces.

---

### 3. **Loading and Splitting Data**
```python
def load_and_preprocess_data():
    train_data = pd.read_csv('demo_train_dataset.csv', sep='\t') 
    train_data['text'] = train_data["Review's Title"].fillna('') + " " + train_data["Review's Content"].fillna('')
    train_data = train_data.drop(["Review's Title", "Review's Content"], axis=1)
    train_data = train_data.rename(columns={'Label': 'label'})
    train_data['text'] = train_data['text'].apply(preprocess_text)  # Apply preprocessing
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        train_data['text'], train_data['label'], test_size=0.2, random_state=42, stratify=train_data['label']
    )
    return train_texts, test_texts, train_labels, test_labels
```
- **Purpose:** Load the dataset and split it into training and testing sets.
- **Steps:**
  1. Load the dataset from a tab-separated file.
  2. Combine title and content into one `text` column.
  3. Apply text preprocessing.
  4. Split the data into training (80%) and testing (20%) sets.

---

### 4. **Custom Dataset Class**
```python
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
```
- **Purpose:** Define a custom PyTorch-compatible dataset
- **Key Methods:**
  - `__len__`: Returns the number of samples in the dataset.
  - `__getitem__`: Retrieves the `idx`-th sample and returns tokenized inputs and labels.
- **Why?** Hugging Face `Trainer` requires datasets in this format.

---

### 5. **Tokenization**
```python
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
```
- **Purpose:** Convert raw text into token IDs for the RoBERTa model.
- **Steps:**
  - `truncation=True`: Ensures sequences longer than the maximum length are truncated.
  - `padding=True`: Adds padding to shorter sequences to match the maximum length.

---

### 6. **Label Mapping**
```python
label_mapping = {'NEG': 0, 'NEU': 1, 'POS': 2}
train_labels = [label_mapping[label] for label in train_labels]
test_labels = [label_mapping[label] for label in test_labels]
```
- **Purpose:** Map string labels (`NEG`, `NEU`, `POS`) to numeric values required by the model.

---

### 7. **Model and Training Setup**
```python
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8, 
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
```
- **Purpose:**
  - Load a pre-trained RoBERTa model and fine-tune it for sentiment classification.
  - Define training arguments like learning rate, batch size, and number of epochs.
- **Parameters:**
  - `batch_size`: average=8: 
    - larger batch sizes (e.g. 16, 32) potentiall faster convergene and smoother gradient updates; 
    - however, the larger batch size, the more memory is required; it may cause Out of Memmory (OOM) errors; 
    - larger models (e.g., roberta-large) require more memory > may need to reduce the bath size. 
  - `eval_stategy='epoch'`:
    - evaluation runs at the end of every epoch.
    - may use 'steps' for frequent evaluation on large datasets.
  - `learning_rate=2e-5`: 
    - step size for gradient updates; can be adjust between `1e-5` and `5e-5`
    - a lower learning rate results in slower but potentially more stable convergence.
    - a higher learning rate can speed up training but may lead to instability.
  - `number_train_epochs=3`: 
    - number of times the model will iterate over the entire training dataset.
    - more epochs allow the model to learn better but increase training time.
    - start with 3-5 epochs for pre-trained models.
    - use early stopping or cross-validation to prevent overfitting.
  - `weight_decay=0.01`: 
    - adds a penalty to the loss function
    - helps regularize the model, preventing overfitting.
    - the default value (0.01) works well for most tasks
  - `save_total_limit=2`: 
    - limits the number of saved checkpoints
    - prevents storage overflow by keeping only the most recent n checkpoints.
    - `2-3` is good enough. 
---

### 8. **Training and Evaluation**
```python
trainer.train()
eval_results = trainer.evaluate()
print(eval_results)
```
- **Purpose:** Train the model and evaluate it on the test set.
- **Output:** Prints metrics like accuracy, precision, recall, and F1 score.

---

### 9. **Save the Model and Tokenizer**
```python
model.save_pretrained('./saved_fine_tuned_model')
tokenizer.save_pretrained('./saved_fine_tuned_model')
```
- **Purpose:** Save the fine-tuned model and tokenizer for future use.

---

### 10. **Deployment Functions**
```python
def load_model_and_tokenizer(model_path='./saved_fine_tuned_model'):
    try:
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

        # Sets the model to evaluation mode.
        model.eval()

        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Log success
        print(f"Model and tokenizer successfully loaded from {model_path}")
        print(f"Model configured with {model.config.num_labels} labels.")   

        return model, tokenizer

    except Exception as e:
        raise ValueError(f"Error loading model or tokenizer from {model_path}: {e}")

def predict(texts, model, tokenizer):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).numpy()
    label_mapping_reverse = {0: 'NEG', 1: 'NEU', 2: 'POS'}
    return [label_mapping_reverse[pred] for pred in predictions]
```
- **Purpose:**
  - `load_model_and_tokenizer`: Load the saved model and tokenizer.
  - `predict`: Perform inference and return sentiment predictions for input texts.
    - `return_tensors='pt'`: Returns the tokenized data as PyTorch tensors for compatibility with the model.
    - disables gradient computation (`torch.no_grad()`) for inference, reducing memory usage.
    - passes the tokenized inputs to the model for prediction.
    - `outputs.logits`: Raw output scores for each class from the model.
    - `torch.argmax(..., dim=1)`: Selects the index of the highest score for each text, representing the predicted class.
    - converts the predictions to a NumPy array for easy mapping.
  - **(Supplement) Batch Processing for Large Inputs**:
    - If the texts list is very large, the function might exceed memory limits.
    - Split the input into smaller batches:
    ```python
    batch_size = 16  # Adjust based on memory
    predictions = []
    for i in range(0, len(texts), batch_size):
      batch = texts[i:i + batch_size] 
      inputs = tokenizer(batch, truncation=True, padding=True, return_tensors='pt')
      with torch.no_grad():
        outputs = model(**inputs)
      batch_predictions = torch.argmax(outputs.logits, dim=1).numpy()
      predictions.extend(batch_predictions)
    return [label_mapping_reverse[pred] for pred in predictions]
    ```
    `batch = texts[i:i + batch_size]`: 
    - For each iteration, this slices the texts list from index i to i + batch_size.
    - This creates a sublist (batch) of up to batch_size items.

  - **(Supplement) Add Confidence Scores**:
    - Return both the predicted labels and their associated confidence scores:
    ```python
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    confidence_scores = [prob.max() for prob in probabilities]
    return [(label_mapping_reverse[pred], score) for pred, score in zip(predictions, confidence_scores)]
    ```
---

### 11. **Example Usage**
```python
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer('./saved_fine_tuned_model')
    sample_texts = [
        "The product is excellent, works as expected!",
        "Terrible experience, I want my money back.",
        "It's okay, not great but not bad either."
    ]
    predictions = predict(sample_texts, model, tokenizer)
    for text, label in zip(sample_texts, predictions):
        print(f"Text: {text}\nPredicted Sentiment: {label}\n")
```
- **Purpose:** Demonstrate how to load the model and use it for prediction.
- **Output:** Prints the sentiment for each sample text.
