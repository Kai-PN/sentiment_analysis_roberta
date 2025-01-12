import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"

def preprocess_text(text):
    text = re.sub(r'["“”]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\$', ' dollar ', text)
    text = re.sub(r'%', ' percent ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = text.lower()
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'!!+', '!', text)
    return text

def load_and_preprocess_data():
    train_data = pd.read_csv('train_dataset.csv', sep='\t')
    # train_data = train_data.head(1000)  # Limit the number of samples for demonstration purposes
    train_data['text'] = train_data["Review's Title"].fillna('') + " " + train_data["Review's Content"].fillna('')
    train_data = train_data.drop(["Review's Title", "Review's Content"], axis=1)
    train_data = train_data.rename(columns={'Label': 'label'})
    train_data['text'] = train_data['text'].apply(preprocess_text)
    train_data = train_data[['text', 'label']]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        train_data['text'], train_data['label'], test_size=0.2, random_state=42, stratify=train_data['label']
    )
    return train_texts, test_texts, train_labels, test_labels

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

train_texts, test_texts, train_labels, test_labels = load_and_preprocess_data()

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

label_mapping = {'NEG': 0, 'NEU': 1, 'POS': 2}
train_labels = [label_mapping[label] for label in train_labels]
test_labels = [label_mapping[label] for label in test_labels]

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.015,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()
print(eval_results)

model.save_pretrained('./saved_fine_tuned_model')
tokenizer.save_pretrained('./saved_fine_tuned_model')

# Deployment Functions
def load_model_and_tokenizer(model_path='./saved_fine_tuned_model'):
    try:
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

        # Sets the model to evaluation mode.
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

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

# Example Usage
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