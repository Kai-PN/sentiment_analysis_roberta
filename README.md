# Sentiment Analysis with Fine-Tuned RoBERTa

This repository contains a fine-tuned RoBERTa model for sentiment classification. The model predicts the sentiment (`NEG`, `NEU`, `POS`) of input text based on a custom dataset. It can be used for applications like product reviews, feedback analysis, or general sentiment classification tasks.

---

## Features

- **Base Model:** RoBERTa (`roberta-base`)
- **Fine-Tuned Classes:**
  - `NEG`: Negative Sentiment
  - `NEU`: Neutral Sentiment
  - `POS`: Positive Sentiment
- **Flexible Deployment:** Pre-trained model and tokenizer can be used for predictions or further fine-tuning.

---

## Installation

Install the required Python libraries:

`pip install torch transformers scikit-learn datasets` or `pip install -r requirements.txt`

## Clone the Repository
```bash
git clone https://github.com/Kai-PN/sentiment-analysis-roberta.git
cd sentiment-analysis-roberta
```

## Usage
### 1. Training and Evaluation
To train the model, use the `train_sentiment_analysis_roberta.py` script. This script fine-tunes a pre-trained RoBERTa model using your dataset.

Place your training dataset (train_dataset.csv) in the repository directory.
Modify the file paths as needed.
Run the script:

`python train_sentiment_analysis_roberta.py`

The script will:
- Preprocess the dataset.
- Fine-tune a RoBERTa model on the dataset.
- Evaluate the model and print metrics (accuracy, precision, recall, F1-score).
- Save the fine-tuned model and tokenizer in the `saved_fine_tuned_model` directory.

### **Making Predictions**
You can use the saved model to predict sentiments for new input texts. 

**Example code:**
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentiment_analysis_roberta import load_model_and_tokenizer, predict

# Load the saved model and tokenizer
model, tokenizer = load_model_and_tokenizer('./saved_fine_tuned_model')

# Example input texts
texts = [
    "The product is excellent, works as expected!",
    "Terrible experience, I want my money back.",
    "It's okay, not great but not bad either."
]

# Predict sentiments
predictions = predict(texts, model, tokenizer)
for text, sentiment in zip(texts, predictions):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n") 
```

**Expected Output:**
```plaintext
Text: The product is excellent, works as expected!
Predicted Sentiment: POS

Text: Terrible experience, I want my money back.
Predicted Sentiment: NEG

Text: It's okay, not great but not bad either.
Predicted Sentiment: NEU
```

### 2. Deployment and Inference
To use the fine-tuned model for inference, use the `deploy_fine_tuned_model.py` script.

**Using Direct Input**

Run the script with input texts:
```python
python deploy_fine_tuned_model.py --texts "I love this product!"
```
**Using a File**

Provide a text file with one input per line:
```python
python deploy_fine_tuned_model.py --file data/input_texts.txt
```

### 3. Model on Hugging Face
The fine-tuned model is hosted on Hugging Face for easy access. You can load it directly in your scripts using the following:
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
model = RobertaForSequenceClassification.from_pretrained("kai-pn/sentiment-analysis-roberta")
tokenizer = RobertaTokenizer.from_pretrained("kai-pn/sentiment-analysis-roberta")
```
Visit the model page: 
[kai-pn/sentiment-analysis-roberta](https://huggingface.co/kai-pn/sentiment-analysis-roberta). 

### 4. Datasets
Place your training datasets and add correct path  to the directory. The training dataset should follow this format:

**Dataset Format**
 
The training dataset should follow this format:

| Review's Title      | Review's Content                  | Label   |
|----------------------|-----------------------------------|---------|
| Amazing product!     | Works like a charm, I love it!   | POS     |
| Terrible experience  | Completely disappointed with it. | NEG     |
| Decent value         | It's okay, nothing spectacular. | NEU     |

**`Label`**: Sentiment of the review. Must be one of:
  - `NEG`: Negative
  - `NEU`: Neutral
  - `POS`: Positive

## File Structure
```
/sentiment-analysis-roberta
    ├── train_sentiment_analysis_roberta.py      # Main script for full training, eval, & prediction
    ├── deploy_fine_tuned_model.py               # Deployment/inference script
    ├── huggingface_upload.ipynb                 # Huggingface Upload Script - personal (uneccessary)
    ├── train_dataset.csv                        # Sample training dataset (optional)
    ├── README.md                                # Documentation
    ├── LICENSE                                  # Documentation
    ├── detailed_notebook.md                     # Personal Detailed Documents - personal (uneccessary)
    └── requirements.txt                         # Python dependencies
    
```