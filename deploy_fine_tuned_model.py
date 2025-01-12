import os
import re
import argparse
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

def preprocess_text(text):
    """Clean and normalize text data."""
    text = re.sub(r'["“”]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\$', ' dollar ', text)
    text = re.sub(r'%', ' percent ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = text.lower()
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'!!+', '!', text)
    return text

def load_model_and_tokenizer(model_path):
    """Load fine-tuned model and tokenizer from Hugging Face or a local directory."""
    try:
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print(f"Model and tokenizer successfully loaded from {model_path}")
        return model, tokenizer, device

    except Exception as e:
        raise ValueError(f"Error loading model or tokenizer from {model_path}: {e}")

def predict(texts, model, tokenizer, device):
    """Predict sentiments for a list of texts."""
    # Tokenize inputs and move them to the same device as the model
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()  # Move outputs to CPU for processing

    label_mapping_reverse = {0: 'NEG', 1: 'NEU', 2: 'POS'}
    return [label_mapping_reverse[pred] for pred in predictions]

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis with Fine-Tuned RoBERTa")
    parser.add_argument(
        "--model_path",
        type=str,
        default="kai-pn/sentiment-analysis-roberta",
        help="Path to the fine-tuned model (default: Hugging Face hub 'kai-pn/sentiment-analysis-roberta')."
    )
    parser.add_argument("--texts", type=str, nargs='+', help="List of input texts for sentiment analysis.")
    parser.add_argument("--file", type=str, help="Path to a text file containing one input per line.")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_path)

    if args.texts:
        input_texts = args.texts
    elif args.file:
        # Read inputs from the provided file
        with open(args.file, 'r', encoding='utf-8') as f:
            input_texts = [line.strip() for line in f.readlines()]
    else:
        print("Error: No input texts provided. Use --texts or --file.")
        return

    input_texts = [preprocess_text(text) for text in input_texts]
    predictions = predict(input_texts, model, tokenizer, device)

    for text, sentiment in zip(input_texts, predictions):
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()