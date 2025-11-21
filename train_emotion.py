import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

def main():
    #load dataset from huggingface hub
    dataset = load_dataset("dair-ai/emotion")

    #get label names from dataset itself
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    print("Labels:", label_names)

    #load tokenizer and base model (small DistilBERT)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    #TOkenization function