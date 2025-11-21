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

    #Tokenization function
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=128,
        )
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #metric for evaluation
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    #training arguments
    training_args = TrainingArguments(
        output_dir="distilbert-emotion-model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        push_to_hub=False, #we'll push manually after training
    )

    #create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    #train
    trainer.train()

    print("Evalating on test set...")
    metrics = trainer.evaluate(tokenized_dataset["test"])
    print(metrics)

    #save model locally
    save_dir = "distilbert-emotion-model"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Model and tokenzier saved to {save_dir}")

if __name__ == "__main__":
    main()