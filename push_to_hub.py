from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "distilbert-emotion-model"
hub_model_id = "johnntianokye/distilbert-emotion-simple"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

print("Pusing model to Hugging Face Hub as:", hub_model_id)

model.push_to_hub(hub_model_id)
tokenizer.push_to_hub(hub_model_id)

print("Done!")

