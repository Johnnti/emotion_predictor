from transformers import pipeline

model_dir = "distilbert-emotion-model"
classifier = pipeline("text-classification", model=model_dir, top_k=None)

examples = [
    "I am so happy today!",
    "I feel sad and alone.",
    "That movie scared me a lot.",
    "I love my family so much",
]

for text in examples:
    preds = classifier(text)[0]

    best = max(preds, key=lambda x: x['score'])
    print(f"Text: {text}")
    print(f" -> Predicted: {best['label']} ({best['score']:.3f})")
    print()

    