"""Gradio demo for the fine-tuned FinBERT financial sentiment classifier.

Deployed on Render (not Hugging Face Spaces - Spaces now requires a paid
plan for both Gradio and Docker SDKs). Loads the model directly from the
Hugging Face Hub - no local model files needed in this repo.
"""
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = "UtkarshMishra2209/finbert-finetuned-financial-sentiment"
LABELS = ["negative", "neutral", "positive"]  # index 0,1,2 - must match training label order

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()  # inference mode - disables dropout etc.


def predict(sentence: str):
    if not sentence or not sentence.strip():
        return {}

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]

    # Gradio's "label" output expects a dict of {class: probability}
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


examples = [
    "The company reported a 20% increase in quarterly profit.",
    "The executive group will participate in the adjustments with a fixed-term 5% salary cut.",
    "The company is headquartered in Espoo, Finland.",
    "Sales fell sharply in the third quarter amid weak demand.",
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Financial news sentence",
        placeholder="e.g. The company reported a 20% increase in quarterly profit.",
        lines=3,
    ),
    outputs=gr.Label(label="Predicted sentiment", num_top_classes=3),
    examples=examples,
    title="Financial Sentiment Classifier (FinBERT)",
    description=(
        "A FinBERT model fine-tuned on Financial PhraseBank for 3-class financial "
        "sentiment classification (negative / neutral / positive), from an investor's "
        "perspective. Test set performance: 89.3% accuracy, 0.880 macro F1.\n\n"
        "Part of the project: Financial Sentiment Classification — Fine-Tuning FinBERT vs. BERT."
    ),
)

if __name__ == "__main__":
    # Render sets $PORT at runtime; must bind to 0.0.0.0, not the default
    # 127.0.0.1, so Render's router can actually reach the app.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
