from datasets import load_dataset as hf_load_dataset
import random

def load_email_dataset(sample_size=50):
    dataset = hf_load_dataset("SetFit/enron_spam", split="train")

    processed = []

    for item in dataset:
        text = item["text"]
        label = "spam" if item["label"] == 1 else "normal"

        if "urgent" in text.lower() or "asap" in text.lower():
            priority = 3
            department = "engineering"
            label = "urgent"
        else:
            priority = 1 if label == "normal" else 0
            department = "management" if label == "normal" else "none"

        processed.append({
            "text": text,
            "label": label,
            "priority": priority,
            "department": department
        })

    random.shuffle(processed)

    return processed[:sample_size]