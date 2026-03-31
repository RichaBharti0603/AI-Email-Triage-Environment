from transformers import pipeline

# Load once (IMPORTANT: do not reload inside function)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


def llm_agent(state):
    text = state["email_text"]

    # Step 1: classify label
    label_result = classifier(
        text,
        candidate_labels=["urgent", "normal", "spam"]
    )

    label = label_result["labels"][0]

    # Step 2: derive structured output
    if label == "urgent":
        return {
            "label": "urgent",
            "priority": 3,
            "department": "engineering"
        }

    elif label == "spam":
        return {
            "label": "spam",
            "priority": 0,
            "department": "none"
        }

    else:
        return {
            "label": "normal",
            "priority": 1,
            "department": "management"
        }