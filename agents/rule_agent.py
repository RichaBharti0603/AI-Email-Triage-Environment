def rule_agent(state):
    text = state["email_text"].lower()

    result = {
        "label": "normal",
        "priority": 1,
        "department": "management"
    }

    if "urgent" in text or "asap" in text:
        result.update({
            "label": "urgent",
            "priority": 3,
            "department": "engineering"
        })

    elif "lottery" in text or "win" in text:
        result.update({
            "label": "spam",
            "priority": 0,
            "department": "none"
        })

    return result