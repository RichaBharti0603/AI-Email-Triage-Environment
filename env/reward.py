def compute_reward(action, gt, weights):
    reward = 0

    if action["label"] == gt["label"]:
        reward += weights["label"]

    if action["priority"] == gt["priority"]:
        reward += weights["priority"]

    if action["department"] == gt["department"]:
        reward += weights["department"]

    return reward