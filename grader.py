from tasks import easy, medium, hard

def evaluate(pred_action: dict, truth: dict) -> dict:
    """
    Evaluates a prediction against ground truth for all tasks.
    Ensures consistent Title Case comparison via internal graders.
    
    Args:
        pred_action (dict): The action predicted by the agent.
        truth (dict): The ground truth action.
        
    Returns:
        dict: A dictionary containing scores for 'easy', 'medium', and 'hard' tasks.
    """
    return {
        "easy": easy.grade(pred_action, truth),
        "medium": medium.grade(pred_action, truth),
        "hard": hard.grade(pred_action, truth)
    }