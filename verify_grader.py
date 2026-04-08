import grader

def test_grader():
    # Ground truth
    truth = {
        "category": "Urgent",
        "priority": "High",
        "department": "Tech"
    }

    # Perfect prediction
    perfect_pred = {
        "category": "Urgent",
        "priority": "High",
        "department": "Tech"
    }

    # Partial prediction
    partial_pred = {
        "category": "Urgent",
        "priority": "Low",
        "department": "HR"
    }

    # Incorrect prediction
    incorrect_pred = {
        "category": "Normal",
        "priority": "Low",
        "department": "HR"
    }

    print("Testing Grader System...")
    
    # Test Perfect
    scores_perfect = grader.evaluate(perfect_pred, truth)
    print(f"Perfect Prediction Scores: {scores_perfect}")
    assert scores_perfect["easy"] == 1.0
    assert scores_perfect["medium"] == 1.0
    assert scores_perfect["hard"] == 1.0

    # Test Partial
    scores_partial = grader.evaluate(partial_pred, truth)
    print(f"Partial Prediction Scores: {scores_partial}")
    assert scores_partial["easy"] == 1.0
    assert scores_partial["medium"] == 0.5
    assert scores_partial["hard"] == 0.5 # Category only (0.5), Priority and Dept are wrong

    # Test Incorrect
    scores_incorrect = grader.evaluate(incorrect_pred, truth)
    print(f"Incorrect Prediction Scores: {scores_incorrect}")
    assert scores_incorrect["easy"] == 0.0
    assert scores_incorrect["medium"] == 0.0
    assert scores_incorrect["hard"] == 0.0

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_grader()
