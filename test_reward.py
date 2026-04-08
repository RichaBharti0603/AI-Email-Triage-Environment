from reward import MultiObjectiveReward

def test_reward_system():
    # Ground truth
    truth = {
        "category": "Urgent",
        "priority": "High",
        "department": "Tech"
    }

    reward_fn = MultiObjectiveReward()

    print("--- Case 1: Perfect Match ---")
    perfect_pred = {"category": "Urgent", "priority": "High", "department": "Tech"}
    reward, breakdown = reward_fn.calculate(perfect_pred, truth)
    print(f"Reward: {reward}, Breakdown: {breakdown}")
    assert reward == 1.0
    assert breakdown["total"] == 1.0
    assert breakdown["penalty"] == 0.0

    print("\n--- Case 2: Partial Match (Category Only) ---")
    partial_pred = {"category": "Urgent", "priority": "Low", "department": "HR"}
    reward, breakdown = reward_fn.calculate(partial_pred, truth)
    print(f"Reward: {reward}, Breakdown: {breakdown}")
    assert reward == 0.5
    assert breakdown["components"]["category"] == 0.5
    assert breakdown["penalty"] == 0.0

    print("\n--- Case 3: Missing Fields ---")
    missing_pred = {"category": "Urgent"} # priority and department missing
    reward, breakdown = reward_fn.calculate(missing_pred, truth)
    print(f"Reward: {reward}, Breakdown: {breakdown}")
    # Category (0.5), Missing Priority (-0.1), Missing Dept (-0.1) = 0.3
    assert reward == 0.3
    assert breakdown["penalty"] == -0.2

    print("\n--- Case 4: Invalid Values ---")
    invalid_pred = {"category": "Urgent", "priority": "InvalidValue", "department": "Tech"}
    reward, breakdown = reward_fn.calculate(invalid_pred, truth)
    print(f"Reward: {reward}, Breakdown: {breakdown}")
    # Category (0.5), Invalid Priority (-0.2), Dept (0.2) = 0.5
    assert reward == 0.5
    assert breakdown["penalty"] == -0.2

    print("\n--- Case 5: Empty Action ---")
    empty_pred = {}
    reward, breakdown = reward_fn.calculate(empty_pred, truth)
    print(f"Reward: {reward}, Breakdown: {breakdown}")
    # Penalty -0.3, clipped to 0.0
    assert reward == 0.0
    assert breakdown["penalty"] == -0.3

    print("\n--- Case 6: Completely Wrong Action ---")
    wrong_pred = {"category": "Spam", "priority": "Low", "department": "Sales"}
    reward, breakdown = reward_fn.calculate(wrong_pred, truth)
    print(f"Reward: {reward}, Breakdown: {breakdown}")
    # No matches, no penalties = 0.0
    assert reward == 0.0
    assert breakdown["penalty"] == 0.0

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_reward_system()
