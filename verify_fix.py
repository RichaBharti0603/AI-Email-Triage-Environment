from environment import EmailTriageEnv

def test_determinism():
    env = EmailTriageEnv(target_difficulty="easy")
    
    # Run 1
    obs1, info1 = env.reset(seed=42)
    # Run 2
    obs2, info2 = env.reset(seed=42)
    
    assert obs1 == obs2
    assert info1 == info2
    print("Determinism check passed!")

if __name__ == "__main__":
    test_determinism()
