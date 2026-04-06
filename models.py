import torch
from transformers import pipeline
from typing import Dict, List

# Centralized Labels for Consistency
LABELS = {
    'category': ['urgent', 'normal', 'spam'],
    'priority': ['High', 'Medium', 'Low'],
    'department': ['Tech', 'HR', 'Sales', 'Billing']
}

class TriageModel:
    """Singleton wrapper for the HuggingFace Zero-Shot Classification pipeline"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TriageModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the distilled model pipeline for CPU"""
        print("Loading DistilBART-MNLI model (optimized for CPU)...")
        # Use valhalla/distilbart-mnli-12-3 as requested
        self.classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=-1 # Force CPU
        )
        
        # Consistent label set
        self.labels = LABELS

    def predict(self, text: str) -> Dict[str, str]:
        """Predict all three objectives for a given email text"""
        if not text or not text.strip():
            return {"category": "none", "priority": "none", "department": "none"}
            
        results = {}
        
        # Separated calls for zero-shot classification on different label sets
        for key, candidate_labels in self.labels.items():
            res = self.classifier(text, candidate_labels=candidate_labels, multi_label=False)
            results[key] = res['labels'][0] # Top prediction
            
        return results

def get_triage_model():
    """Access point for the singleton model"""
    return TriageModel()

if __name__ == "__main__":
    # Test
    model = get_triage_model()
    test_text = "URGENT: The server is down and I cannot access my files!"
    print(f"Testing with: {test_text}")
    print(f"Result: {model.predict(test_text)}")
