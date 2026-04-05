import torch
from transformers import pipeline
from typing import Dict, List

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
        
        # Define labels
        self.labels = {
            'category': ['urgent', 'normal', 'spam'],
            'priority': ['High', 'Medium', 'Low'],
            'department': ['Tech', 'HR', 'Sales', 'Billing']
        }

    def predict(self, text: str) -> Dict[str, str]:
        """Predict all three objectives for a given email text"""
        results = {}
        
        # We can run them separately or try to combine. 
        # For accuracy, separate calls on labels is usually better for zero-shot.
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
