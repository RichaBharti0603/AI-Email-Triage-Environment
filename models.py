from schemas import Category, Priority, Department, TriageAction
from transformers import pipeline

# Centralized Labels for Consistency with OpenEnv
LABELS = {
    'category': [e.value for e in Category],
    'priority': [e.value for e in Priority],
    'department': [e.value for e in Department]
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

    def predict(self, text: str) -> TriageAction:
        """Predict all three objectives for a given email text"""
        if not text or not text.strip():
            return TriageAction(
                category=Category.INQUIRY,
                priority=Priority.LOW,
                department=Department.HR
            )
            
        results = {}
        
        # Separated calls for zero-shot classification on different label sets
        # Multi-label False ensures top prediction for each field
        for key, candidate_labels in self.labels.items():
            res = self.classifier(text, candidate_labels=candidate_labels, multi_label=False)
            results[key] = res['labels'][0] # Top prediction
            
        return TriageAction(**results)

def get_triage_model():
    """Access point for the singleton model"""
    return TriageModel()

if __name__ == "__main__":
    # Test
    model = get_triage_model()
    test_text = "URGENT: The server is down and I cannot access my files!"
    print(f"Testing with: {test_text}")
    print(f"Result: {model.predict(test_text)}")
