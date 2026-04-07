# grader.py
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class EmailGrader:
    """Grade agent performance with multiple metrics (Category, Priority, Department)"""
    
    def __init__(self):
        self.results = [] # List of dicts: {'predicted': {...}, 'ground_truth': {...}}
    
    def grade(self, prediction: Dict[str, str], ground_truth: Dict[str, str], metadata: Dict = None) -> Dict:
        """Grade a multi-field prediction"""
        res = {
            'predicted': prediction,
            'ground_truth': ground_truth,
            'metadata': metadata or {}
        }
        self.results.append(res)
        
        # Binary correct/incorrect for each field
        field_correct = {
            field: prediction.get(field) == ground_truth.get(field)
            for field in ground_truth.keys()
        }
        
        return {
            'overall_correct': all(field_correct.values()),
            'field_correct': field_correct,
            'prediction': prediction,
            'ground_truth': ground_truth
        }

    def get_final_grade(self) -> Dict[str, Any]:
        """Calculate final grade with metrics for each field"""
        if not self.results:
            return {'error': 'No predictions to grade'}
        
        fields = ['category', 'priority', 'department']
        # Convert keys in results to match expected capitalization if needed
        final_metrics = {}
        
        for field in fields:
            y_true = [r['ground_truth'].get(field) for r in self.results if field in r['ground_truth']]
            y_pred = [r['predicted'].get(field) for r in self.results if field in r['predicted']]
            
            if not y_true:
                continue
                
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            final_metrics[field] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
            
        # Overall Accuracy: All 3 must match
        overall_correct = sum(
            1 for r in self.results 
            if all(r['predicted'].get(f) == r['ground_truth'].get(f) for f in fields)
        )
        
        return {
            'field_metrics': final_metrics,
            'overall_accuracy': overall_correct / len(self.results),
            'total_predictions': len(self.results),
            'correct_predictions_all_fields': overall_correct
        }

    def llm_scoring(self) -> Dict[str, Any]:
        """Generate feedback based on multi-field metrics"""
        final_grade = self.get_final_grade()
        if 'error' in final_grade:
            return final_grade
            
        feedback = []
        metrics = final_grade['field_metrics']
        
        for field, m in metrics.items():
            if m['accuracy'] > 0.8:
                feedback.append(f"Strong performance in {field} classification.")
            elif m['accuracy'] > 0.5:
                feedback.append(f"Moderate accuracy in {field}. Room for improvement.")
            else:
                feedback.append(f"Low accuracy in {field}. Review label definitions.")
                
        return {
            **final_grade,
            'llm_feedback': " ".join(feedback),
            'recommendations': feedback
        }