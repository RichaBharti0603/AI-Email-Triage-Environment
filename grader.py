# grader.py
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class EmailGrader:
    """Grade agent performance with multiple metrics"""
    
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        self.metadata = []
    
    def grade(self, prediction: str, ground_truth: str, metadata: Dict = None) -> Dict:
        """Grade a single prediction"""
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        
        if metadata:
            self.metadata.append(metadata)
        return {
            'correct': prediction == ground_truth,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'confidence': self._calculate_confidence(prediction, ground_truth)
        }
    def _calculate_confidence(self, prediction: str, ground_truth: str) -> float:
        """Calculate confidence based on pattern matching"""
        if prediction == ground_truth:
            return 1.0
        if (prediction == 'urgent' and ground_truth == 'normal') or \
           (prediction == 'normal' and ground_truth == 'urgent'):
            return 0.5 
        return 0.0
    def get_final_grade(self) -> Dict[str, Any]:
        """Calculate final grade with multiple metrics"""
        if not self.predictions:
            return {'error': 'No predictions to grade'}
        categories = ['urgent', 'normal', 'spam']
        y_true = [categories.index(gt) for gt in self.ground_truths]
        y_pred = [categories.index(pred) for pred in self.predictions]
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        per_category = {}
        for i, category in enumerate(categories):
            category_mask = [gt == category for gt in self.ground_truths]
            if any(category_mask):
                cat_accuracy = sum(
                    1 for j, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths))
                    if gt == category and pred == category
                ) / sum(category_mask)
                per_category[category] = {
                    'accuracy': cat_accuracy,
                    'total': sum(category_mask)
                }
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_category_accuracy': per_category,
            'total_predictions': len(self.predictions),
            'correct_predictions': sum(1 for p, gt in zip(self.predictions, self.ground_truths) if p == gt),
            'final_score': accuracy * 100  
        }
    def llm_scoring(self) -> Dict[str, Any]:
        """LLM-based scoring (for hackathon evaluation)"""
        final_grade = self.get_final_grade()
        comments = []
        if final_grade['accuracy'] > 0.9:
            comments.append("Excellent performance! Very accurate classification.")
        elif final_grade['accuracy'] > 0.7:
            comments.append("Good performance, but room for improvement.")
        elif final_grade['accuracy'] > 0.5:
            comments.append("Fair performance. Consider improving urgent email detection.")
        else:
            comments.append("Needs significant improvement. Review the reward function.")
        if final_grade['per_category_accuracy'].get('spam', {}).get('accuracy', 0) < 0.5:
            comments.append("Spam detection needs improvement.")
        if final_grade['per_category_accuracy'].get('urgent', {}).get('accuracy', 0) < 0.7:
            comments.append("Urgent email classification needs attention.")
        return {
            **final_grade,
            'llm_feedback': ' '.join(comments),
            'recommendations': comments
        }