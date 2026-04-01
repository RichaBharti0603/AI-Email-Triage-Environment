# agent_improved.py (improved agent)
import re
from typing import Dict, Any

class ImprovedRuleBasedAgent:
    """Improved agent with better email classification rules"""
    
    def __init__(self):
        # Priority keywords (higher weight = more important)
        self.urgent_keywords = {
            'urgent': 3,
            'critical': 3,
            'emergency': 3,
            'immediate': 2,
            'asap': 2,
            'important': 1,
            'security': 2,
            'breach': 3,
            'down': 2,
            'failure': 2,
            'alert': 1,
            'warning': 1
        }
        
        self.spam_keywords = {
            'free': 2,
            'win': 2,
            'prize': 2,
            'cheap': 1,
            '!!!': 3,
            '$$$': 3,
            'click': 1,
            'offer': 1,
            'limited': 1,
            'guaranteed': 1,
            'viagra': 3,
            'lottery': 2
        }
        
        # Trusted senders (never spam)
        self.trusted_domains = [
            '@company.com',
            '@work.com',
            '@official.com',
            '@bank.com'
        ]
        
        # Spam senders
        self.spam_domains = [
            '@prizes.com',
            '@cheapstuff.com',
            '@spam.com',
            '@offer.com'
        ]
    
    def predict(self, observation: Dict[str, Any]) -> int:
        """Predict email category with improved logic"""
        
        # Extract email text and sender
        email_text = observation['email_text'].lower()
        sender = observation['sender'].lower()
        
        # Calculate scores
        urgent_score = 0
        spam_score = 0
        
        # Check keywords
        for keyword, weight in self.urgent_keywords.items():
            if keyword in email_text:
                urgent_score += weight
        
        for keyword, weight in self.spam_keywords.items():
            if keyword in email_text:
                spam_score += weight
        
        # Check sender domain
        is_trusted = any(domain in sender for domain in self.trusted_domains)
        is_spam_sender = any(domain in sender for domain in self.spam_domains)
        
        # Adjust scores based on sender
        if is_trusted:
            spam_score = 0  # Trusted senders are never spam
            if urgent_score > 0:
                urgent_score += 1  # Boost urgent for trusted senders
        
        if is_spam_sender:
            spam_score += 3  # Likely spam
        
        # Check for exclamation marks (common in urgent/spam)
        exclamation_count = email_text.count('!')
        if exclamation_count > 2:
            if urgent_score > 0:
                urgent_score += 1
            else:
                spam_score += 1
        
        # Check for ALL CAPS words
        caps_words = re.findall(r'[A-Z]{3,}', email_text)
        if caps_words:
            if any(word.lower() in self.urgent_keywords for word in caps_words):
                urgent_score += 2
            else:
                spam_score += 1
        
        # Decision logic with thresholds
        if urgent_score >= 3:
            return 0  # urgent
        elif spam_score >= 3:
            return 2  # spam
        elif urgent_score >= 1 and spam_score < 2:
            return 0  # urgent if some urgency and not spammy
        else:
            return 1  # normal
    
    def predict_with_confidence(self, observation: Dict[str, Any]) -> tuple:
        """Return prediction with confidence score"""
        prediction = self.predict(observation)
        
        # Calculate confidence based on rules
        email_text = observation['email_text'].lower()
        
        if prediction == 0:  # urgent
            urgent_count = sum(1 for kw in self.urgent_keywords if kw in email_text)
            confidence = min(urgent_count / 5, 1.0)
        elif prediction == 2:  # spam
            spam_count = sum(1 for kw in self.spam_keywords if kw in email_text)
            confidence = min(spam_count / 5, 1.0)
        else:  # normal
            # Check if there are no strong indicators
            has_urgent = any(kw in email_text for kw in self.urgent_keywords)
            has_spam = any(kw in email_text for kw in self.spam_keywords)
            if not has_urgent and not has_spam:
                confidence = 0.8
            else:
                confidence = 0.5
        
        return prediction, confidence