from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class Category(str, Enum):
    URGENT = "Urgent"
    NORMAL = "Normal"
    SPAM = "Spam"

class Priority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class Department(str, Enum):
    TECH = "Tech"
    HR = "HR"
    SALES = "Sales"
    BILLING = "Billing"

class EmailObservation(BaseModel):
    """Observation space for the email triage environment"""
    email_text: str = Field(..., description="The content of the email to be triaged")
    previous_action: Optional[str] = Field(None, description="The category assigned to the previous email")

class TriageAction(BaseModel):
    """Action space for the email triage environment"""
    category: Category
    priority: Priority
    department: Department

    class Config:
        use_enum_values = True
