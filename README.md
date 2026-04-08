# OpenEnv Email Triage Project

## Project Description
A fully compliant OpenEnv environment designed for the OpenEnv Hackathon. This project simulates a professional customer support workflow where an agent must automatically triage incoming emails.

## Motivation
Automating email triage is a critical business process that improves response times and ensures customer issues are handled by the right departments. This environment provides a realistic testbed for RL agents to learn multi-objective classification tasks.

## Observation Space
| Field | Type | Description |
| :--- | :--- | :--- |
| `subject` | string | Email subject line |
| `body` | string | Full email content |
| `sender` | string | Sender's email address |
| `urgency_hint` | string | Reasoning hint about urgency |
| `intent_hint` | string | Reasoning hint about primary intent |

## Action Space
| Field | Type | Options |
| :--- | :--- | :--- |
| `category` | string | Spam, Inquiry, Complaint, Request |
| `priority` | string | Low, Medium, High, Urgent |
| `department` | string | Sales, Support, HR, Finance, Tech |

## Tasks
1. **Easy**: Clear, direct requests (e.g., password resets).
2. **Medium**: Emails with professional context requiring reasoning on priority.
3. **Hard**: High-pressure, multi-intent emails with ambiguous formatting.

## Reward Design
The environment provides a dense internal reward based on matching the ground truth:
- Category (0.4)
- Priority (0.3)
- Department (0.3)
Final evaluation is handled by deterministic graders in `graders.py`.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set environment variables if using LLMs: `OPENAI_API_KEY`.

## Running Evaluation
Run the strict inference script:
```bash
python inference.py
```

## Docker Instructions
Build and run the container:
```bash
docker build -t email-triage .
docker run email-triage
```
