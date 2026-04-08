# AI Email Triage Environment (OpenEnv)

A production-grade, Gymnasium-compatible reinforcement learning environment for automated email triage.

## 🚀 Overview

The **EmailTriageEnv** simulates a real-world inbox session where an agent must classify incoming emails by **Category**, **Priority**, and **Department**.

- **Compliance**: OpenEnv 1.0.0, Gymnasium 0.29.0
- **Action Space**: Multi-objective (Dict)
- **Observation Space**: Email content and previous context.

## 🛠 Action Space
| Field | Values |
| :--- | :--- |
| **Category** | Spam, Inquiry, Complaint, Request |
| **Priority** | Low, Medium, High, Urgent |
| **Department** | Sales, Support, HR, Finance, Tech |

## 🎓 Task Difficulties
1. **Easy**: Clear, single-intent emails. Evaluation focuses on Category/Priority.
2. **Medium**: Ambiguous emails. Evaluation is balanced across all fields.
3. **Hard**: Noisy, multi-intent, or deceptive emails. Evaluation prioritizes correct routing (Department).

## 📊 Evaluation Logic
The environment uses a multi-objective reward function normalized to `[0.0, 1.0]`. Partial correctness is supported via the `graders.py` module.

## 🏃 Quick Start

### Local Runs
```bash
# Run baseline evaluation
python run_baseline.py --difficulty easy --episodes 20
```

### Docker Deployment
```bash
docker build -t email-triage .
docker run email-triage
```

### Hugging Face Spaces
This project is pre-configured for Hugging Face Spaces. Simply upload the files and ensure `requirements.txt` is present.

## 🧠 Training an Agent
```python
from environment import EmailTriageEnv
env = EmailTriageEnv(config={'difficulty': 'medium', 'max_steps': 5})
obs, info = env.reset()
```
Use standard RL libraries like Stable Baselines3 or RLlib to train on this environment.
