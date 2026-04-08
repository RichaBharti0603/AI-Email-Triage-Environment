# EmailTriageEnv - OpenEnv Hackathon Submission

## Overview
EmailTriageEnv is a production-grade reinforcement learning environment designed for automated email classification and routing. It complies with the **OpenEnv** specification, providing a standardized interface for agents to interact with customer support workflows.

## Environment Specifications

### Observation Space
The observation is a dictionary containing:
- `subject`: The email subject line.
- `body`: The full content of the email.
- `sender`: The sender's email address.
- `urgency_hint`: AI-generated hint about urgency (difficulty-dependent).
- `intent_hint`: AI-generated hint about primary intent.

### Action Space
The action is a dictionary with three categorical fields:
- `category`: `[Spam, Inquiry, Complaint, Request]`
- `priority`: `[Low, Medium, High, Urgent]`
- `department`: `[Sales, Support, HR, Finance, Tech]`

### Reward System
A dense reward in the range `[0.0, 1.0]` based on:
- **Category Match**: 0.4
- **Priority Match**: 0.3
- **Department Match**: 0.3

## Task Difficulties
1. **Easy**: Clear, single-intent emails with direct keywords.
2. **Medium**: Ambiguous emails requiring context for urgency and routing.
3. **Hard**: Multi-intent, noisy emails requiring deep reasoning and priority balancing.

## Setup & Installation

### Local Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the environment validation:
   ```bash
   python environment.py
   ```

### Running Baseline Inference
```bash
python inference.py
```

## Docker Instructions

### Build the Image
```bash
docker build -t email-triage-env .
```

### Run the Container
```bash
docker run -e OPENAI_API_KEY=your_key_here email-triage-env
```

## OpenEnv Compliance
This project passes `openenv validate` and implements the standard `[START]`, `[STEP]`, `[END]` logging format in `inference.py`.
