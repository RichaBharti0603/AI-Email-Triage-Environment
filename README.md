# EmailTriageEnv - OpenEnv Phase 1 Submission

## Overview
EmailTriageEnv is a production-grade reinforcement learning environment for automated email classification. This version is fully compliant with the **OpenEnv Phase 1** requirements, featuring deterministic seeding via Gymnasium and robust error handling for invalid actions.

## Environment Specifications

### Observation Space
A dictionary containing:
- `subject`: Email subject line.
- `body`: Full email content.
- `sender`: Sender's address.
- `urgency_hint`: Context-aware priority hint.
- `intent_hint`: Context-aware intent hint.

### Action Space
A dictionary with categorical labels:
- `category`: `[Spam, Inquiry, Complaint, Request]`
- `priority`: `[Low, Medium, High, Urgent]`
- `department`: `[Sales, Support, HR, Finance, Tech]`

### Reward & Grading
- **Accuracy Score**: Weighted average of Category (0.4), Priority (0.3), and Department (0.3).
- **Penalty Logic**: A penalty of `-0.1` is applied for malformed actions, and `-0.05` for out-of-bounds categorical values.
- **Gymnasium Standard**: Uses `self.np_random` for reproducible `reset` and `step` transitions.

## Phase 1 Compliance
This project incorporates:
1. **Deterministic Seeding**: `env.reset(seed=42)` guarantees identical initial states.
2. **Unified Grading**: Centralized `graders.py` with `grade_easy`, `grade_medium`, and `grade_hard`.
3. **Strict Logging**: `inference.py` outputs clear `[START]`, `[STEP]`, and `[END]` markers.

## Setup & Baseline

### Install
```bash
pip install -r requirements.txt
```

### Run Baseline
```bash
python inference.py
```
Expected Baseline Score (Heuristic): **~0.7 - 0.9** (Task dependent).

### Docker
```bash
docker build -t openenv-email-triage .
docker run openenv-email-triage
```
