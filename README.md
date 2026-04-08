# 📧 AI Email Triage System (Production-Grade Evaluation Pipeline)

An end-to-end **AI-powered email classification and evaluation system** designed with **deterministic inference, structured grading, and reward-based feedback**.  
Built to simulate real-world **AI agent evaluation environments** with reproducible benchmarking.

---

## 🚀 Live Demo

🔗 https://richab820-email-triage-ai.hf.space

> Fully deployed on Hugging Face Spaces with an interactive Gradio UI.

---

## 🧠 Problem Statement

Modern organizations receive thousands of emails daily across multiple departments.  
Manual triaging is:
- ❌ Time-consuming  
- ❌ Error-prone  
- ❌ Not scalable  

This system automates:
- Email classification  
- Priority assignment  
- Department routing  

While also providing a **quantifiable evaluation framework** for AI performance.

---

## 🏗️ System Architecture
User Input (Email)
↓
LLM / Zero-shot Classifier (DistilBART-MNLI)
↓
Structured Output (Category, Priority, Department)
↓
Parsing & Normalization Layer
↓
Evaluation Engine (Grader)
↓
Reward Function (Step-wise Feedback)
↓
Logging + Metrics + UI Display


---

## ⚙️ Features

### ✅ Intelligent Email Classification
- Category: `Spam | Inquiry | Complaint | Request`
- Priority: `Low | Medium | High | Urgent`
- Department: `Sales | Support | HR | Finance | Tech`

---

### ✅ Deterministic Inference
- Fixed seeds (`random`, `numpy`, `torch`)
- Zero-temperature inference
- Reproducible outputs across runs

---

### ✅ Multi-Level Task Evaluation

| Task Level | Objective |
|-----------|----------|
| 🟢 Easy | Predict Category |
| 🟡 Medium | Predict Category + Priority |
| 🔴 Hard | Predict Category + Priority + Department |

---

### ✅ Programmatic Graders (0.0 → 1.0)

- Exact match scoring
- Weighted evaluation:
  - Easy → binary
  - Medium → averaged
  - Hard → full structured scoring

---

### ✅ Reward Function (RL-style Feedback)

Provides **step-wise rewards**, not just final output:

- +0.3 → Correct Category  
- +0.3 → Correct Priority  
- +0.4 → Correct Department  

Penalties:
- ❌ Invalid JSON → -1.0  
- ❌ Empty Output → -0.5  
- ❌ Infinite loops → -0.2  

---

### ✅ Robust Output Parsing

- Regex-based JSON extraction
- Safe parsing (`json.loads`)
- Controlled normalization (Title Case mapping)
- Fault-tolerant execution

---

### ✅ Baseline Evaluation Pipeline

- Deterministic script: `baseline_inference.py`
- Fixed dataset slice
- Reproducible benchmarking
- Outputs:
  - Average scores (Easy/Medium/Hard)
  - Prediction vs Ground Truth comparison
  - Saved results (`baseline_results.json`)

---

### ✅ Observability & Logging

- Structured logs for:
  - Input
  - Raw model output
  - Parsed predictions
  - Scores & reward
- Debug-friendly and failure-resilient

---

### ✅ Interactive UI (Gradio)

- Real-time predictions
- Displays:
  - Category, Priority, Department
  - Task scores (Easy/Medium/Hard)
  - Reward signal
- Sample test cases included

---

## 📂 Project Structure

├── app.py # Gradio UI
├── environment.py # EmailTriageEnv (core environment)
├── schemas.py # Label definitions
├── grader.py # Task evaluators
├── reward.py # Reward function logic
├── baseline_inference.py # Deterministic evaluation script
├── baseline_results.json # Benchmark outputs
├── requirements.txt
└── README.md


---

## 🧪 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/email-triage-ai.git
cd email-triage-ai

2. Install dependencies
pip install -r requirements.txt
3. Run the app
python app.py
📊 Run Baseline Evaluation
python baseline_inference.py

Output:

Console metrics
baseline_results.json
🔐 Environment Variables

If using API-based inference:

export HF_TOKEN=your_token_here

👩‍💻 Author

Richa Bharti
Final Year Engineering Student

⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!

