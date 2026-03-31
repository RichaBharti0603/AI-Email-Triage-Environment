# AI Email Triage RL Environment

## Overview
This project simulates an email classification environment where an agent categorizes emails into:
- urgent
- normal
- spam

## Features
- Custom RL environment
- Reward-based feedback system
- Automated grading system
- Rule-based agent demo

## How it Works
1. Environment provides email text
2. Agent selects an action
3. Reward is computed
4. Grader evaluates performance

## Results
Achieved XX% accuracy using rule-based agent.

## Future Improvements
- Train RL agent
- Use real datasets
- Multi-step workflow (classify + route)

## Architecture

This project follows a modular RL environment design:

- env/: core environment logic
- agents/: agent implementations
- evaluation/: grading system
- config.py: experiment configuration

The environment follows a Gym-style API:
- reset()
- step(action)