# Predictive Maintenance RL Agent

## Overview

This project implements a Reinforcement Learning based Predictive Maintenance system using Deep Q-Networks (DQN). The objective is to optimize maintenance scheduling for industrial machines by minimizing breakdowns while avoiding unnecessary maintenance operations.

The project combines:
- Reinforcement Learning (Part A)
- MLOps practices (Part B)

The agent learns maintenance behavior through interaction with a custom simulated factory environment.

---

# SDG Alignment

This project supports:
- SDG 9 — Industry, Innovation and Infrastructure
- SDG 12 — Responsible Consumption and Production

The RL agent improves industrial reliability while reducing unnecessary maintenance and operational waste.

---

# Project Objectives

- Simulate machine degradation in a factory environment
- Train a DQN agent for predictive maintenance
- Compare RL performance with baseline strategies
- Track experiments and model versions
- Demonstrate reproducibility and MLOps workflow

---

# Reinforcement Learning Setup

## Environment

Custom Gymnasium environment:

sim/factory_env.py

The environment simulates:
- machine degradation
- vibration
- temperature
- production load
- maintenance cycles
- breakdown conditions

---

## State Space

The agent receives:

1. Machine health
2. Vibration level
3. Temperature
4. Days since maintenance
5. Production load

---

## Action Space

| Action | Description |
|---|---|
| 0 | Continue running |
| 1 | Planned maintenance |
| 2 | Emergency repair |

---

## Reward Function

The reward system encourages:
- smooth machine operation
- fewer breakdowns
- efficient maintenance scheduling

Penalties are given for:
- machine failures
- unnecessary maintenance
- emergency repairs

---

# DQN Implementation

The project uses:
- Deep Q-Network (DQN)
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration

DQN was selected because the environment contains continuous sensor values, making traditional Q-learning tables impractical.

---

# Exploration Strategy

An epsilon-greedy strategy is used:
- Initial training uses random exploration
- Epsilon gradually decays
- Final policy exploits learned behavior

This allows the agent to balance exploration and exploitation.

---

# Training

Training command:

python3 train.py

Training duration:
- 100000 timesteps

The training curve demonstrates:
- increasing rewards
- convergence over time
- stable learned policy

---

# Evaluation

Evaluation command:

python3 evaluate.py

Evaluation metrics include:
- average reward
- breakdown frequency
- uptime
- maintenance frequency

---

# Baseline Comparison

Comparison command:

python3 baseline_comparison.py

The project compares:
- Random Agent
- Rule-Based Agent
- DQN RL Agent

Results show that the DQN agent:
- achieves higher rewards
- reduces breakdowns
- performs fewer unnecessary maintenance operations

---

# Results

## Key Outcomes

| Metric | DQN Result |
|---|---|
| Average Reward | ~350 |
| Breakdowns per Year | ~0 |
| Uptime | ~94% |
| Maintenance Operations | Reduced |

The RL agent successfully learns predictive maintenance behavior through trial and error.

---

# Project Structure

predictive_maintenance_rl/
│
├── configs/
│   └── dqn_v1.yaml
│
├── experiments/
│   └── results.csv
│
├── logs/
│   ├── DQN_1/
│   ├── DQN_2/
│   └── DQN_3/
│
├── models/
│   ├── best_model.zip
│   ├── policy_v1.pkl
│   └── policy_v2_explored.pkl
│
├── results/
│   ├── baseline_comparison.png
│   ├── evaluation_results.png
│   └── training_curve.png
│
├── sim/
│   ├── __init__.py
│   └── factory_env.py
│
├── baseline_comparison.py
├── evaluate.py
├── train.py
├── requirements.txt
└── README.md

---

# Experiment Tracking

Experiment tracking is implemented using:

experiments/results.csv

Tracked parameters include:
- run_id
- average reward
- learning rate
- epsilon
- experiment notes

This demonstrates MLOps experiment management.

---

# YAML Configuration

Configuration file:

configs/dqn_v1.yaml

Stores:
- learning rate
- gamma
- epsilon
- training hyperparameters

This improves:
- reproducibility
- experiment tuning
- configuration management

---

# TensorBoard Logs

Training logs are stored in:

logs/

Each DQN folder corresponds to a separate training session and stores:
- reward history
- loss values
- training metrics

TensorBoard visualization:

tensorboard --logdir logs

---

# Versioning (MLOps)

Git and GitHub were used for:
- version control
- experiment tracking
- project history
- reproducibility

Different commits represent:
- experiment updates
- hyperparameter tuning
- debugging
- final project cleanup

---

# Reproducibility

Clone repository:

git clone <repository-url>

cd predictive_maintenance_rl

Install dependencies:

pip install -r requirements.txt

Run training:

python3 train.py

Run evaluation:

python3 evaluate.py

Run baseline comparison:

python3 baseline_comparison.py

---

# Monitoring Plan (Deployment Design)

If deployed in a real factory environment, the following metrics would be monitored:
- machine health
- vibration spikes
- breakdown frequency
- maintenance frequency
- emergency repair rate
- long-term reward degradation

Retraining would be triggered if model performance drops significantly.

---

# Limitations

- Simulated environment — real industrial noise patterns may differ
- Single-machine simulation only
- No real-world deployment performed
- Simplified degradation behavior

---

# Future Improvements

- Multi-machine scheduling
- Real sensor integration
- Online RL training
- Cloud deployment pipeline
- Real-time monitoring dashboard

---

# Conclusion

This project demonstrates how Reinforcement Learning and MLOps practices can be combined to build an intelligent predictive maintenance system.

The DQN agent successfully learns maintenance scheduling behavior that minimizes breakdowns while optimizing operational efficiency.
