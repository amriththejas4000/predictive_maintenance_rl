# Predictive Maintenance RL Agent
### Smart Manufacturing Production Optimization

**Course:** Reinforcement Learning (24AM6PCREL)  
**Team:** Adithya Vipin · Afrin MTP · Amrith Thejas · Akif Karuvath  
**SDGs:** SDG 9 (Industry & Innovation) · SDG 12 (Responsible Consumption)

---

## Problem Statement

> "Determine the optimal maintenance action (run, service, or emergency repair) for a factory machine based on real-time sensor readings, to maximize uptime and minimize breakdowns — supporting SDG 9 (Industry & Innovation) and SDG 12 (Responsible Consumption)."

---

## SDG Alignment

| SDG | Connection |
|-----|------------|
| **SDG 9** – Industry, Innovation & Infrastructure | Improves industrial reliability through intelligent, data-driven maintenance decisions |
| **SDG 12** – Responsible Consumption & Production | Reduces unnecessary maintenance, waste, and resource consumption by acting only when needed |

---

## Project Overview

A Reinforcement Learning agent that learns **when to maintain factory machines** to prevent breakdowns while minimizing unnecessary downtime.

The agent observes real-time sensor readings (health, vibration, temperature) and decides whether to:
- Keep running
- Perform planned maintenance
- Perform emergency repair

---

## File Structure

```
predictive_maintenance_rl/
│
├── sim/
│   └── factory_env.py          # Custom Gym environment
│
├── configs/
│   └── dqn_v1.yaml             # Hyperparameter config for reproducibility
│
├── experiments/
│   └── results.csv             # Experiment tracking log
│
├── models/
│   ├── policy_v1.pkl           # Policy after initial training
│   └── policy_v2_explored.pkl  # Policy after exploration tuning
│
├── results/
│   ├── training_curve.png
│   ├── evaluation_results.png
│   └── baseline_comparison.png
│
├── logs/
│   ├── DQN_1/
│   ├── DQN_2/
│   └── DQN_3/
│
├── train.py
├── evaluate.py
├── baseline_comparison.py
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/amriththejas4000/predictive_maintenance_rl.git
cd predictive_maintenance_rl

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Train the model
```bash
python train.py --config configs/dqn_v1.yaml
```

### Evaluate the model
```bash
python evaluate.py
```

### Run baseline comparison
```bash
python baseline_comparison.py
```

---

## Part A – RL Methodology

### Algorithm: DQN (Deep Q-Network)

**Why DQN?**  
DQN is suitable because the state space is continuous (sensor readings) while the action space is discrete (3 actions). DQN handles continuous observations efficiently using a neural network as a function approximator, making it more appropriate than tabular methods like Q-learning or SARSA for this environment.

---

### State, Action, Reward

| Component | Description |
|-----------|-------------|
| **State** | `[health, vibration, temperature, days_since_service, production_load]` — 5 continuous sensor readings |
| **Action** | `0` = Keep running · `1` = Schedule maintenance · `2` = Emergency repair |
| **Reward** | `+1.0` smooth operation · `-0.5` early maintenance · `+0.3` smart maintenance · `-3.0` emergency repair · `-5.0` breakdown |

---

### Exploration Strategy

- **Epsilon-greedy** with linear decay
- Starts at `ε = 1.0` (full exploration) → decays to `ε = 0.05` (mostly exploitation)
- Exploration fraction: 30% of total training timesteps
- This ensures the agent explores diverse maintenance strategies early before committing to a policy

---

### Convergence

Over 100,000 training timesteps, the agent's average episode reward increased from approximately -50 in early episodes to a stable ~350 by the end of training. The smoothed learning curve shows consistent improvement with reduced variance, indicating the agent has converged to a reliable maintenance policy — scheduling service proactively before breakdowns occur rather than reacting after the fact.

---

### Saved Policies

| File | Description |
|------|-------------|
| `models/policy_v1.pkl` | Policy after initial training run (exp-dqn-1) |
| `models/policy_v2_explored.pkl` | Policy after exploration tuning (exp-dqn-2) |

---

## Part B – MLOps Implementation

### Versioning

Git commits and tags track each experiment:

| Tag | Description |
|-----|-------------|
| `exp-dqn-1` | Initial model — base DQN training |
| `exp-dqn-2` | Experiment tracking and logging added |
| `exp-dqn-3` | Final hyperparameter tuning |

---

### Experiment Tracking

All runs logged in `experiments/results.csv`:

| run_id | episodes | avg_reward | avg_breakdowns | maintenance_count | learning_rate | epsilon | notes |
|--------|----------|------------|----------------|-------------------|---------------|---------|-------|
| run1 | 100000 | 351 | 0.05 | 14.8 | 0.001 | 0.05 | initial training |
| run2 | 100000 | 348 | 0.06 | 15.2 | 0.001 | 0.05 | tuned parameters |
| run3 | 100000 | 350 | 0.05 | 15.0 | 0.001 | 0.05 | final tuning |

---

### Reproducibility

To reproduce any experiment exactly:

```bash
# Clone the repo
git clone https://github.com/amriththejas4000/predictive_maintenance_rl.git
cd predictive_maintenance_rl

# Install dependencies
pip install -r requirements.txt

# Run with the exact config used in exp-dqn-1
python train.py --config configs/dqn_v1.yaml
```

Anyone should be able to clone this repo and reproduce the same results using the config files in `configs/`.

---

### Monitoring Plan

If deployed in a real factory, we would monitor average machine health across all active units, breakdown frequency per week, and the ratio of planned-to-emergency maintenance events. Sensor anomalies such as sudden spikes in vibration or temperature beyond normal thresholds would trigger alerts. We would also track reward trends over time — a drop in average reward would indicate the policy is no longer effective, possibly due to machine aging or changed operating conditions, triggering a retraining pipeline.

---

## Baseline vs RL Comparison

| Metric | Random Agent | Rule-Based Agent | DQN (Ours) |
|--------|-------------|-----------------|------------|
| Avg Reward / Year | ~50 | ~280 | ~350 |
| Avg Breakdowns / Year | ~8 | ~1.2 | ~0.05 |
| Avg Maintenances / Year | ~120 | ~14 | ~15 |

DQN outperforms both baselines — achieving the highest reward with the fewest breakdowns while keeping maintenance frequency efficient.

---

## SDG Impact

Compared to the random agent baseline, our DQN policy reduces breakdowns by over **99%** and eliminates unnecessary emergency repairs. This directly supports:

- **SDG 9:** More reliable industrial infrastructure through intelligent, predictive maintenance
- **SDG 12:** Responsible resource use by avoiding unnecessary maintenance interventions and reducing machine downtime waste

---

## Key Insights

- RL outperforms both random and rule-based agents significantly
- The agent learns to act in the **0.3–0.7 health range** — the optimal maintenance window
- Emergency repairs drop to near zero once the agent converges
- Rule-based agents are rigid; RL adapts to varying production loads

---

## Limitations

- Simulated environment — real sensor noise patterns may differ
- Simplified single-machine system (no multi-machine dependencies)
- Degradation model is linear — real machines may degrade non-linearly
- No real-world deployment testing

---
