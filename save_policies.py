"""
save_policies.py
================
Saves two policy versions from the trained DQN model:
  - policy_v1.pkl        : Best model (fully trained, deterministic)
  - policy_v2_explored.pkl : Model with higher exploration (epsilon = 0.3)

Run:
    python save_policies.py
"""

import pickle
from stable_baselines3 import DQN
from sim.factory_env import FactoryMachineEnv

# ── Load trained model ────────────────────────────────────────────────────────
print("Loading trained model...")
model = DQN.load("models/dqn_predictive_maintenance")

# ── Policy v1: fully trained, deterministic ───────────────────────────────────
policy_v1 = {
    "description"  : "Fully trained DQN policy — deterministic (exp-dqn-1)",
    "epsilon"      : 0.05,
    "learning_rate": 0.001,
    "total_timesteps": 100000,
    "policy_state" : model.policy.state_dict()
}

with open("models/policy_v1.pkl", "wb") as f:
    pickle.dump(policy_v1, f)

print("✅ Saved: models/policy_v1.pkl")

# ── Policy v2: same model, higher exploration for comparison ──────────────────
model.exploration_rate = 0.3   # simulate a more exploratory version

policy_v2 = {
    "description"  : "Exploratory DQN policy — epsilon=0.3 (exp-dqn-2)",
    "epsilon"      : 0.3,
    "learning_rate": 0.001,
    "total_timesteps": 100000,
    "policy_state" : model.policy.state_dict()
}

with open("models/policy_v2_explored.pkl", "wb") as f:
    pickle.dump(policy_v2, f)

print("✅ Saved: models/policy_v2_explored.pkl")
print("\n🎉 Both policy versions saved successfully!")
print("   models/policy_v1.pkl")
print("   models/policy_v2_explored.pkl")
