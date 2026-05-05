"""
factory_env.py
==============
Custom OpenAI Gym environment simulating a factory machine
for the Predictive Maintenance RL project.

Subject  : Reinforcement Learning (24AM6PCREL)
Team     : Adithya Vipin, Afrin MTP, Amrith Thejas, Akif Karuvath
SDGs     : SDG 9 (Industry & Innovation), SDG 12 (Responsible Consumption)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FactoryMachineEnv(gym.Env):
    """
    A simulated factory machine environment for predictive maintenance.

    The machine degrades over time based on usage and load.
    The agent observes sensor readings and decides when to act.

    Observation Space (5 features):
        - health        : Machine health level        [0.0, 1.0]
        - vibration     : Vibration sensor reading    [0.0, 1.0]
        - temperature   : Temperature sensor reading  [0.0, 1.0]
        - days_since_mx : Days since last maintenance [0, 100]
        - production_load: Current production load    [0.0, 1.0]

    Action Space (3 discrete actions):
        - 0 : Keep running     (no intervention)
        - 1 : Schedule service (planned maintenance)
        - 2 : Emergency repair (forced shutdown)

    Reward:
        - +1.0  per step machine runs fine
        - −0.5  for scheduling maintenance too early (health > 0.7)
        - +0.3  for smart maintenance (0.3 < health <= 0.7)
        - −3.0  for emergency repair
        - −5.0  for machine breakdown (health reaches 0)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=365):
        super().__init__()
        self.max_steps = max_steps  # Simulate one year by default

        # Define observation space: 5 continuous features
        self.observation_space = spaces.Box(
            low  = np.array([0.0, 0.0, 0.0,  0.0, 0.0], dtype=np.float32),
            high = np.array([1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)

        # Internal state variables (initialised in reset)
        self.health         = None
        self.vibration      = None
        self.temperature    = None
        self.days_since_mx  = None
        self.production_load= None
        self.current_step   = None
        self.breakdown_count= None
        self.maintenance_count = None

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start with a healthy machine
        self.health          = 1.0
        self.vibration       = self.np_random.uniform(0.0, 0.1)
        self.temperature     = self.np_random.uniform(0.2, 0.4)
        self.days_since_mx   = 0.0
        self.production_load = self.np_random.uniform(0.4, 0.8)
        self.current_step    = 0
        self.breakdown_count = 0
        self.maintenance_count = 0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward   = 0.0
        info     = {}
        done     = False

        # ── DEGRADE the machine each step ──────────────────────────────
        # Degradation rate depends on load and how long since last service
        age_factor         = min(self.days_since_mx / 60.0, 1.0)  # worsens after 60 days
        load_factor        = self.production_load
        degradation        = 0.01 + 0.03 * load_factor + 0.02 * age_factor
        degradation       += self.np_random.uniform(0.0, 0.005)   # small random noise

        self.health        = max(0.0, self.health - degradation)
        self.days_since_mx += 1.0

        # Sensors reflect machine health with realistic noise
        noise              = self.np_random.uniform(-0.05, 0.05)
        self.vibration     = np.clip(1.0 - self.health + noise, 0.0, 1.0)
        self.temperature   = np.clip(0.3 + (1.0 - self.health) * 0.6 + noise, 0.0, 1.0)

        # Randomly vary the production load each step
        load_change        = self.np_random.uniform(-0.05, 0.05)
        self.production_load = np.clip(self.production_load + load_change, 0.2, 1.0)

        # ── PROCESS the agent's action ──────────────────────────────────
        if action == 0:
            # Keep running
            if self.health <= 0.0:
                # Machine has broken down!
                reward = -5.0
                self.breakdown_count += 1
                self._do_emergency_repair()
                info["event"] = "breakdown"
            else:
                reward = 1.0  # smooth operation
                info["event"] = "running"

        elif action == 1:
            # Schedule maintenance (planned)
            self.maintenance_count += 1
            if self.health > 0.7:
                # Too early — wasteful but not catastrophic
                reward = -0.5
                info["event"] = "early_maintenance"
            elif self.health > 0.3:
                # Smart timing — efficient and preventive
                reward = 0.3
                info["event"] = "smart_maintenance"
            else:
                # Just in time — better than breakdown but still cutting it close
                reward = 0.1
                info["event"] = "late_maintenance"
            self._do_maintenance()

        elif action == 2:
            # Emergency repair — costly
            reward = -3.0
            self.breakdown_count += 1
            self._do_emergency_repair()
            info["event"] = "emergency_repair"

        # ── CHECK TERMINATION ───────────────────────────────────────────
        self.current_step += 1
        done = self.current_step >= self.max_steps

        info["health"]           = self.health
        info["days_since_mx"]    = self.days_since_mx
        info["breakdown_count"]  = self.breakdown_count
        info["maintenance_count"]= self.maintenance_count

        return self._get_obs(), reward, done, False, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        return np.array([
            self.health,
            self.vibration,
            self.temperature,
            self.days_since_mx,
            self.production_load
        ], dtype=np.float32)

    def _do_maintenance(self):
        """Planned maintenance restores machine to near-full health."""
        self.health        = self.np_random.uniform(0.85, 0.95)
        self.days_since_mx = 0.0
        self.vibration     = self.np_random.uniform(0.0, 0.1)
        self.temperature   = self.np_random.uniform(0.2, 0.35)

    def _do_emergency_repair(self):
        """Emergency repair restores health but not as well as planned maintenance."""
        self.health        = self.np_random.uniform(0.65, 0.80)
        self.days_since_mx = 0.0
        self.vibration     = self.np_random.uniform(0.05, 0.15)
        self.temperature   = self.np_random.uniform(0.25, 0.45)

    # ------------------------------------------------------------------
    def render(self, mode="human"):
        health_bar = "█" * int(self.health * 20) + "░" * (20 - int(self.health * 20))
        print(
            f"Step {self.current_step:>3} | "
            f"Health [{health_bar}] {self.health:.2f} | "
            f"Vibration: {self.vibration:.2f} | "
            f"Temp: {self.temperature:.2f} | "
            f"Days since service: {self.days_since_mx:.0f}"
        )
