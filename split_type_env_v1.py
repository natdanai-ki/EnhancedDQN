import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SplitTypeEnv(gym.Env):
    """
    Hourly HVAC env (365 days = 8,760 steps)

    Observation (8):
      indoor_temp, indoor_hum, outdoor_temp, outdoor_hum, pm10, filter_health, hour, day_of_year

    Action: MultiDiscrete [vane(2), temp(14 -> 18..31), fan(3), mode(5)]
      total discrete actions = 2*14*3*5 = 420

    Reward: -(wE*energy + wM*maint_penalty + wC*comfort_penalty)
    """

    metadata = {"render_modes": []}

    def __init__(self, weather_csv: str, reward_weights=None):
        super().__init__()

        self.df = pd.read_csv(weather_csv)
        self.total_steps = len(self.df)

        # Actions: vane, temp_idx, fan, mode
        self.action_space = spaces.MultiDiscrete([2, 14, 3, 5])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.current_step = 0
        self.indoor_temp = 30.0
        self.indoor_hum = 60.0
        self.filter_health = 1.0

        if reward_weights is None:
            reward_weights = {"w_energy": 1.0, "w_maint": 1.0, "w_comfort": 1.0}

        self.reward_weights = {
            "w_energy": float(reward_weights.get("w_energy", 1.0)),
            "w_maint": float(reward_weights.get("w_maint", 1.0)),
            "w_comfort": float(reward_weights.get("w_comfort", 1.0)),
        }

    def set_reward_weights(self, w_energy: float, w_maint: float, w_comfort: float):
        self.reward_weights = {"w_energy": float(w_energy), "w_maint": float(w_maint), "w_comfort": float(w_comfort)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.indoor_temp = 30.0
        self.indoor_hum = 60.0
        self.filter_health = 1.0
        return self._get_obs(self.current_step), {}

    def step(self, action):
        vane, temp_idx, fan, mode = action
        target_temp = 18 + int(temp_idx)

        row = self.df.iloc[self.current_step]
        outdoor_temp = float(row.get("outdoor_temp", row.get("OutdoorTemp", 30.0)))
        outdoor_hum = float(row.get("outdoor_humidity", row.get("OutdoorHum", 60.0)))
        pm10 = float(row.get("pm10", row.get("PM10", 0.0)))
        hour = float(row.get("hour", self.current_step % 24))
        day_of_year = float(row.get("day_of_year", self.current_step // 24 + 1))

        # Simple dynamics
        cooling_power = max(0.0, self.indoor_temp - target_temp) * (0.3 + 0.1 * int(fan))
        self.indoor_temp += 0.1 * (outdoor_temp - self.indoor_temp) - cooling_power

        # Energy proxy
        energy = max(0.5, cooling_power * 1.2)

        # Filter degradation influenced by pm10
        self.filter_health -= 0.00005 * (1.0 + pm10 / 100.0)
        self.filter_health = max(self.filter_health, 0.0)

        comfort_penalty = abs(self.indoor_temp - target_temp)
        maintenance_penalty = (1.0 - self.filter_health)

        wE = self.reward_weights["w_energy"]
        wM = self.reward_weights["w_maint"]
        wC = self.reward_weights["w_comfort"]
        reward = -(wE * energy) - (wM * maintenance_penalty) - (wC * comfort_penalty)

        info = {
            "energy": float(energy),
            "temp": float(self.indoor_temp),
            "filter": float(self.filter_health),
            "comfort_target": float(target_temp),
            "pm10": float(pm10),
            "comfort_penalty": float(comfort_penalty),
            "maintenance_penalty": float(maintenance_penalty),
            "hour": float(hour),
            "day_of_year": float(day_of_year),
        }

        self.current_step += 1
        terminated = self.current_step >= (self.total_steps - 1)
        truncated = False

        obs = self._get_obs(self.current_step if not terminated else self.total_steps - 1)
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self, idx: int):
        row = self.df.iloc[idx]
        outdoor_temp = float(row.get("outdoor_temp", row.get("OutdoorTemp", 30.0)))
        outdoor_hum = float(row.get("outdoor_humidity", row.get("OutdoorHum", 60.0)))
        pm10 = float(row.get("pm10", row.get("PM10", 0.0)))
        hour = float(row.get("hour", idx % 24))
        day_of_year = float(row.get("day_of_year", idx // 24 + 1))

        return np.array(
            [
                self.indoor_temp,
                self.indoor_hum,
                outdoor_temp,
                outdoor_hum,
                pm10,
                self.filter_health,
                hour,
                day_of_year,
            ],
            dtype=np.float32,
        )
