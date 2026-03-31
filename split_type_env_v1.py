import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SplitTypeEnv(gym.Env):
    """
    Hourly HVAC env (365 days = 8,760 steps)

    Observation (8, normalized to [0,1]):
      indoor_temp, indoor_hum, outdoor_temp, outdoor_hum, pm10, filter_health, hour, day_of_year

    Action: MultiDiscrete [vane(2), temp(14 -> 18..31), fan(3), mode(5)]
      total discrete actions = 2*14*3*5 = 420

    Reward: -(wE*energy_penalty + wM*maint_penalty + wC*comfort_penalty)
    """

    metadata = {"render_modes": []}

    def __init__(self, weather_csv: str, reward_weights=None):
        super().__init__()

        self.df = pd.read_csv(weather_csv)
        self.total_steps = len(self.df)

        self.action_space = spaces.MultiDiscrete([2, 14, 3, 5])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

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

        self.temp_min, self.temp_max = 16.0, 40.0
        self.hum_min, self.hum_max = 20.0, 100.0
        self.pm10_max = max(200.0, float(pd.to_numeric(self.df.get("pm10", self.df.get("PM10", 0.0)), errors="coerce").max(skipna=True) or 200.0))

    def set_reward_weights(self, w_energy: float, w_maint: float, w_comfort: float):
        self.reward_weights = {
            "w_energy": float(w_energy),
            "w_maint": float(w_maint),
            "w_comfort": float(w_comfort),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        first_row = self.df.iloc[0]
        outdoor_temp = float(first_row.get("outdoor_temp", first_row.get("OutdoorTemp", 30.0)))
        outdoor_hum = float(first_row.get("outdoor_humidity", first_row.get("OutdoorHum", 60.0)))

        self.indoor_temp = float(np.clip(outdoor_temp - 2.0, 24.0, 30.0))
        self.indoor_hum = float(np.clip(outdoor_hum - 5.0, 50.0, 70.0))
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

        fan_multiplier = [0.75, 1.0, 1.25][int(fan)]
        mode_multiplier = 1.0 if int(mode) in (0, 1, 2, 3, 4) else 1.0
        temp_gap = max(0.0, self.indoor_temp - target_temp)
        cooling_power = temp_gap * 0.22 * fan_multiplier * mode_multiplier

        envelope_gain = 0.12 * (outdoor_temp - self.indoor_temp)
        self.indoor_temp += envelope_gain - cooling_power
        self.indoor_temp = float(np.clip(self.indoor_temp, 18.0, 35.0))

        humidity_drift = 0.08 * (outdoor_hum - self.indoor_hum)
        cooling_dryness = 1.2 * cooling_power
        self.indoor_hum += humidity_drift - cooling_dryness
        self.indoor_hum = float(np.clip(self.indoor_hum, 40.0, 75.0))

        degradation_rate = 0.00005 * (1.0 + pm10 / 100.0)
        self.filter_health = float(max(0.0, self.filter_health - degradation_rate))

        degradation_factor = 1.0 + 0.35 * (1.0 - self.filter_health)
        pm10_factor = 1.0 + 0.0015 * max(pm10, 0.0)
        energy = max(0.3, cooling_power * 1.5 * degradation_factor * pm10_factor)

        raw_comfort_penalty = abs(self.indoor_temp - target_temp)
        comfort_penalty = min(raw_comfort_penalty / 10.0, 1.0)
        maintenance_penalty = 1.0 - self.filter_health
        energy_penalty = min(energy / 5.0, 1.0)

        wE = self.reward_weights["w_energy"]
        wM = self.reward_weights["w_maint"]
        wC = self.reward_weights["w_comfort"]
        reward = -(wE * energy_penalty) - (wM * maintenance_penalty) - (wC * comfort_penalty)

        info = {
            "energy": float(energy),
            "temp": float(self.indoor_temp),
            "humidity": float(self.indoor_hum),
            "filter": float(self.filter_health),
            "comfort_target": float(target_temp),
            "pm10": float(pm10),
            "comfort_penalty": float(comfort_penalty),
            "comfort_penalty_raw": float(raw_comfort_penalty),
            "maintenance_penalty": float(maintenance_penalty),
            "energy_penalty": float(energy_penalty),
            "hour": float(hour),
            "day_of_year": float(day_of_year),
        }

        self.current_step += 1
        terminated = self.current_step >= (self.total_steps - 1)
        truncated = False

        obs_idx = self.current_step if not terminated else self.total_steps - 1
        obs = self._get_obs(obs_idx)
        return obs, float(reward), terminated, truncated, info

    def _norm(self, value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    def _get_obs(self, idx: int):
        row = self.df.iloc[idx]
        outdoor_temp = float(row.get("outdoor_temp", row.get("OutdoorTemp", 30.0)))
        outdoor_hum = float(row.get("outdoor_humidity", row.get("OutdoorHum", 60.0)))
        pm10 = float(row.get("pm10", row.get("PM10", 0.0)))
        hour = float(row.get("hour", idx % 24))
        day_of_year = float(row.get("day_of_year", idx // 24 + 1))

        return np.array(
            [
                self._norm(self.indoor_temp, self.temp_min, self.temp_max),
                self._norm(self.indoor_hum, self.hum_min, self.hum_max),
                self._norm(outdoor_temp, self.temp_min, self.temp_max),
                self._norm(outdoor_hum, self.hum_min, self.hum_max),
                self._norm(pm10, 0.0, self.pm10_max),
                self._norm(self.filter_health, 0.0, 1.0),
                self._norm(hour, 0.0, 23.0),
                self._norm(day_of_year, 1.0, 365.0),
            ],
            dtype=np.float32,
        )
