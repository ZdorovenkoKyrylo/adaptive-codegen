from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class Obstacle:
    center: np.ndarray
    radius: float

    def contains(self, point: np.ndarray) -> bool:
        return np.linalg.norm(point - self.center) < self.radius

    def distance_to(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.center) - self.radius


@dataclass
class Workspace:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    obstacles: List[Obstacle]

    @property
    def dimension(self) -> int:
        return len(self.bounds_min)

    def filter_relevant_obstacles(self, trajectory: np.ndarray,
                                   margin: float) -> List[Obstacle]:
        relevant = []
        for obs in self.obstacles:
            min_dist = float('inf')
            for point in trajectory:
                dist = obs.distance_to(point)
                min_dist = min(min_dist, dist)
            if min_dist < margin:
                relevant.append(obs)
        return relevant


@dataclass
class TrajectorySpec:
    n_points: int
    state_dim: int
    control_dim: int
    dt: float

    start_state: np.ndarray
    goal_state: np.ndarray

    state_bounds: Tuple[np.ndarray, np.ndarray]
    control_bounds: Tuple[np.ndarray, np.ndarray]
