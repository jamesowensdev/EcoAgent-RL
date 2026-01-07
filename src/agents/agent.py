import numpy as np

from src.utils.logger import setup_logger


class ForagingAgent:
    def __init__(self, config, start_x, start_y):
        self.config = config
        self.logger = setup_logger("AGENT")
        self.x = float(start_x)
        self.y = float(start_y)

        self.energy = 1.0
        self.is_alive = True
        self.has_prey = False
        self.is_at_nest = True
        self.hunt_attempt = False

        self.total_distance_traveled = 0.0
        self.successful_hunts = 0
        self.logger.info(
            f"Agent initialized: StartPos=({start_x}, {start_y}), Energy={self.energy}"
        )

    def apply_energetics(self, delta):
        if delta > 0.5:
            self.logger.warning(f" Excessive energy consumption detected: {delta}")

        self.energy -= delta

        self.energy = float(np.clip(self.energy, 0.0, 1.0))

        if self.energy == 0.0:
            self.is_alive = False
            self.logger.info("Agent died due to energy depletion")

    def update_location(self, new_x, new_y, nest_x, nest_y):
        old_x, old_y = self.x, self.y
        self.x = float(new_x)
        self.y = float(new_y)
        nest_transition = False
        nest_state = self.is_at_nest
        self.is_at_nest = int(self.x) == nest_x and int(self.y) == nest_y
        if nest_state != self.is_at_nest:
            nest_transition = True

        if nest_transition and self.is_at_nest:
            self.logger.info("Agent entered the nest")
        elif nest_transition and not self.is_at_nest:
            self.logger.info("Agent left the nest")

        dist = np.sqrt((self.x - old_x) ** 2 + (self.y - old_y) ** 2)
        self.total_distance_traveled += dist

        return dist

    def reset(self, start_x, start_y):
        self.x = float(start_x)
        self.y = float(start_y)
        self.is_at_nest = True
        self.energy = 1.0
        self.is_alive = True
        self.has_prey = False
        self.hunt_attempt = False
        self.total_distance_traveled = 0.0
        self.successful_hunts = 0
        self.logger.info(
            f"Agent reset: StartPos=({start_x}, {start_y}), Energy={self.energy}"
        )
