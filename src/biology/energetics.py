class Energetics:
    def __init__(self, config=None):
        self.config = config if config else {}

        self.bmr = self.config.get("base_metabolism", 0.005)
        self.move_coeff = self.config.get("move_coefficient", 0.01)
        self.rest_gain = self.config.get("rest_energy_gain", 0.02)
        self.hunt_cost = self.config.get("hunt_cost", 0.05)
        self.prey_penalty = self.config.get("prey_penalty", 1.2)
        self.max_energy = 1.0

    def calculate_step_cost(
        self, resistance, distance, has_prey=False, is_at_nest=False, hunt_attempt=False
    ):
        cost = self.bmr

        if distance > 0:
            weight_mult = self.prey_penalty if has_prey else 1.0
            move_cost = distance * self.move_coeff * (1.0 + resistance) * weight_mult
            cost += move_cost

        if hunt_attempt:
            cost += self.hunt_cost

        if is_at_nest and distance == 0 and not has_prey:
            cost -= self.rest_gain

        return float(cost)
