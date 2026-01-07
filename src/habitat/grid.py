import os

import numpy as np
import rasterio

from src.utils.logger import setup_logger


class GridManager:
    def __init__(self, config):
        self.logger = setup_logger("GRID")
        self.config = config

        self.species = self.config.get("species_name", "default_species")
        profiles = self.config.get("species_profiles", {})

        default_profile = {"shape": 5.0, "scale": 10.0, "view_dist": 15}
        self.bio_profile = profiles.get(self.species, default_profile)

        self.view_dist = self.bio_profile.get("view_dist", 15)
        self.kernel_shape = self.bio_profile.get("shape", 5.0)
        self.kernel_scale = self.bio_profile.get("scale", 10.0)
        self.intimidation_size = self.bio_profile.get("intimidation_size", 5)
        self.intimidation_sigma = self.bio_profile.get("intimidation_sigma", 1.0)
        self.base_metabolism = self.bio_profile.get("base_metabolism", 0.01)
        self.intimidation_decay_rate: float = float(
            self.bio_profile.get("intimidation_decay_rate", 0.95)
        )

        if config and "primary_layer" in config:
            p_key = config["primary_layer"]
            self.load_landscape_layers(config[p_key], layer_type="resistance")
        else:
            self._initialise_generic_grid()

        self.layer_shape = (self.size_y, self.size_x)
        self.intimidation_layer = np.zeros(self.layer_shape, dtype=np.float32)
        self.human_presence_layer = np.zeros(self.layer_shape, dtype=np.float32)
        if config:
            for key, path in config.items():
                if key == "primary_layer" or key == config.get("primary_layer"):
                    continue

                if key.endswith("_path"):
                    layer_name = key.replace("_path", "")
                    if hasattr(self, f"{layer_name}_layer"):
                        continue

                    self.load_landscape_layers(path, layer_type=layer_name)
                    self.logger.info(
                        f"Dynamically loaded secondary layer:  {layer_name}"
                    )
        for layer in ["prey_layer", "utility_layer"]:
            if not hasattr(self, layer):
                self.logger.info(f"{layer} not found in config. Initialised as empty.")
                setattr(self, layer, np.zeros(self.layer_shape, dtype=np.float32))

        self.layer_names = sorted(
            [attr for attr in dir(self) if attr.endswith("_layer")]
        )
        self._generate_spatial_anchors()
        self._normalise_layers()

    def load_landscape_layers(self, file_path, layer_type):
        self.logger.info(f"Attempting to load {layer_type} layer from {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return

        try:
            if file_path.endswith(".tif") or file_path.endswith(".tiff"):
                with rasterio.open(file_path) as src:
                    data = src.read(1).astype(np.float32)

                    if src.nodata is not None:
                        data = np.where(np.isclose(data, src.nodata), 0.0, data)

                    if not hasattr(self, "crs"):
                        self.crs = src.crs
                        self.transform = src.transform
                    elif self.crs != src.crs:
                        self.logger.error(
                            f"CRS mismatch between {self.crs} and {src.crs}"
                        )
                        return

            elif file_path.endswith(".npy"):
                data = np.load(file_path).astype(np.float32)
            else:
                self.logger.error(f"Unsupported file format: {file_path}")
                return
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path} : {e}")
            return

        current_y, current_x = data.shape

        if not hasattr(self, "size_x") or self.size_x is None:
            self.size_y, self.size_x = data.shape
            self.logger.info(
                f"Grid size set to {self.size_y} rows x {self.size_x} columns"
            )
        else:
            if (current_y, current_x) != (self.size_y, self.size_x):
                self.logger.error(
                    f"Grid size mismatch: {current_y} rows x {current_x} columns vs {self.size_y} rows x {self.size_x} columns"
                )
                return

        attr_name = f"{layer_type}_layer"
        setattr(self, attr_name, data)
        self.logger.info(
            f"Succesfully loaded {attr_name}. Min: {np.min(data)} Max: {np.max(data)}"
        )

    def _generate_spatial_anchors(self):
        x_range = np.arange(self.size_x)
        y_range = np.arange(self.size_y)
        self.x_coords, self.y_coords = np.meshgrid(x_range, y_range, indexing="xy")

        self.logger.info(
            f"Generated spatial anchors. X/Y mesh shape: {self.x_coords.shape}"
        )

        return self.x_coords, self.y_coords

    def _initialise_generic_grid(self):
        self.size_x = 100
        self.size_y = 100
        layer_shape = (self.size_y, self.size_x)

        self.resistance_layer = np.full(layer_shape, 0.5, dtype=np.float32)
        self.prey_layer = np.zeros(layer_shape, dtype=np.float32)
        self.utility_layer = np.zeros(layer_shape, dtype=np.float32)

    def _normalise_layers(self, target_layers=None):
        if target_layers is None:
            target_layers = self.layer_names

        for attr in target_layers:
            data = getattr(self, attr)

            if not isinstance(data, np.ndarray):
                self.logger.warning(f"{attr} is not a numpy array")
                continue

            d_min, d_max = np.min(data), np.max(data)
            denom = d_max - d_min

            if denom > 1e-8:
                normalised = (data - d_min) / denom
                setattr(self, attr, normalised.astype(np.float32))
                self.logger.info(f"Normalised {attr} to range [0, 1]")
            else:
                self.logger.warning(
                    f"Layer {attr} is uniform; no normalisation applied"
                )

    def _apply_padding(self, layer, x_max, x_min, y_max, y_min):
        p_top = max(0, -y_min)
        p_bottom = max(0, y_max - self.size_y)
        p_left = max(0, -x_min)
        p_right = max(0, x_max - self.size_x)

        v_y_min, v_y_max = max(0, y_min), min(self.size_y, y_max)
        v_x_min, v_x_max = max(0, x_min), min(self.size_x, x_max)

        chunk = layer[v_y_min:v_y_max, v_x_min:v_x_max]

        if p_top > 0 or p_bottom > 0 or p_left > 0 or p_right > 0:
            chunk = np.pad(
                chunk,
                ((p_top, p_bottom), (p_left, p_right)),
                mode="constant",
                constant_values=0.0,
            )

        return chunk

    def _generate_dist_matrix(self, target_x, target_y):
        dist_matrix = np.sqrt(
            (self.x_coords - target_x) ** 2 + (self.y_coords - target_y) ** 2
        ).astype(np.float32)

        return dist_matrix

    def set_nest_location(self, x, y):
        if not self.is_within_bounds(x, y):
            self.logger.error(f"Nest position ({x}, {y}) is out of bounds")
            return False

        self.nest_x, self.nest_y = x, y

        self.nest_dist_matrix = self._generate_dist_matrix(self.nest_x, self.nest_y)

        self.utility_layer = self._generate_gamma_distribution(
            self.nest_dist_matrix, self.kernel_shape, self.kernel_scale
        )

        return True

    def _generate_gamma_distribution(self, dist_matrix, shape, scale):
        eps = 1e-6
        d = dist_matrix + eps
        log_gamma = (shape - 1) * np.log(d) - (d / scale)
        utility = np.exp(log_gamma - np.max(log_gamma))

        return utility.astype(np.float32)

    def is_within_bounds(self, x, y):
        is_x_valid = 0 <= x < self.size_x
        is_y_valid = 0 <= y < self.size_y

        return is_x_valid and is_y_valid

    def get_local_observation(self, agent_x, agent_y):
        ax, ay = int(agent_x), int(agent_y)

        y_min, y_max = (ay - self.view_dist), (ay + self.view_dist + 1)
        x_min, x_max = (ax - self.view_dist), (ax + self.view_dist + 1)

        layer_names = self.layer_names

        obs_stack = []

        for name in layer_names:
            layer = getattr(self, name)
            obs_stack.append(self._apply_padding(layer, x_max, x_min, y_max, y_min))

        return np.stack(obs_stack, axis=-1)

    def resolve_search_action(self, agent_x, agent_y, action_idx):
        action_map = {
            0: (0, 0),
            1: (0, -1),
            2: (1, -1),
            3: (1, 0),
            4: (1, 1),
            5: (0, 1),
            6: (-1, 1),
            7: (-1, 0),
            8: (-1, -1),
        }

        dx, dy = action_map.get(action_idx, (0, 0))

        target_x = agent_x + dx
        target_y = agent_y + dy

        if not self.is_within_bounds(target_x, target_y):
            self.logger.debug(
                f"Movement blocked, target position ({target_x}, {target_y}) is out of bounds."
            )
            return agent_x, agent_y, self.base_metabolism + 0.1

        resistance = self.resistance_layer[int(target_y), int(target_x)]

        if dx == 0 and dy == 0:
            n_distance = 0.0
        else:
            n_distance = 1.414 if dx != 0 and dy != 0 else 1.0

        return target_x, target_y, n_distance

    def _get_intimidation_kernel(self):
        ax = np.linspace(
            -(self.intimidation_size - 1) / 2.0,
            (self.intimidation_size - 1) / 2.0,
            self.intimidation_size,
        )
        gauss = np.exp(-0.5 * np.square(ax) / np.square(self.intimidation_sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / kernel.max()

    def apply_intimidation(self, x, y):
        self.intimidation_layer[int(y), int(x)] = 1.0

        ix, iy = int(x), int(y)

        kernel = self._get_intimidation_kernel()
        offset = self.intimidation_size // 2

        y1, y2 = max(0, iy - offset), min(self.size_y, iy + offset + 1)
        x1, x2 = max(0, ix - offset), min(self.size_x, ix + offset + 1)

        ky1 = offset - (iy - y1)
        ky2 = ky1 + (y2 - y1)
        kx1 = offset - (ix - x1)
        kx2 = kx1 + (x2 - x1)

        self.intimidation_layer[y1:y2, x1:x2] = np.maximum(
            self.intimidation_layer[y1:y2, x1:x2], kernel[ky1:ky2, kx1:kx2]
        )

    def intimidation_decay(self):
        self.intimidation_layer *= self.intimidation_decay_rate
        self.intimidation_layer[self.intimidation_layer < 1e-4] = 0.0

    def generate_sighting_event(self, agent_x, agent_y, is_hunting=False):
        base_visibility = 0.05
        activity_mult = 5.0 if is_hunting else 1.0
        bias = 1.0

        if hasattr(self, "human_presence_layer"):
            bias = self.human_presence_layer[int(agent_y), int(agent_x)]

        detection_chance = base_visibility * bias * activity_mult

        if np.random.random() < detection_chance:
            self.logger.info(f"Sighting event recorded at ({agent_x}, {agent_y})")
            return (agent_x, agent_y)

        return None

    def get_foraging_snapshot(self, x, y):
        snapshot = {}

        for layer_cfg in self.config["habitat"]["layers"]:
            if layer_cfg.get("use_in_suitability", False):
                layer_name = layer_cfg.get("name")

                if layer_name is None:
                    self.logger.warning(f"Layer name is None for {layer_cfg}")
                    continue

                layer_data = getattr(self, f"{layer_name}_layer", None)

                if layer_data is not None:
                    val = float(layer_data[int(y), int(x)])
                    snapshot[layer_name] = val
                else:
                    self.logger.error(f"Data for layer '{layer_name}' is missing!")
                    snapshot[layer_name] = 0.0

        return snapshot
