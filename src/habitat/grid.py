import logging
import os

import numpy as np
import rasterio


class GridManager:
    def __init__(self, view_dist=15, config=None):
        self.logger = logging.getLogger(__name__)
        self.view_dist = view_dist

        if config and "primary_layer" in config:
            p_key = config["primary_layer"]
            self.load_landscape_layers(config[p_key], layer_type="resistance")
        else:
            self._initialise_generic_grid()

        self._generate_spatial_anchors()
        self.layer_shape = (self.size_y, self.size_x)

        if config:
            for key, path in config.items():
                if key == "primary_layer" or not key.endswith("_path"):
                    continue

                layer_name = key.replace("_path", "")
                if layer_name == "resistance":
                    continue

                self.load_landscape_layers(path, layer_type=layer_name)
                self.logger.info(f"Dynamically loaded secondary layer:  {layer_name}")

        if not hasattr(self, "prey_layer"):
            self.prey_layer = np.zeros(self.layer_shape, dtype=np.float32)
            self.logger.info("Prey layer not found in config. Initialised as empty.")

        if not hasattr(self, "utility_layer"):
            self.utility_ = np.zeros(self.layer_shape, dtype=np.float32)
            self.logger.info("Utility layer not found in config. Initialised as empty.")

    def load_landscape_layers(self, file_path, layer_type):
        self.logger.info(f"Attempting to load {layer_type} layer from {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return

        try:
            if file_path.endswith(".tif") or file_path.endswith(".tiff"):
                with rasterio.open(file_path) as src:
                    data = src.read(1).astype(np.float32)
                    self.transform = src.transform
                    self.crs = src.crs

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
            self.size_x = current_x
            self.size_y = current_y
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
        return 0

    def _initialise_generic_grid(self):
        self.size_x = 100
        self.size_y = 100
        layer_shape = (self.size_y, self.size_x)

        self.resistance_layer = np.full(layer_shape, 0.5, dtype=np.float32)
        self.prey_layer = np.zeros(layer_shape, dtype=np.float32)
        self.utility_map = np.zeros(layer_shape, dtype=np.float32)

    def _normalise_layers(self, layers):
        return 0

    def _apply_padding(self, dist_matrix):
        return 0

    def _generate_dist_matrix(self, nest_pos):
        return 0

    def update_foraging_utility_map(self, nest_pos, d_min, d_peak, d_max):
        return 0

    def validate_nest_pos(self, nest_pos):
        return 0 <= nest_pos[0] < self.size_x and 0 <= nest_pos[1] < self.size_y

    def get_utility_value(self, nest_pos):
        return 0

    def get_distance_to_nest(self, agent_pos, nest_pos):
        return 0

    def get_normalised_vector_to_nest(self, agent_pos, nest_pos):
        return 0

    def is_within_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.size_x and 0 <= y < self.size_y

    def get_local_observation(self, agent_pos):
        return 0

    def resolve_search_action(self, agent_pos, nest_pos):
        return 0

    def deplete_prey(self, nest_pos):
        return 0

    def regenerate_prey(self, nest_pos):
        return 0

    def generate_sighting_event(self, nest_pos):
        return 0

    def get_global_state(self, nest_pos):
        return 0

    def render_utility_heatmap(self, nest_pos):
        return 0

    def export_grid_to_raster(self, nest_pos):
        return 0
