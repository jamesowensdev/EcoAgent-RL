from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.habitat.grid import GridManager


@pytest.fixture
def basic_config():
    return {
        "species_name": "owl",
        "species_profiles": {
            "owl": {
                "shape": 2.0,
                "scale": 5.0,
                "view_dist": 2,
                "intimidation_decay_rate": 0.5,
                "base_metabolism": 0.05,
            }
        },
    }


def test_initialization_generic(basic_config):
    # Test initialization without file paths
    gm = GridManager(basic_config)
    assert gm.size_x == 100
    assert gm.size_y == 100
    assert hasattr(gm, "resistance_layer")
    assert hasattr(gm, "prey_layer")
    assert hasattr(gm, "intimidation_layer")
    assert "resistance_layer" in gm.layer_names


def test_load_landscape_layers_npy(tmp_path, basic_config):
    # Test loading from .npy file
    d = tmp_path / "test_layer.npy"
    data = np.random.rand(50, 50).astype(np.float32)
    np.save(d, data)

    config = basic_config.copy()
    config["primary_layer"] = "resistance_path"
    config["resistance_path"] = str(d)

    gm = GridManager(config)
    assert gm.size_x == 50
    assert gm.resistance_layer.shape == (50, 50)


@patch("rasterio.open")
@patch("os.path.exists")
def test_load_landscape_layers_tiff(mock_exists, mock_rasterio, basic_config):
    mock_exists.return_value = True

    # Mock Rasterio Dataset
    mock_src = MagicMock()
    mock_src.read.return_value = np.ones((10, 20), dtype=np.float32)
    mock_src.nodata = -9999
    mock_src.crs = "EPSG:4326"
    mock_src.transform = [1, 0, 0, 0, 1, 0]
    mock_src.width = 20
    mock_src.height = 10
    mock_src.shape = (10, 20)
    mock_rasterio.return_value.__enter__.return_value = mock_src

    config = basic_config.copy()
    config["primary_layer"] = "resistance_path"
    config["resistance_path"] = "fake.tif"

    gm = GridManager(config)
    assert gm.size_x == 20
    assert gm.size_y == 10
    assert gm.crs == "EPSG:4326"


def test_grid_mismatch_error(tmp_path, basic_config):
    # Create two files with different sizes
    p1 = tmp_path / "small.npy"
    p2 = tmp_path / "large.npy"
    np.save(p1, np.zeros((10, 10)))
    np.save(p2, np.zeros((20, 20)))

    config = basic_config.copy()
    config["primary_layer"] = "resistance_path"
    config["resistance_path"] = str(p1)
    config["extra_path"] = str(p2)

    gm = GridManager(config)
    # The second layer should not have been loaded due to size mismatch
    assert not hasattr(gm, "extra_layer")


def test_is_within_bounds(basic_config):
    gm = GridManager(basic_config)  # 100x100
    assert gm.is_within_bounds(0, 0) is True
    assert gm.is_within_bounds(99, 99) is True
    assert gm.is_within_bounds(-1, 50) is False
    assert gm.is_within_bounds(50, 100) is False


def test_resolve_search_action(basic_config):
    gm = GridManager(basic_config)
    # Action 3: East (1, 0)
    new_x, new_y, cost = gm.resolve_search_action(10, 10, 3)
    assert new_x == 11 and new_y == 10
    assert cost == (0.5 * 1.0) + gm.base_metabolism

    # Action 4: Southeast (1, 1) - Diagonal
    _, _, diag_cost = gm.resolve_search_action(10, 10, 4)
    assert diag_cost == (0.5 * 1.414) + gm.base_metabolism

    # Action 0: Stay (0, 0)
    _, _, stay_cost = gm.resolve_search_action(10, 10, 0)
    assert stay_cost == gm.base_metabolism

    # Out of bounds movement
    bx, by, b_cost = gm.resolve_search_action(0, 0, 7)  # West from 0
    assert bx == 0 and by == 0
    assert b_cost == gm.base_metabolism + 0.1


def test_intimidation_mechanics(basic_config):
    gm = GridManager(basic_config)
    gm.apply_intimidation(50, 50)
    assert gm.intimidation_layer[50, 50] == 1.0
    # Check splash (neighboring pixel)
    assert gm.intimidation_layer[51, 51] > 0

    # Test decay
    gm.intimidation_decay()
    assert gm.intimidation_layer[50, 50] == 0.5  # 1.0 * 0.5


def test_nest_and_gamma_utility(basic_config):
    gm = GridManager(basic_config)
    success = gm.set_nest_location(50, 50)
    assert success is True
    assert gm.utility_layer.shape == (100, 100)
    # Peak of Gamma should not be at distance 0
    assert gm.utility_layer[50, 50] < 1.0

    # Test invalid nest
    assert gm.set_nest_location(200, 200) is False


def test_get_local_observation_padding(basic_config):
    gm = GridManager(basic_config)  # view_dist = 2
    # Corner case (0,0) requires padding
    obs = gm.get_local_observation(0, 0)
    # Window size should be (2*dist + 1) = 5
    assert obs.shape == (5, 5, len(gm.layer_names))


def test_sighting_events(basic_config):
    gm = GridManager(basic_config)
    # Ensure the layer exists so bias isn't 0 or missing
    gm.human_presence_layer = np.ones((100, 100), dtype=np.float32)

    # Force sighting: 0.0 is always less than detection_chance
    with patch("numpy.random.random", return_value=0.0):
        sighting = sighting = gm.generate_sighting_event(50, 50, is_hunting=True)
        assert sighting == (50, 50)


def test_normalisation_uniform(basic_config):
    gm = GridManager(basic_config)
    gm.prey_layer = np.full((100, 100), 10.0)  # Uniform
    gm._normalise_layers(["prey_layer"])
    # Should still be 10.0 because denom is 0
    assert gm.prey_layer[0, 0] == 10.0


def test_load_file_not_found(basic_config):
    gm = GridManager(basic_config)
    # Should log error and return None without crashing
    gm.load_landscape_layers("non_existent_file.npy", "test")
    assert not hasattr(gm, "test_layer")
