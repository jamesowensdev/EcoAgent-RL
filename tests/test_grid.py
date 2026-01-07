import logging
import os
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
                "shape": 2.0, "scale": 5.0, "view_dist": 2,
                "intimidation_decay_rate": 0.5, "base_metabolism": 0.05,
                "intimidation_size": 5,
            }
        },
    }

# --- 1. Initialization Variants ---

def test_initialization_variants(basic_config):
    # Coverage for line 48 (None config)
    assert GridManager(None).size_x == 100
    # Standard init
    gm = GridManager(basic_config)
    assert gm.size_x == 100
    assert gm.stacked_layers.dtype == np.float32

def test_initialization_with_secondary_layer(tmp_path, basic_config):
    prey_path = tmp_path / "prey.npy"
    np.save(prey_path, np.ones((100, 100), dtype=np.float32))
    config = basic_config.copy()
    config.update({"prey_path": str(prey_path)})
    gm = GridManager(config)
    assert hasattr(gm, "prey_layer")
    assert gm.prey_layer.shape == (100, 100)

def test_initialization_without_primary_layer(basic_config):
    # This test covers the case where 'primary_layer' is not in the config
    # which should trigger _initialise_generic_grid()
    gm = GridManager(basic_config)
    assert hasattr(gm, "resistance_layer")
    assert gm.resistance_layer.shape == (100, 100)
    assert gm.x_coords is not None
    assert gm.y_coords is not None

# --- 2. Landscape Loading (NPY, TIFF, & Errors) ---

def test_load_landscape_npy(tmp_path, basic_config):
    path = tmp_path / "test.npy"
    np.save(path, np.zeros((100, 100), dtype=np.float32))
    config = basic_config.copy()
    config.update({"primary_layer": "res_path", "res_path": str(path)})
    gm = GridManager(config)
    assert hasattr(gm, "resistance_layer")

@patch("rasterio.open")
@patch("os.path.exists", return_value=True)
def test_load_landscape_tiff_with_nodata(mock_exists, mock_rasterio, basic_config):
    # Setup Success Mock
    mock_src = MagicMock()
    mock_src.read.return_value = np.array([[1, 2], [3, -9999]], dtype=np.float32)
    mock_src.width, mock_src.height, mock_src.nodata = 2, 2, -9999
    mock_src.crs = "EPSG:4326"
    mock_rasterio.return_value.__enter__.return_value = mock_src

    config = basic_config.copy()
    config.update({"primary_layer": "res_path", "res_path": "fake.tif"})

    # Successful TIFF load
    gm = GridManager(config)
    assert gm.size_x == 2
    assert gm.size_y == 2
    assert gm.resistance_layer[1, 1] == 0.0


@patch("rasterio.open")
@patch("os.path.exists", return_value=True)
def test_load_landscape_tiff_success(mock_exists, mock_rasterio, basic_config):
    # Setup Success Mock
    mock_src = MagicMock()
    mock_src.read.return_value = np.zeros((100, 100), dtype=np.float32)
    mock_src.width, mock_src.height, mock_src.nodata = 100, 100, None
    mock_src.crs = "EPSG:4326"
    mock_rasterio.return_value.__enter__.return_value = mock_src

    config = basic_config.copy()
    config.update({"primary_layer": "res_path", "res_path": "fake.tif"})

    # Successful TIFF load
    gm = GridManager(config)
    assert gm.size_x == 100


@patch("rasterio.open")
@patch("os.path.exists", return_value=True)
def test_load_landscape_tiff_exception(mock_exists, mock_rasterio, basic_config, caplog):
    # Line 108-112: Rasterio Exception Path
    mock_rasterio.side_effect = Exception("Rasterio Crash")
    config = basic_config.copy()
    config.update({"primary_layer": "res_path", "res_path": "fake.tif"})
    with caplog.at_level(logging.ERROR):
        GridManager(config)
    assert "Failed to parse" in caplog.text

@patch("rasterio.open")
@patch("os.path.exists", return_value=True)
def test_load_landscape_tiff_crs_mismatch(mock_exists, mock_rasterio, basic_config, caplog):
    # Setup first TIFF load
    mock_src1 = MagicMock()
    mock_src1.read.return_value = np.zeros((100, 100), dtype=np.float32)
    mock_src1.width, mock_src1.height, mock_src1.nodata = 100, 100, None
    mock_src1.crs = "EPSG:4326"

    # Setup second TIFF load with different CRS
    mock_src2 = MagicMock()
    mock_src2.read.return_value = np.zeros((100, 100), dtype=np.float32)
    mock_src2.width, mock_src2.height, mock_src2.nodata = 100, 100, None
    mock_src2.crs = "EPSG:3857"

    # Alternate between the two mocks
    mock_rasterio.side_effect = [
        MagicMock(__enter__=MagicMock(return_value=mock_src1)),
        MagicMock(__enter__=MagicMock(return_value=mock_src2))
    ]

    config = basic_config.copy()
    config.update({
        "primary_layer": "res_path",
        "res_path": "fake1.tif",
        "another_layer_path": "fake2.tif"
    })

    with caplog.at_level(logging.ERROR):
        GridManager(config)
    assert "CRS mismatch" in caplog.text

def test_load_logic_failures(basic_config, tmp_path, caplog):
    gm = GridManager(basic_config)
    # Lines 85-89: File not found
    with caplog.at_level(logging.ERROR):
        gm.load_landscape_layers("absent.npy", "err")
    assert "File not found" in caplog.text

    # Lines 96-98: Unsupported format
    bad = tmp_path / "test.txt"
    bad.write_text("...")
    with caplog.at_level(logging.ERROR):
        gm.load_landscape_layers(str(bad), "err")
    assert "Unsupported file format" in caplog.text

    # Line 111: Size Mismatch Error
    path = tmp_path / "mismatch.npy"
    np.save(path, np.zeros((10, 10), dtype=np.float32))
    with caplog.at_level(logging.ERROR):
        gm.load_landscape_layers(str(path), "mismatch")
    assert "Grid size mismatch" in caplog.text

# --- 3. Normalization & Layer Mechanics ---

def test_normalization_branches(basic_config, caplog):
    gm = GridManager(basic_config)

    # Line 148-149: Non-numpy array attribute
    setattr(gm, "bad_attr", "not_an_array")
    with caplog.at_level(logging.WARNING):
        gm._normalise_layers(["bad_attr"])
    assert "is not a numpy array" in caplog.text

    # Line 161: Uniform layer warning
    gm.prey_layer = np.full((100, 100), 5.0, dtype=np.float32)  # pyright: ignore
    with caplog.at_level(logging.WARNING):
        gm._normalise_layers(["prey_layer"])
    assert "is uniform" in caplog.text

    # Successful Normalization
    gm.prey_layer = np.array([[0, 1], [1, 0]], dtype=np.float32)  # pyright: ignore
    gm._normalise_layers(["prey_layer"])
    assert gm.prey_layer.max() <= 1.0

# --- 4. Spatial Logic & Observables ---

def test_resolve_search_action_valid_moves(basic_config):
    gm = GridManager(basic_config)

    # Straight move
    x, y, dist = gm.resolve_search_action(10, 10, 1)  # Up
    assert (x, y) == (10, 9)
    assert np.isclose(dist, 1.0)

    # Diagonal move
    x, y, dist = gm.resolve_search_action(10, 10, 2)  # Up-Right
    assert (x, y) == (11, 9)
    assert np.isclose(dist, 1.414)


def test_spatial_logic(basic_config):
    gm = GridManager(basic_config)
    assert gm.is_within_bounds(0, 0)
    assert not gm.is_within_bounds(100, 100)

    # Resolve search actions
    assert gm.resolve_search_action(10, 10, 0)[2] == 0.0 # Stay
    assert gm.resolve_search_action(0, 0, 7)[2] > 0.1    # OOB Penalty

def test_intimidation_and_utility(basic_config):
    gm = GridManager(basic_config)
    gm.apply_intimidation(50, 50)
    gm.intimidation_decay()
    assert gm.intimidation_layer[50, 50] == 0.5

    assert gm.set_nest_location(50, 50) is True
    assert gm.set_nest_location(200, 200) is False

def test_get_intimidation_kernel_size_one(basic_config):
    basic_config["species_profiles"]["owl"]["intimidation_size"] = 1
    gm = GridManager(basic_config)
    kernel = gm._get_intimidation_kernel()
    assert kernel.shape == (1, 1)
    assert np.max(kernel) == 1.0


def test_get_intimidation_kernel_edge_cases(basic_config):
    # Test with intimidation_size = 0
    basic_config["species_profiles"]["owl"]["intimidation_size"] = 0
    gm = GridManager(basic_config)
    kernel = gm._get_intimidation_kernel()
    assert kernel.shape == (0, 0)


def test_get_intimidation_kernel(basic_config):
    gm = GridManager(basic_config)
    kernel = gm._get_intimidation_kernel()
    assert kernel.shape == (
        basic_config["species_profiles"]["owl"].get("intimidation_size", 5),
        basic_config["species_profiles"]["owl"].get("intimidation_size", 5)
    )
    assert np.max(kernel) == 1.0

def test_observation_integrity(basic_config):
    gm = GridManager(basic_config)
    obs = gm.get_local_observation(0, 0)
    assert obs.shape[2] == len(gm.layer_names)

    res_idx = gm.layer_names.index("resistance_layer")
    assert np.array_equal(gm.stacked_layers[..., res_idx], gm.resistance_layer)

# --- 5. Foraging & Sightings ---

def test_foraging_snapshot_branches(basic_config, caplog):
    config = basic_config.copy()
    config["habitat"] = {"layers": [
        {"name": "prey", "use_in_suitability": True},
        {"name": "missing", "use_in_suitability": True},
        {"use_in_suitability": True} # Missing name (Line 317)
    ]}
    gm = GridManager(config)
    gm.prey_layer = np.ones((100, 100), dtype=np.float32)  # pyright: ignore
    if hasattr(gm, "missing_layer"): delattr(gm, "missing_layer")

    with caplog.at_level(logging.WARNING):
        snap = gm.get_foraging_snapshot(50, 50)
    assert snap["prey"] == 1.0
    assert "Layer name is None" in caplog.text
    assert "Data for layer 'missing' is missing!" in caplog.text

def test_sighting_branches(basic_config):
    gm = GridManager(basic_config)
    # Line 307: Missing human presence layer check
    if hasattr(gm, "human_presence_layer"): delattr(gm, "human_presence_layer")
    assert gm.generate_sighting_event(0, 0) is None

    gm.human_presence_layer = np.ones((100, 100), dtype=np.float32)
    with patch("numpy.random.random", return_value=0.0):
        assert gm.generate_sighting_event(50, 50, is_hunting=True) == (50, 50)
