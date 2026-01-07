import pytest

from src.biology.energetics import Energetics


@pytest.fixture
def energetics_config():
    # We ensure these match the .get() keys in energetics.py
    return {
        "base_metabolism": 0.01,
        "move_coefficient": 0.02,
        "rest_energy_gain": 0.05,
        "hunt_cost": 0.1,
        "prey_penalty": 1.5,
    }


def test_bmr_only(energetics_config):
    eng = Energetics(energetics_config)
    cost = eng.calculate_step_cost(resistance=0.5, distance=0, is_at_nest=False)
    assert cost == 0.01


def test_movement_cost(energetics_config):
    eng = Energetics(energetics_config)
    # BMR (0.01) + [Dist (1.0) * Coeff (0.02) * (1 + Res (0.5))]
    # 0.01 + [0.02 * 1.5] = 0.01 + 0.03 = 0.04
    cost = eng.calculate_step_cost(resistance=0.5, distance=1.0, is_at_nest=False)
    assert pytest.approx(cost) == 0.04


def test_prey_penalty(energetics_config):
    eng = Energetics(energetics_config)
    # BMR (0.01) + [Dist (1.0) * Coeff (0.02) * (1 + Res (0.0)) * Penalty (1.5)]
    # 0.01 + [0.02 * 1.0 * 1.5] = 0.01 + 0.03 = 0.04
    cost = eng.calculate_step_cost(
        resistance=0.0, distance=1.0, has_prey=True, is_at_nest=False
    )
    assert pytest.approx(cost) == 0.04


def test_hunt_attempt(energetics_config):
    eng = Energetics(energetics_config)
    # BMR (0.01) + Hunt (0.1) = 0.11
    cost = eng.calculate_step_cost(resistance=0.0, distance=0, hunt_attempt=True)
    assert pytest.approx(cost) == 0.11


def test_nest_recovery(energetics_config):
    eng = Energetics(energetics_config)
    # BMR (0.01) - Rest (0.05) = -0.04
    cost = eng.calculate_step_cost(resistance=0.0, distance=0, is_at_nest=True)
    assert pytest.approx(cost) == -0.04


def test_no_recovery_with_prey(energetics_config):
    eng = Energetics(energetics_config)
    # If carrying prey, recovery is blocked. Should just be BMR.
    cost = eng.calculate_step_cost(
        resistance=0.0, distance=0, is_at_nest=True, has_prey=True
    )
    assert cost == 0.01
