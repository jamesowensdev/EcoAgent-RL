import pytest

from src.agents.agent import ForagingAgent


@pytest.fixture
def agent_config():
    return {"max_energy": 1.0, "species": "owl"}


def test_agent_initialization(agent_config):
    agent = ForagingAgent(agent_config, start_x=10, start_y=10)
    assert agent.x == 10.0
    assert agent.y == 10.0
    assert agent.energy == 1.0
    assert agent.is_at_nest is True
    assert agent.is_alive is True


def test_update_location_and_nest_logic(agent_config):
    # Start at nest (10, 10)
    agent = ForagingAgent(agent_config, start_x=10, start_y=10)
    nest_x, nest_y = 10, 10

    # 1. Move away from nest
    dist = agent.update_location(new_x=15, new_y=10, nest_x=nest_x, nest_y=nest_y)
    assert dist == 5.0
    assert agent.is_at_nest is False

    # 2. Move back to nest
    agent.update_location(new_x=10, new_y=10, nest_x=nest_x, nest_y=nest_y)
    assert agent.is_at_nest is True


def test_energy_drain_mechanics(agent_config):
    agent = ForagingAgent(agent_config, start_x=10, start_y=10)

    # Test partial drain
    agent.apply_energetics(0.5)
    assert agent.energy == 0.5
    assert agent.is_alive is True

    # Test death threshold
    agent.apply_energetics(0.5)
    assert agent.energy == 0.0
    assert agent.is_alive is False


def test_energy_recovery_cap(agent_config):
    agent = ForagingAgent(agent_config, start_x=10, start_y=10)
    # Try to gain energy via negative delta while already at max
    agent.apply_energetics(-0.2)
    assert agent.energy == 1.0


def test_agent_reset_clears_all_states(agent_config):
    agent = ForagingAgent(agent_config, start_x=10, start_y=10)
    # Set messy state
    agent.energy = 0.1
    agent.has_prey = True
    agent.total_distance_traveled = 100.0
    agent.is_at_nest = False

    # Reset to a different start position
    agent.reset(start_x=20, start_y=20)

    assert agent.x == 20.0
    assert agent.energy == 1.0
    assert agent.has_prey is False
    assert agent.total_distance_traveled == 0.0
    assert agent.is_at_nest is True
