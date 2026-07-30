"""Tests for the Week 2 deterministic world simulator."""

import pytest

from tiny_transformer_world.week2_dataset import build_week2_dataset
from tiny_transformer_world.world import (
    drop,
    make_initial_state,
    move,
    physical_location,
    step,
    who_is_carrying,
)


def test_arrival_picks_up_every_loose_object() -> None:
    state = make_initial_state(
        {
            "Hairbrush": "Mercury",
            "Sneakers": "Venus",
            "Glasses": "Venus",
            "Key": "Mars",
        }
    )
    state, _, details = step(state, move("Jade", "Venus"))
    assert details["automatic_pickups"] == ["Glasses", "Sneakers"]
    assert who_is_carrying(state, "Glasses") == "Jade"
    assert who_is_carrying(state, "Sneakers") == "Jade"


def test_carried_objects_follow_their_agent() -> None:
    state = make_initial_state(
        {
            "Hairbrush": "Mercury",
            "Sneakers": "Venus",
            "Glasses": "Venus",
            "Key": "Moon",
        }
    )
    state, _, _ = step(state, move("Jade", "Venus"))
    state, _, _ = step(state, move("Jade", "Mars"))
    assert who_is_carrying(state, "Glasses") == "Jade"
    assert who_is_carrying(state, "Sneakers") == "Jade"
    assert physical_location(state, "Glasses") == "Mars"
    assert physical_location(state, "Sneakers") == "Mars"


def test_drop_does_not_trigger_immediate_pickup() -> None:
    state = make_initial_state()
    state, _, _ = step(state, move("Lammy", "Venus"))
    state, _, _ = step(state, drop("Lammy", "Sneakers"))
    assert who_is_carrying(state, "Sneakers") == "Nobody"
    assert physical_location(state, "Sneakers") == "Venus"


def test_another_agent_can_pick_up_a_dropped_object() -> None:
    state = make_initial_state()
    state, _, _ = step(state, move("Lammy", "Venus"))
    state, _, _ = step(state, drop("Lammy", "Sneakers"))
    state, _, _ = step(state, move("Jade", "Venus"))
    assert who_is_carrying(state, "Sneakers") == "Jade"


def test_invalid_drop_is_rejected() -> None:
    state = make_initial_state()
    with pytest.raises(ValueError, match="is not carrying"):
        step(state, drop("Anneena", "Sneakers"))


def test_week2_dataset_answers_and_evidence() -> None:
    dataset = build_week2_dataset()
    assert len(dataset) == 11
    assert [record["answer"] for record in dataset] == [
        "Lammy",
        "Anneena",
        "Jade",
        "Nobody",
        "Penguin",
        "Anneena",
        "Jade",
        "Penguin",
        "Jade",
        "Lammy",
        "Jade",
    ]
    assert dataset[1]["evidence_event_ids"] == [1, 3, 4]
    assert dataset[1]["metrics"]["reasoning_depth"] == 3
    assert dataset[3]["evidence_event_ids"] == []
    assert dataset[3]["metrics"]["reasoning_depth"] == 0
