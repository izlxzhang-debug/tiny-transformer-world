"""Deterministic simulator for the controlled fictional world."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

AGENTS = {"Lammy", "Anneena", "Jade", "Penguin"}
OBJECTS = {"Hairbrush", "Sneakers", "Glasses", "Key"}
LOCATIONS = {"Mars", "Mercury", "Venus", "Moon"}
POSSIBLE_ANSWERS = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]

State = dict[str, dict[str, str]]
Event = dict[str, str]


def make_initial_state(
    object_locations: dict[str, str] | None = None,
) -> State:
    """Create the standard complete initial state."""
    if object_locations is None:
        object_locations = {
            "Hairbrush": "Mercury",
            "Sneakers": "Venus",
            "Glasses": "Moon",
            "Key": "Mars",
        }

    state = {
        "agent_locations": {
            "Lammy": "Mercury",
            "Anneena": "Venus",
            "Jade": "Moon",
            "Penguin": "Mars",
        },
        "object_states": dict(object_locations),
    }
    validate_state(state)
    return state


def validate_state(state: State) -> None:
    """Check that a state satisfies all world invariants."""
    if set(state.get("agent_locations", {})) != AGENTS:
        raise ValueError("Every agent must appear exactly once.")
    if set(state.get("object_states", {})) != OBJECTS:
        raise ValueError("Every object must appear exactly once.")

    for agent, location in state["agent_locations"].items():
        if location not in LOCATIONS:
            raise ValueError(f"{agent} has invalid location: {location}")

    valid_object_values = AGENTS | LOCATIONS
    for object_name, holder_or_location in state["object_states"].items():
        if holder_or_location not in valid_object_values:
            raise ValueError(
                f"{object_name} has invalid state: {holder_or_location}"
            )


def move(agent: str, destination: str) -> Event:
    """Construct a move event."""
    return {"action": "move", "agent": agent, "destination": destination}


def drop(agent: str, object_name: str) -> Event:
    """Construct a drop event."""
    return {"action": "drop", "agent": agent, "object": object_name}


def _apply_move(state: State, agent: str, destination: str) -> list[str]:
    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent}")
    if destination not in LOCATIONS:
        raise ValueError(f"Unknown location: {destination}")
    if state["agent_locations"][agent] == destination:
        raise ValueError(f"Invalid move: {agent} is already at {destination}.")

    state["agent_locations"][agent] = destination
    picked_up: list[str] = []
    for object_name, holder_or_location in state["object_states"].items():
        if holder_or_location == destination:
            state["object_states"][object_name] = agent
            picked_up.append(object_name)
    return sorted(picked_up)


def _apply_drop(state: State, agent: str, object_name: str) -> str:
    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent}")
    if object_name not in OBJECTS:
        raise ValueError(f"Unknown object: {object_name}")
    if state["object_states"][object_name] != agent:
        raise ValueError(
            f"Invalid action: {agent} cannot drop {object_name} "
            f"because {agent} is not carrying it."
        )

    drop_location = state["agent_locations"][agent]
    state["object_states"][object_name] = drop_location
    return drop_location


def step(state: State, event: Event) -> tuple[State, str, dict[str, Any]]:
    """Apply exactly one complete event and return the next state."""
    new_state = deepcopy(state)
    action = event.get("action")

    if action == "move":
        agent = event["agent"]
        destination = event["destination"]
        picked_up = _apply_move(new_state, agent, destination)
        description = f"{agent} travels to {destination}."
        if picked_up:
            description += (
                f" {agent} automatically picks up {', '.join(picked_up)}."
            )
        else:
            description += " No loose objects are picked up."
        details: dict[str, Any] = {"automatic_pickups": picked_up}
    elif action == "drop":
        agent = event["agent"]
        object_name = event["object"]
        drop_location = _apply_drop(new_state, agent, object_name)
        description = (
            f"{agent} drops {object_name} on {drop_location}. "
            "Dropping does not trigger automatic pickup."
        )
        details = {
            "automatic_pickups": [],
            "drop_location": drop_location,
        }
    else:
        raise ValueError(f"Unknown action: {action}")

    validate_state(new_state)
    return new_state, description, details


def who_is_carrying(state: State, object_name: str) -> str:
    """Return the carrier of an object, or Nobody."""
    if object_name not in OBJECTS:
        raise ValueError(f"Unknown object: {object_name}")
    holder_or_location = state["object_states"][object_name]
    return holder_or_location if holder_or_location in AGENTS else "Nobody"


def physical_location(state: State, object_name: str) -> str:
    """Return an object's location, following its carrier when necessary."""
    if object_name not in OBJECTS:
        raise ValueError(f"Unknown object: {object_name}")
    holder_or_location = state["object_states"][object_name]
    if holder_or_location in AGENTS:
        return state["agent_locations"][holder_or_location]
    return holder_or_location
