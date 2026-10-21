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


import json
from pathlib import Path


STORIES: list[dict[str, Any]] = [
    {
        "number": 1,
        "seed": 1001,
        "events": [move("Lammy", "Venus"), move("Lammy", "Mars")],
        "question_object": "Key",
    },
    {
        "number": 2,
        "seed": 1002,
        "events": [
            move("Lammy", "Venus"),
            move("Lammy", "Mars"),
            drop("Lammy", "Sneakers"),
            move("Anneena", "Mars"),
        ],
        "question_object": "Sneakers",
    },
    {
        "number": 3,
        "seed": 1003,
        "events": [move("Jade", "Mercury")],
        "question_object": "Hairbrush",
    },
    {
        "number": 4,
        "seed": 1004,
        "events": [move("Lammy", "Mars"), move("Anneena", "Mercury")],
        "question_object": "Glasses",
    },
    {
        "number": 5,
        "seed": 1005,
        "events": [move("Penguin", "Venus"), move("Penguin", "Moon")],
        "question_object": "Glasses",
    },
    {
        "number": 6,
        "seed": 1006,
        "events": [move("Anneena", "Mercury"), move("Anneena", "Venus")],
        "question_object": "Sneakers",
    },
    {
        "number": 7,
        "seed": 1007,
        "events": [
            move("Lammy", "Venus"),
            drop("Lammy", "Sneakers"),
            move("Jade", "Venus"),
        ],
        "question_object": "Sneakers",
    },
    {
        "number": 8,
        "seed": 1008,
        "events": [move("Penguin", "Mercury"), move("Penguin", "Venus")],
        "question_object": "Hairbrush",
    },
    {
        "number": 9,
        "seed": 1009,
        "object_locations": {
            "Hairbrush": "Mercury",
            "Sneakers": "Venus",
            "Glasses": "Venus",
            "Key": "Mars",
        },
        "events": [move("Jade", "Venus")],
        "question_object": "Glasses",
    },
    {
        "number": 10,
        "seed": 1010,
        "events": [
            move("Anneena", "Mars"),
            drop("Anneena", "Key"),
            move("Lammy", "Mars"),
        ],
        "question_object": "Key",
    },
    {
        "number": 11,
        "seed": 1011,
        "object_locations": {
            "Hairbrush": "Mercury",
            "Sneakers": "Venus",
            "Glasses": "Venus",
            "Key": "Moon",
        },
        "events": [move("Jade", "Venus"), move("Jade", "Mars")],
        "question_object": "Glasses",
    },
]


def build_dataset_record(
    *,
    story_id: str,
    seed: int,
    events: list[Event],
    question_object: str,
    object_locations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Simulate one story and create an auditable dataset record."""
    state = make_initial_state(object_locations)
    initial_state = deepcopy(state)
    event_records: list[dict[str, Any]] = []
    state_trace: list[dict[str, Any]] = [
        {"time": 0, "event_id": None, "state": deepcopy(state)}
    ]
    evidence_event_ids: list[int] = []

    for event_id, event in enumerate(events, start=1):
        before_state = deepcopy(state)
        before_query_value = before_state["object_states"][question_object]
        state, description, details = step(state, event)
        after_state = deepcopy(state)
        after_query_value = after_state["object_states"][question_object]
        affects_question = before_query_value != after_query_value
        if affects_question:
            evidence_event_ids.append(event_id)

        event_record = {
            "event_id": event_id,
            "time_before": event_id - 1,
            "time_after": event_id,
            "action": event["action"],
            "agent": event["agent"],
            "automatic_pickups": details["automatic_pickups"],
            "affects_question": affects_question,
            "queried_object_before": before_query_value,
            "queried_object_after": after_query_value,
            "description": description,
        }
        if event["action"] == "move":
            event_record["destination"] = event["destination"]
        else:
            event_record["object"] = event["object"]
            event_record["drop_location"] = details["drop_location"]
        event_records.append(event_record)
        state_trace.append(
            {"time": event_id, "event_id": event_id, "state": after_state}
        )

    answer = who_is_carrying(state, question_object)
    return {
        "schema_version": "1.0",
        "world_version": "week2-v1",
        "story_id": story_id,
        "seed": seed,
        "entities": {
            "agents": sorted(AGENTS),
            "objects": sorted(OBJECTS),
            "locations": sorted(LOCATIONS),
        },
        "initial_state": initial_state,
        "events": event_records,
        "question": {
            "type": "object_carrier",
            "object": question_object,
            "text": f"Who is carrying the {question_object.lower()}?",
            "possible_answers": POSSIBLE_ANSWERS,
        },
        "answer": answer,
        "answer_details": {
            "carrier": answer,
            "physical_location": physical_location(state, question_object),
        },
        "metrics": {
            "story_length": len(events),
            "reasoning_depth": len(evidence_event_ids),
        },
        "evidence_event_ids": evidence_event_ids,
        "final_state": deepcopy(state),
        "state_trace": state_trace,
    }


def build_week2_dataset() -> list[dict[str, Any]]:
    """Build all hand-designed Week 2 coverage stories."""
    return [
        build_dataset_record(
            story_id=f"story_{story['number']:04d}",
            seed=story["seed"],
            events=story["events"],
            question_object=story["question_object"],
            object_locations=story.get("object_locations"),
        )
        for story in STORIES
    ]


def main() -> None:
    dataset = build_week2_dataset()
    output_path = Path(__file__).resolve().with_name("dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(dataset)} stories to {output_path}")
    for record in dataset:
        print(
            f"{record['story_id']}: {record['answer']} "
            f"(length={record['metrics']['story_length']}, "
            f"depth={record['metrics']['reasoning_depth']}, "
            f"evidence={record['evidence_event_ids']})"
        )


if __name__ == "__main__":
    main()
