"""Week 2 stories and JSON dataset-record construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .world import (
    AGENTS,
    LOCATIONS,
    OBJECTS,
    POSSIBLE_ANSWERS,
    Event,
    drop,
    make_initial_state,
    move,
    physical_location,
    step,
    who_is_carrying,
)

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
