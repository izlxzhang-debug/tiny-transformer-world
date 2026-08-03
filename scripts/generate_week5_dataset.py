#!/usr/bin/env python3
"""Generate the complete Week 5 train/validation/test dataset.

The structured simulator is the source of truth.  English stories and their
answers are derived from simulator states, and every accepted example is
replayed before the output files are committed to disk.

The script uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


AGENTS = ("Lammy", "Anneena", "Jade", "Penguin")
OBJECTS = ("Hairbrush", "Sneakers", "Glasses", "Key")
LOCATIONS = ("Mars", "Mercury", "Venus", "Moon")
ANSWERS = (*AGENTS, "Nobody")

WORLD_VERSION = "1.0"
GENERATOR_VERSION = "1.0"
MIN_EVENTS = 1
MAX_EVENTS = 6
DROP_PROBABILITY = 0.30
BASE_SEEDS = {"train": 100, "validation": 200, "test": 300}

State = dict[str, dict[str, str]]
Event = dict[str, Any]
Story = dict[str, Any]


class InvalidStateError(ValueError):
    """The stored world is not a legal world state."""


class InvalidActionError(ValueError):
    """An event cannot legally occur in the current state."""


def validate_state(state: State) -> None:
    if set(state.get("agent_locations", {})) != set(AGENTS):
        raise InvalidStateError("Every agent must occur exactly once.")
    if set(state.get("object_states", {})) != set(OBJECTS):
        raise InvalidStateError("Every object must occur exactly once.")
    if any(location not in LOCATIONS for location in state["agent_locations"].values()):
        raise InvalidStateError("Every agent must be at a known location.")
    legal_object_values = set(AGENTS) | set(LOCATIONS)
    if any(value not in legal_object_values for value in state["object_states"].values()):
        raise InvalidStateError("Every object must be loose or carried by one agent.")


def random_initial_state(rng: random.Random) -> State:
    state = {
        "agent_locations": {agent: rng.choice(LOCATIONS) for agent in AGENTS},
        "object_states": {obj: rng.choice(LOCATIONS) for obj in OBJECTS},
    }
    validate_state(state)
    return state


def carrier_of(state: State, obj: str) -> str:
    if obj not in OBJECTS:
        raise InvalidActionError(f"Unknown object: {obj}")
    value = state["object_states"][obj]
    return value if value in AGENTS else "Nobody"


def effective_location(state: State, obj: str) -> str:
    if obj not in OBJECTS:
        raise InvalidActionError(f"Unknown object: {obj}")
    value = state["object_states"][obj]
    return state["agent_locations"][value] if value in AGENTS else value


def move_in_place(state: State, agent: str, destination: str) -> Event:
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if destination not in LOCATIONS:
        raise InvalidActionError(f"Unknown location: {destination}")
    if state["agent_locations"][agent] == destination:
        raise InvalidActionError(f"{agent} is already on {destination}")

    state["agent_locations"][agent] = destination
    pickups = [obj for obj in OBJECTS if state["object_states"][obj] == destination]
    for obj in pickups:
        state["object_states"][obj] = agent
    validate_state(state)
    return {
        "action": "move",
        "agent": agent,
        "destination": destination,
        "automatic_pickups": pickups,
    }


def drop_in_place(state: State, agent: str, obj: str) -> Event:
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if obj not in OBJECTS:
        raise InvalidActionError(f"Unknown object: {obj}")
    if state["object_states"][obj] != agent:
        raise InvalidActionError(f"{agent} is not carrying {obj}")

    location = state["agent_locations"][agent]
    state["object_states"][obj] = location
    # A drop is not an arrival, so it deliberately triggers no pickup.
    validate_state(state)
    return {
        "action": "drop",
        "agent": agent,
        "object": obj,
        "drop_location": location,
        "automatic_pickups": [],
    }


def replay_event_in_place(state: State, event: Event) -> Event:
    action = event.get("action")
    if action == "move":
        return move_in_place(state, event["agent"], event["destination"])
    if action == "drop":
        return drop_in_place(state, event["agent"], event["object"])
    raise InvalidActionError(f"Unsupported action: {action}")


def random_valid_event(state: State, rng: random.Random) -> Event:
    valid_drops = [
        (holder, obj)
        for obj, holder in state["object_states"].items()
        if holder in AGENTS
    ]
    if valid_drops and rng.random() < DROP_PROBABILITY:
        agent, obj = rng.choice(valid_drops)
        return drop_in_place(state, agent, obj)

    agent = rng.choice(AGENTS)
    current = state["agent_locations"][agent]
    destination = rng.choice([location for location in LOCATIONS if location != current])
    return move_in_place(state, agent, destination)


def english_object_list(objects: list[str]) -> str:
    names = [f"the {obj.lower()}" for obj in objects]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def render_event(event: Event) -> str:
    if event["action"] == "drop":
        return (
            f"{event['agent']} drops the {event['object'].lower()} "
            f"on {event['drop_location']}."
        )
    prefix = f"{event['agent']} travels to {event['destination']}"
    pickups = event["automatic_pickups"]
    if not pickups:
        return prefix + "."
    return f"{prefix} and automatically picks up {english_object_list(pickups)}."


def render_story(initial_state: State, events: list[Event], question_object: str) -> str:
    lines = ["Initial state:"]
    lines.extend(
        f"{agent} is on {initial_state['agent_locations'][agent]}." for agent in AGENTS
    )
    lines.extend(
        f"The {obj.lower()} is lying on {initial_state['object_states'][obj]}."
        for obj in OBJECTS
    )
    lines.extend(("", "Events:"))
    lines.extend(f"{event['event_id']}. {render_event(event)}" for event in events)
    lines.extend(("", f"Question: Who is carrying the {question_object.lower()}?"))
    return "\n".join(lines)


def evidence_event_ids(initial_state: State, events: list[Event], obj: str) -> list[int]:
    state = deepcopy(initial_state)
    evidence: list[int] = []
    for event in events:
        before = state["object_states"][obj]
        replay_event_in_place(state, event)
        if state["object_states"][obj] != before:
            evidence.append(event["event_id"])
    return evidence


def generate_candidate(split: str, story_seed: int) -> Story:
    rng = random.Random(story_seed)
    initial_state = random_initial_state(rng)
    state = deepcopy(initial_state)
    events: list[Event] = []
    for event_id in range(1, rng.randint(MIN_EVENTS, MAX_EVENTS) + 1):
        event = random_valid_event(state, rng)
        event["event_id"] = event_id
        events.append(event)

    question_object = rng.choice(OBJECTS)
    answer = carrier_of(state, question_object)
    evidence = evidence_event_ids(initial_state, events, question_object)
    return {
        "story_id": None,
        "split": split,
        "seed": story_seed,
        "world_version": WORLD_VERSION,
        "generator_version": GENERATOR_VERSION,
        "initial_state": initial_state,
        "events": events,
        "question": {
            "type": "object_carrier",
            "object": question_object,
            "text": f"Who is carrying the {question_object.lower()}?",
            "possible_answers": list(ANSWERS),
        },
        "answer": answer,
        "answer_details": {
            "carrier": answer,
            "effective_location": effective_location(state, question_object),
        },
        "metrics": {
            "story_length": len(events),
            "reasoning_depth": len(evidence),
        },
        "evidence_event_ids": evidence,
        "final_state": deepcopy(state),
        "story_text": render_story(initial_state, events, question_object),
    }


def structural_event(event: Event) -> dict[str, str]:
    if event["action"] == "move":
        return {
            "action": "move",
            "agent": event["agent"],
            "destination": event["destination"],
        }
    return {"action": "drop", "agent": event["agent"], "object": event["object"]}


def fingerprint(story: Story) -> str:
    structure = {
        "initial_state": story["initial_state"],
        "events": [structural_event(event) for event in story["events"]],
        "question_object": story["question"]["object"],
    }
    encoded = json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_story(story: Story) -> None:
    validate_state(story["initial_state"])
    state = deepcopy(story["initial_state"])
    for stored in story["events"]:
        replayed = replay_event_in_place(state, stored)
        if replayed["automatic_pickups"] != stored["automatic_pickups"]:
            raise AssertionError(f"{story['story_id']}: automatic-pickup mismatch")
        if stored["action"] == "drop" and replayed["drop_location"] != stored["drop_location"]:
            raise AssertionError(f"{story['story_id']}: drop-location mismatch")
    if state != story["final_state"]:
        raise AssertionError(f"{story['story_id']}: final-state mismatch")

    obj = story["question"]["object"]
    if carrier_of(state, obj) != story["answer"]:
        raise AssertionError(f"{story['story_id']}: answer mismatch")
    if effective_location(state, obj) != story["answer_details"]["effective_location"]:
        raise AssertionError(f"{story['story_id']}: effective-location mismatch")
    if evidence_event_ids(story["initial_state"], story["events"], obj) != story["evidence_event_ids"]:
        raise AssertionError(f"{story['story_id']}: evidence mismatch")
    if render_story(story["initial_state"], story["events"], obj) != story["story_text"]:
        raise AssertionError(f"{story['story_id']}: rendered-story mismatch")


def generate_split(
    split: str,
    size: int,
    seed: int,
    used_fingerprints: set[str],
) -> tuple[list[Story], int]:
    if size <= 0 or size % (len(OBJECTS) * len(ANSWERS)):
        raise ValueError(f"{split} size must be a positive multiple of 20")

    target_per_pair = size // (len(OBJECTS) * len(ANSWERS))
    pair_counts: Counter[tuple[str, str]] = Counter()
    records: list[Story] = []
    rng = random.Random(seed)
    attempts = 0
    max_attempts = max(100_000, size * 1_000)

    while len(records) < size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not complete {split} within {max_attempts:,} attempts")
        candidate = generate_candidate(split, rng.randrange(1, 2**63))
        pair = (candidate["question"]["object"], candidate["answer"])
        if pair_counts[pair] >= target_per_pair:
            continue
        story_fingerprint = fingerprint(candidate)
        if story_fingerprint in used_fingerprints:
            continue

        candidate["story_id"] = f"{split}_{len(records) + 1:06d}"
        validate_story(candidate)
        records.append(candidate)
        pair_counts[pair] += 1
        used_fingerprints.add(story_fingerprint)
        if len(records) % 1_000 == 0:
            print(f"  {split}: {len(records):,}/{size:,}")
    return records, attempts


def split_statistics(records: list[Story], attempts: int) -> dict[str, Any]:
    answer_counts = Counter(story["answer"] for story in records)
    object_counts = Counter(story["question"]["object"] for story in records)
    pair_counts = Counter(
        (story["question"]["object"], story["answer"]) for story in records
    )
    length_counts = Counter(story["metrics"]["story_length"] for story in records)
    actions = Counter(event["action"] for story in records for event in story["events"])
    locations = Counter(
        event["destination"]
        for story in records
        for event in story["events"]
        if event["action"] == "move"
    )
    return {
        "stories": len(records),
        "candidate_attempts": attempts,
        "rule_based_accuracy": 1.0,
        "invalid_actions": 0,
        "answer_counts": {answer: answer_counts[answer] for answer in ANSWERS},
        "question_object_counts": {obj: object_counts[obj] for obj in OBJECTS},
        "object_answer_pair_counts": {
            f"{obj}|{answer}": pair_counts[(obj, answer)]
            for obj in OBJECTS
            for answer in ANSWERS
        },
        "story_length_counts": {
            str(length): length_counts[length] for length in range(MIN_EVENTS, MAX_EVENTS + 1)
        },
        "action_counts": dict(sorted(actions.items())),
        "movement_destination_counts": {
            location: locations[location] for location in LOCATIONS
        },
    }


def atomic_save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def validate_complete_dataset(splits: dict[str, list[Story]]) -> None:
    seen: set[str] = set()
    for split, records in splits.items():
        answer_counts = Counter()
        object_counts = Counter()
        pair_counts = Counter()
        target = len(records) // 20
        for story in records:
            validate_story(story)
            story_fingerprint = fingerprint(story)
            if story_fingerprint in seen:
                raise AssertionError(f"Duplicate structured story: {story['story_id']}")
            seen.add(story_fingerprint)
            answer_counts[story["answer"]] += 1
            object_counts[story["question"]["object"]] += 1
            pair_counts[(story["question"]["object"], story["answer"])] += 1
        if any(pair_counts[(obj, answer)] != target for obj in OBJECTS for answer in ANSWERS):
            raise AssertionError(f"{split} is not jointly balanced")
        if any(answer_counts[answer] != len(records) // 5 for answer in ANSWERS):
            raise AssertionError(f"{split} answer classes are not balanced")
        if any(object_counts[obj] != len(records) // 4 for obj in OBJECTS):
            raise AssertionError(f"{split} question objects are not balanced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/week5"))
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to each documented split seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    if any(size <= 0 or size % 20 for size in sizes.values()):
        raise SystemExit("All split sizes must be positive multiples of 20.")

    used_fingerprints: set[str] = set()
    splits: dict[str, list[Story]] = {}
    attempts: dict[str, int] = {}
    print("Generating Week 5 dataset...")
    for split in ("train", "validation", "test"):
        print(f"Generating {split}: {sizes[split]:,} stories")
        records, split_attempts = generate_split(
            split,
            sizes[split],
            BASE_SEEDS[split] + args.seed_offset,
            used_fingerprints,
        )
        splits[split] = records
        attempts[split] = split_attempts

    validate_complete_dataset(splits)
    for split, records in splits.items():
        atomic_save_json(records, args.output_dir / f"{split}.json")

    statistics = {
        "world_version": WORLD_VERSION,
        "generator_version": GENERATOR_VERSION,
        "configuration": {
            "minimum_events": MIN_EVENTS,
            "maximum_events": MAX_EVENTS,
            "drop_probability": DROP_PROBABILITY,
            "seed_offset": args.seed_offset,
            "split_sizes": sizes,
            "split_seeds": {
                split: BASE_SEEDS[split] + args.seed_offset for split in sizes
            },
        },
        "validation": {
            "rule_based_accuracy": 1.0,
            "invalid_actions": 0,
            "cross_split_duplicate_structures": 0,
            "all_splits_jointly_balanced": True,
        },
        "splits": {
            split: split_statistics(records, attempts[split])
            for split, records in splits.items()
        },
    }
    atomic_save_json(statistics, args.output_dir / "dataset_statistics.json")

    print(f"Saved validated dataset to {args.output_dir.resolve()}")
    print(f"Stories: {sum(sizes.values()):,}; rule-based accuracy: 100%; invalid actions: 0")


if __name__ == "__main__":
    main()
