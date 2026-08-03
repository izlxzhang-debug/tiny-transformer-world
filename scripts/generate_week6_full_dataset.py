#!/usr/bin/env python3
"""Generate and validate the complete Week 6 synthetic world dataset.

The file is deliberately self-contained and uses only Python's standard library.
By default it creates 30,000 examples:

* 20,000 training examples
* 2,000 validation examples
* 2,000 standard test examples
* 2,000 long-story test examples
* 2,000 paraphrase test examples
* 2,000 withheld-combination test examples

Each split is exactly balanced over the Cartesian product of four question
objects and five answers. Four automatic-pickup triples are absent from all
ordinary splits and present in the withheld test split. Every saved example is
replayed by the rule-based simulator before generation is declared successful.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Iterator


AGENTS = ("Lammy", "Anneena", "Jade", "Penguin")
OBJECTS = ("Hairbrush", "Sneakers", "Glasses", "Key")
LOCATIONS = ("Mars", "Mercury", "Venus", "Moon")
ANSWERS = (*AGENTS, "Nobody")

WORLD_VERSION = "1.0"
GENERATOR_VERSION = "3.0"
DEFAULT_DROP_PROBABILITY = 0.30
PAIR_COUNT = len(OBJECTS) * len(ANSWERS)

# (arriving agent, automatically picked-up object, destination)
WITHHELD_COMBINATIONS = frozenset(
    {
        ("Lammy", "Key", "Mars"),
        ("Anneena", "Glasses", "Mercury"),
        ("Jade", "Sneakers", "Venus"),
        ("Penguin", "Hairbrush", "Moon"),
    }
)

PARAPHRASE_AGENT_TEMPLATES = (
    "{agent} is located on {location}.",
    "{agent} starts out on {location}.",
    "At first, {agent} is on {location}.",
)
PARAPHRASE_OBJECT_TEMPLATES = (
    "The {object_name} can be found on {location}.",
    "The {object_name} is resting on {location}.",
    "At first, the {object_name} lies on {location}.",
)
PARAPHRASE_MOVE_TEMPLATES = (
    "{agent} goes to {destination}",
    "{agent} heads to {destination}",
    "{agent} makes the trip to {destination}",
)
PARAPHRASE_DROP_TEMPLATES = (
    "{agent} leaves the {object_name} on {location}.",
    "{agent} sets down the {object_name} on {location}.",
    "{agent} puts the {object_name} down on {location}.",
)


class InvalidActionError(ValueError):
    """Raised when a proposed action violates a physical rule."""


class InvalidStateError(ValueError):
    """Raised when a stored state violates the world specification."""


def validate_state(state: dict[str, Any]) -> None:
    """Validate the complete world state."""
    if not isinstance(state, dict):
        raise InvalidStateError("State must be a dictionary.")
    if set(state.get("agent_locations", {})) != set(AGENTS):
        raise InvalidStateError("Every agent must appear exactly once.")
    if set(state.get("object_states", {})) != set(OBJECTS):
        raise InvalidStateError("Every object must appear exactly once.")

    for agent, location in state["agent_locations"].items():
        if location not in LOCATIONS:
            raise InvalidStateError(f"{agent} has invalid location {location!r}.")

    valid_object_values = set(AGENTS) | set(LOCATIONS)
    for object_name, value in state["object_states"].items():
        if value not in valid_object_values:
            raise InvalidStateError(f"{object_name} has invalid state {value!r}.")


def make_random_initial_state(rng: random.Random) -> dict[str, Any]:
    """Create a valid state in which all objects initially lie loose."""
    state = {
        "agent_locations": {
            agent: rng.choice(LOCATIONS) for agent in AGENTS
        },
        "object_states": {
            object_name: rng.choice(LOCATIONS) for object_name in OBJECTS
        },
    }
    validate_state(state)
    return state


def who_is_carrying(state: dict[str, Any], object_name: str) -> str:
    """Return the object's carrier, or ``Nobody`` when it is loose."""
    if object_name not in OBJECTS:
        raise ValueError(f"Unknown object: {object_name}")
    holder_or_location = state["object_states"][object_name]
    return holder_or_location if holder_or_location in AGENTS else "Nobody"


def effective_location(state: dict[str, Any], object_name: str) -> str:
    """Return the physical location of a loose or carried object."""
    if object_name not in OBJECTS:
        raise ValueError(f"Unknown object: {object_name}")
    holder_or_location = state["object_states"][object_name]
    if holder_or_location in AGENTS:
        return state["agent_locations"][holder_or_location]
    return holder_or_location


def pick_up_in_place(
    state: dict[str, Any], agent: str, object_name: str
) -> None:
    """Apply the internal pickup effect after checking co-location."""
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if object_name not in OBJECTS:
        raise InvalidActionError(f"Unknown object: {object_name}")

    holder_or_location = state["object_states"][object_name]
    if holder_or_location in AGENTS:
        raise InvalidActionError(f"{object_name} is already being carried.")
    if holder_or_location != state["agent_locations"][agent]:
        raise InvalidActionError(f"{agent} and {object_name} are not co-located.")
    state["object_states"][object_name] = agent


def apply_move_in_place(
    state: dict[str, Any], agent: str, destination: str
) -> dict[str, Any]:
    """Move one agent and atomically apply every automatic pickup."""
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if destination not in LOCATIONS:
        raise InvalidActionError(f"Unknown destination: {destination}")
    if state["agent_locations"][agent] == destination:
        raise InvalidActionError(f"{agent} is already on {destination}.")

    state["agent_locations"][agent] = destination
    pickups = [
        object_name
        for object_name in OBJECTS
        if state["object_states"][object_name] == destination
    ]
    for object_name in pickups:
        pick_up_in_place(state, agent, object_name)
    validate_state(state)
    return {
        "action": "move",
        "agent": agent,
        "destination": destination,
        "automatic_pickups": pickups,
    }


def apply_drop_in_place(
    state: dict[str, Any], agent: str, object_name: str
) -> dict[str, Any]:
    """Drop an object at its carrier's location without repicking it up."""
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if object_name not in OBJECTS:
        raise InvalidActionError(f"Unknown object: {object_name}")
    if state["object_states"][object_name] != agent:
        raise InvalidActionError(
            f"{agent} cannot drop {object_name}; {agent} is not carrying it."
        )

    drop_location = state["agent_locations"][agent]
    state["object_states"][object_name] = drop_location
    validate_state(state)
    return {
        "action": "drop",
        "agent": agent,
        "object": object_name,
        "drop_location": drop_location,
        "automatic_pickups": [],
    }


def replay_event_in_place(
    state: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    """Replay one stored event through the same formal transition rules."""
    if event.get("action") == "move":
        return apply_move_in_place(state, event["agent"], event["destination"])
    if event.get("action") == "drop":
        return apply_drop_in_place(state, event["agent"], event["object"])
    raise InvalidActionError(f"Unknown action: {event.get('action')!r}")


def valid_drop_choices(state: dict[str, Any]) -> list[tuple[str, str]]:
    """List all currently legal (agent, object) drops."""
    return [
        (holder, object_name)
        for object_name, holder in state["object_states"].items()
        if holder in AGENTS
    ]


def generate_random_event(
    state: dict[str, Any], rng: random.Random, drop_probability: float
) -> dict[str, Any]:
    """Generate and immediately apply one physically valid event."""
    drops = valid_drop_choices(state)
    if drops and rng.random() < drop_probability:
        agent, object_name = rng.choice(drops)
        return apply_drop_in_place(state, agent, object_name)

    agent = rng.choice(AGENTS)
    current = state["agent_locations"][agent]
    destination = rng.choice([location for location in LOCATIONS if location != current])
    return apply_move_in_place(state, agent, destination)


def format_object_list(object_names: list[str]) -> str:
    """Format object names as a grammatical English list."""
    names = [f"the {name.lower()}" for name in object_names]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def render_story(
    initial_state: dict[str, Any],
    events: list[dict[str, Any]],
    question_object: str,
    language_style: str,
    template_seed: int,
) -> str:
    """Render structured data into deterministic controlled English."""
    rng = random.Random(template_seed)
    lines = ["Initial state:"]

    for agent in AGENTS:
        location = initial_state["agent_locations"][agent]
        if language_style == "canonical":
            lines.append(f"{agent} is on {location}.")
        else:
            lines.append(
                rng.choice(PARAPHRASE_AGENT_TEMPLATES).format(
                    agent=agent, location=location
                )
            )

    for object_name in OBJECTS:
        location = initial_state["object_states"][object_name]
        if language_style == "canonical":
            lines.append(f"The {object_name.lower()} is lying on {location}.")
        else:
            lines.append(
                rng.choice(PARAPHRASE_OBJECT_TEMPLATES).format(
                    object_name=object_name.lower(), location=location
                )
            )

    lines.extend(("", "Events:"))
    for event in events:
        if event["action"] == "drop":
            if language_style == "canonical":
                sentence = (
                    f"{event['agent']} drops the {event['object'].lower()} "
                    f"on {event['drop_location']}."
                )
            else:
                sentence = rng.choice(PARAPHRASE_DROP_TEMPLATES).format(
                    agent=event["agent"],
                    object_name=event["object"].lower(),
                    location=event["drop_location"],
                )
        else:
            if language_style == "canonical":
                move_text = f"{event['agent']} travels to {event['destination']}"
            else:
                move_text = rng.choice(PARAPHRASE_MOVE_TEMPLATES).format(
                    agent=event["agent"], destination=event["destination"]
                )
            pickups = event["automatic_pickups"]
            if not pickups:
                sentence = move_text + "."
            elif language_style == "canonical":
                sentence = (
                    f"{move_text} and automatically picks up "
                    f"{format_object_list(pickups)}."
                )
            else:
                sentence = (
                    f"{move_text}, where {event['agent']} collects "
                    f"{format_object_list(pickups)}."
                )
        lines.append(f"{event['event_id']}. {sentence}")

    lines.extend(
        ("", f"Question: Who is carrying the {question_object.lower()}?")
    )
    return "\n".join(lines)


def calculate_evidence_event_ids(
    initial_state: dict[str, Any],
    events: list[dict[str, Any]],
    question_object: str,
) -> list[int]:
    """Identify events that change the queried object's carrier state."""
    state = deepcopy(initial_state)
    evidence: list[int] = []
    for event in events:
        before = state["object_states"][question_object]
        replay_event_in_place(state, event)
        after = state["object_states"][question_object]
        if before != after:
            evidence.append(event["event_id"])
    return evidence


def pickup_combinations(events: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Extract all (agent, object, destination) automatic pickups."""
    return {
        (event["agent"], object_name, event["destination"])
        for event in events
        if event["action"] == "move"
        for object_name in event["automatic_pickups"]
    }


def generate_candidate(
    spec: dict[str, Any], story_seed: int, drop_probability: float
) -> dict[str, Any]:
    """Generate one fully structured candidate example."""
    rng = random.Random(story_seed)
    initial_state = make_random_initial_state(rng)
    state = deepcopy(initial_state)
    event_count = rng.randint(spec["minimum_events"], spec["maximum_events"])
    events: list[dict[str, Any]] = []
    for event_id in range(1, event_count + 1):
        event = generate_random_event(state, rng, drop_probability)
        event["event_id"] = event_id
        events.append(event)

    question_object = rng.choice(OBJECTS)
    answer = who_is_carrying(state, question_object)
    evidence = calculate_evidence_event_ids(initial_state, events, question_object)
    template_seed = rng.randrange(1, 2**63)
    return {
        "story_id": None,
        "split": spec["name"],
        "seed": story_seed,
        "template_seed": template_seed,
        "world_version": WORLD_VERSION,
        "generator_version": GENERATOR_VERSION,
        "language_style": spec["language_style"],
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
        "story_text": render_story(
            initial_state,
            events,
            question_object,
            spec["language_style"],
            template_seed,
        ),
    }


def structural_event(event: dict[str, Any]) -> dict[str, str]:
    """Return only information that defines an event's world transition."""
    if event["action"] == "move":
        return {
            "action": "move",
            "agent": event["agent"],
            "destination": event["destination"],
        }
    return {
        "action": "drop",
        "agent": event["agent"],
        "object": event["object"],
    }


def story_fingerprint(story: dict[str, Any]) -> str:
    """Hash language-independent structure for leakage detection."""
    structure = {
        "initial_state": story["initial_state"],
        "events": [structural_event(event) for event in story["events"]],
        "question_object": story["question"]["object"],
    }
    payload = json.dumps(structure, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_split(
    spec: dict[str, Any],
    used_fingerprints: set[str],
    drop_probability: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], int]:
    """Generate an exactly answer/object-balanced, leakage-free split."""
    if spec["size"] % PAIR_COUNT:
        raise ValueError(f"{spec['name']} size must be divisible by {PAIR_COUNT}.")
    target_per_pair = spec["size"] // PAIR_COUNT
    pair_counts: Counter[tuple[str, str]] = Counter()
    records: list[dict[str, Any]] = []
    rng = random.Random(spec["seed"])
    attempts = 0
    maximum_attempts = max(200_000, spec["size"] * 1_500)
    missing_withheld = set(WITHHELD_COMBINATIONS) if spec["require_withheld"] else set()

    while len(records) < spec["size"]:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError(
                f"Unable to finish {spec['name']} within {maximum_attempts:,} attempts."
            )
        candidate = generate_candidate(
            spec, rng.randrange(1, 2**63), drop_probability
        )
        combinations = pickup_combinations(candidate["events"])
        withheld_here = combinations & WITHHELD_COMBINATIONS

        if spec["exclude_withheld"] and withheld_here:
            continue
        if spec["require_withheld"]:
            if not withheld_here:
                continue
            # Guarantee coverage even in small CLI smoke-test splits.
            if missing_withheld and not (withheld_here & missing_withheld):
                continue

        pair = (candidate["question"]["object"], candidate["answer"])
        if pair_counts[pair] >= target_per_pair:
            continue
        fingerprint = story_fingerprint(candidate)
        if fingerprint in used_fingerprints:
            continue

        candidate["story_id"] = f"{spec['name']}_{len(records) + 1:06d}"
        records.append(candidate)
        pair_counts[pair] += 1
        used_fingerprints.add(fingerprint)
        missing_withheld.difference_update(withheld_here)
        if progress_every and len(records) % progress_every == 0:
            print(f"  {spec['name']}: {len(records):,}/{spec['size']:,}")

    if missing_withheld:
        raise AssertionError(
            f"{spec['name']} is missing withheld combinations: {missing_withheld}"
        )
    return records, attempts


def validate_story(story: dict[str, Any]) -> None:
    """Replay one story and compare every derived stored field."""
    state = deepcopy(story["initial_state"])
    validate_state(state)
    for stored_event in story["events"]:
        replayed = replay_event_in_place(state, stored_event)
        if replayed["automatic_pickups"] != stored_event["automatic_pickups"]:
            raise AssertionError(f"{story['story_id']}: pickup disagreement.")
        if stored_event["action"] == "drop" and (
            replayed["drop_location"] != stored_event["drop_location"]
        ):
            raise AssertionError(f"{story['story_id']}: drop-location disagreement.")

    if state != story["final_state"]:
        raise AssertionError(f"{story['story_id']}: final-state disagreement.")
    question_object = story["question"]["object"]
    if who_is_carrying(state, question_object) != story["answer"]:
        raise AssertionError(f"{story['story_id']}: answer disagreement.")
    if effective_location(state, question_object) != story["answer_details"][
        "effective_location"
    ]:
        raise AssertionError(f"{story['story_id']}: effective-location disagreement.")
    evidence = calculate_evidence_event_ids(
        story["initial_state"], story["events"], question_object
    )
    if evidence != story["evidence_event_ids"]:
        raise AssertionError(f"{story['story_id']}: evidence disagreement.")
    expected_text = render_story(
        story["initial_state"],
        story["events"],
        question_object,
        story["language_style"],
        story["template_seed"],
    )
    if expected_text != story["story_text"]:
        raise AssertionError(f"{story['story_id']}: rendered-text disagreement.")


def format_combination(combination: tuple[str, str, str]) -> str:
    return "|".join(combination)


def validate_split(records: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Validate correctness, action legality, balance, and withholding policy."""
    if len(records) != spec["size"]:
        raise AssertionError(f"{spec['name']} has the wrong number of records.")
    answers: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    withheld: Counter[tuple[str, str, str]] = Counter()
    correct = 0
    invalid_actions = 0

    for story in records:
        try:
            validate_story(story)
        except (InvalidActionError, InvalidStateError):
            invalid_actions += 1
            raise
        correct += 1
        answer = story["answer"]
        object_name = story["question"]["object"]
        answers[answer] += 1
        objects[object_name] += 1
        pairs[(object_name, answer)] += 1
        withheld.update(pickup_combinations(story["events"]) & WITHHELD_COMBINATIONS)

    expected_pair = spec["size"] // PAIR_COUNT
    for object_name in OBJECTS:
        for answer in ANSWERS:
            if pairs[(object_name, answer)] != expected_pair:
                raise AssertionError(
                    f"{spec['name']} is not balanced for {(object_name, answer)}."
                )
    if spec["exclude_withheld"] and sum(withheld.values()):
        raise AssertionError(f"{spec['name']} contains a withheld pickup.")
    if spec["require_withheld"]:
        missing = [item for item in WITHHELD_COMBINATIONS if withheld[item] == 0]
        if missing:
            raise AssertionError(f"{spec['name']} lacks withheld pickups: {missing}")

    return {
        "number_of_stories": len(records),
        "rule_based_correct": correct,
        "rule_based_accuracy": correct / len(records),
        "invalid_action_count": invalid_actions,
        "answer_classes_balanced": True,
        "question_objects_balanced": True,
        "answer_counts": {answer: answers[answer] for answer in ANSWERS},
        "question_object_counts": {
            object_name: objects[object_name] for object_name in OBJECTS
        },
        "object_answer_pair_count": expected_pair,
        "withheld_combination_counts": {
            format_combination(item): withheld[item]
            for item in sorted(WITHHELD_COMBINATIONS)
        },
    }


def calculate_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate useful generation and coverage diagnostics."""
    lengths: Counter[int] = Counter()
    depths: Counter[int] = Counter()
    actions: Counter[str] = Counter()
    acting_agents: Counter[str] = Counter()
    destinations: Counter[str] = Counter()
    initial_agent_locations: Counter[str] = Counter()
    initial_object_locations: Counter[str] = Counter()
    final_question_locations: Counter[str] = Counter()
    for story in records:
        lengths[story["metrics"]["story_length"]] += 1
        depths[story["metrics"]["reasoning_depth"]] += 1
        initial_agent_locations.update(story["initial_state"]["agent_locations"].values())
        initial_object_locations.update(story["initial_state"]["object_states"].values())
        final_question_locations[story["answer_details"]["effective_location"]] += 1
        for event in story["events"]:
            actions[event["action"]] += 1
            acting_agents[event["agent"]] += 1
            if event["action"] == "move":
                destinations[event["destination"]] += 1
    return {
        "story_length_counts": {str(k): v for k, v in sorted(lengths.items())},
        "reasoning_depth_counts": {str(k): v for k, v in sorted(depths.items())},
        "action_counts": dict(sorted(actions.items())),
        "acting_agent_counts": {agent: acting_agents[agent] for agent in AGENTS},
        "movement_destination_counts": {
            location: destinations[location] for location in LOCATIONS
        },
        "initial_agent_location_counts": {
            location: initial_agent_locations[location] for location in LOCATIONS
        },
        "initial_object_location_counts": {
            location: initial_object_locations[location] for location in LOCATIONS
        },
        "final_question_location_counts": {
            location: final_question_locations[location] for location in LOCATIONS
        },
    }


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_saved_files(paths: list[Path]) -> dict[str, Any]:
    """Reload JSONL and recheck correctness plus cross-split uniqueness."""
    fingerprints: set[str] = set()
    total = 0
    correct = 0
    invalid_actions = 0
    for path in paths:
        for story in load_jsonl(path):
            total += 1
            fingerprint = story_fingerprint(story)
            if fingerprint in fingerprints:
                raise AssertionError(f"Duplicate structured story found in {path}.")
            fingerprints.add(fingerprint)
            try:
                validate_story(story)
            except (InvalidActionError, InvalidStateError):
                invalid_actions += 1
                raise
            correct += 1
    return {
        "total_stories": total,
        "rule_based_correct": correct,
        "rule_based_accuracy": correct / total,
        "invalid_action_count": invalid_actions,
        "cross_split_duplicate_count": 0,
    }


def select_manual_examples(
    pool: list[dict[str, Any]], number: int, seed: int
) -> list[dict[str, Any]]:
    """Select an answer-balanced review sample; this does not inspect it."""
    if number % len(ANSWERS):
        raise ValueError(f"Manual inspection size must be divisible by {len(ANSWERS)}.")
    by_answer: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for story in pool:
        by_answer[story["answer"]].append(story)
    per_answer = number // len(ANSWERS)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for answer in ANSWERS:
        if len(by_answer[answer]) < per_answer:
            raise RuntimeError(f"Not enough manual-review candidates for {answer}.")
        selected.extend(rng.sample(by_answer[answer], per_answer))
    rng.shuffle(selected)
    return selected


def save_manual_inspection_files(examples: list[dict[str, Any]], output_dir: Path) -> None:
    """Write full JSON records and a human-fillable CSV review sheet."""
    save_json(examples, output_dir / "manual_inspection_100.json")
    with (output_dir / "manual_inspection_100.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "story_id",
                "answer",
                "question_object",
                "status",
                "notes",
                "story_text",
            ),
        )
        writer.writeheader()
        for story in examples:
            writer.writerow(
                {
                    "split": story["split"],
                    "story_id": story["story_id"],
                    "answer": story["answer"],
                    "question_object": story["question"]["object"],
                    "status": "",
                    "notes": "",
                    "story_text": story["story_text"],
                }
            )


def build_specs(train_size: int, split_size: int, seed_offset: int) -> list[dict[str, Any]]:
    """Create production or CLI-overridden split specifications."""
    return [
        {
            "name": "train",
            "size": train_size,
            "seed": 100 + seed_offset,
            "minimum_events": 1,
            "maximum_events": 6,
            "language_style": "canonical",
            "exclude_withheld": True,
            "require_withheld": False,
        },
        {
            "name": "validation",
            "size": split_size,
            "seed": 200 + seed_offset,
            "minimum_events": 1,
            "maximum_events": 6,
            "language_style": "canonical",
            "exclude_withheld": True,
            "require_withheld": False,
        },
        {
            "name": "test_standard",
            "size": split_size,
            "seed": 300 + seed_offset,
            "minimum_events": 1,
            "maximum_events": 6,
            "language_style": "canonical",
            "exclude_withheld": True,
            "require_withheld": False,
        },
        {
            "name": "test_long",
            "size": split_size,
            "seed": 400 + seed_offset,
            "minimum_events": 7,
            "maximum_events": 10,
            "language_style": "canonical",
            "exclude_withheld": True,
            "require_withheld": False,
        },
        {
            "name": "test_paraphrase",
            "size": split_size,
            "seed": 500 + seed_offset,
            "minimum_events": 1,
            "maximum_events": 6,
            "language_style": "paraphrase",
            "exclude_withheld": True,
            "require_withheld": False,
        },
        {
            "name": "test_withheld",
            "size": split_size,
            "seed": 600 + seed_offset,
            "minimum_events": 1,
            "maximum_events": 6,
            "language_style": "canonical",
            "exclude_withheld": False,
            "require_withheld": True,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/week6_full_dataset"),
        help="Directory for JSONL datasets and reports.",
    )
    parser.add_argument(
        "--train-size", type=int, default=20_000, help="Training split size."
    )
    parser.add_argument(
        "--split-size",
        type=int,
        default=2_000,
        help="Size of validation and each test split.",
    )
    parser.add_argument(
        "--manual-size",
        type=int,
        default=100,
        help="Answer-balanced sample exported for human inspection.",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0, help="Offset added to every split seed."
    )
    parser.add_argument(
        "--drop-probability",
        type=float,
        default=DEFAULT_DROP_PROBABILITY,
        help="Probability of choosing a valid drop instead of a move.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print progress after this many accepted examples; 0 disables it.",
    )
    args = parser.parse_args()
    for label, size in (("train", args.train_size), ("split", args.split_size)):
        if size <= 0 or size % PAIR_COUNT:
            parser.error(f"--{label}-size must be positive and divisible by {PAIR_COUNT}")
    if args.manual_size <= 0 or args.manual_size % len(ANSWERS):
        parser.error(f"--manual-size must be positive and divisible by {len(ANSWERS)}")
    total_examples = args.train_size + 5 * args.split_size
    if args.manual_size > total_examples:
        parser.error("--manual-size cannot exceed the total generated example count")
    if not 0.0 <= args.drop_probability <= 1.0:
        parser.error("--drop-probability must lie between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    specs = build_specs(args.train_size, args.split_size, args.seed_offset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_fingerprints: set[str] = set()
    validation_report: dict[str, Any] = {}
    statistics_report: dict[str, Any] = {}
    generation_attempts: dict[str, int] = {}
    output_paths: list[Path] = []
    manual_pool: list[dict[str, Any]] = []

    print("Generating the complete Week 6 dataset...")
    for spec in specs:
        print(f"\nGenerating {spec['name']}: {spec['size']:,} stories")
        records, attempts = generate_split(
            spec, used_fingerprints, args.drop_probability, args.progress_every
        )
        generation_attempts[spec["name"]] = attempts
        print(f"Validating {spec['name']}...")
        validation_report[spec["name"]] = validate_split(records, spec)
        statistics_report[spec["name"]] = calculate_statistics(records)
        output_path = args.output_dir / f"{spec['name']}.jsonl"
        save_jsonl(records, output_path)
        output_paths.append(output_path)
        manual_pool.extend(records)

    print("\nReloading and validating serialized files...")
    serialized_report = validate_saved_files(output_paths)
    train_withheld = sum(
        validation_report["train"]["withheld_combination_counts"].values()
    )
    if train_withheld:
        raise AssertionError("Withheld combinations leaked into training.")

    manual_examples = select_manual_examples(
        manual_pool, args.manual_size, 999 + args.seed_offset
    )
    # Preserve the production filenames requested by the project; overridden
    # smoke runs can choose another manual size and the report records it.
    save_manual_inspection_files(manual_examples, args.output_dir)

    report = {
        "world_version": WORLD_VERSION,
        "generator_version": GENERATOR_VERSION,
        "configuration": {
            "train_size": args.train_size,
            "validation_and_test_split_size": args.split_size,
            "manual_inspection_export_size": args.manual_size,
            "drop_probability": args.drop_probability,
            "seed_offset": args.seed_offset,
        },
        "withheld_combinations": [
            {"agent": agent, "object": object_name, "location": location}
            for agent, object_name, location in sorted(WITHHELD_COMBINATIONS)
        ],
        "validation": validation_report,
        "serialized_validation": serialized_report,
        "statistics": statistics_report,
        "generation_attempts": generation_attempts,
        "manual_inspection": {
            "examples_exported": args.manual_size,
            "inspection_completed": False,
            "instruction": (
                "Open manual_inspection_100.csv, independently solve every story, "
                "enter pass or fail in status, and document discrepancies in notes."
            ),
        },
    }
    save_json(report, args.output_dir / "validation_report.json")

    print("\nGeneration and validation completed successfully.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Total stories: {serialized_report['total_stories']:,}")
    print(f"Rule-based accuracy: {serialized_report['rule_based_accuracy']:.1%}")
    print(f"Invalid actions: {serialized_report['invalid_action_count']}")
    print(f"Cross-split duplicates: {serialized_report['cross_split_duplicate_count']}")
    print(f"Withheld combinations in training: {train_withheld}")
    print(
        f"Exported {args.manual_size} balanced examples for genuine human review; "
        "the review is not marked complete."
    )


if __name__ == "__main__":
    main()
