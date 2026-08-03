"""Generate 300 balanced stories with a self-contained world simulator.

Everything needed to reproduce and validate the Week 4 dataset is kept in
this one standard-library Python file rather than split across modules.
"""

import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path


AGENTS = ("Lammy", "Anneena", "Jade", "Penguin")
OBJECTS = ("Hairbrush", "Sneakers", "Glasses", "Key")
LOCATIONS = ("Mars", "Mercury", "Venus", "Moon")
ANSWERS = (*AGENTS, "Nobody")

NUMBER_OF_STORIES = 300
MASTER_SEED = 42
MIN_EVENTS = 1
MAX_EVENTS = 6
TARGET_PER_OBJECT_ANSWER_PAIR = 15  # 4 objects x 5 answers x 15 = 300


class InvalidActionError(ValueError):
    """An action violates the world's physical rules."""


class InvalidStateError(ValueError):
    """A state violates the formal world specification."""


def make_random_initial_state(rng):
    """Randomize all agent and loose-object locations."""
    return {
        "agent_locations": {
            agent: rng.choice(LOCATIONS) for agent in AGENTS
        },
        "object_states": {
            object_name: rng.choice(LOCATIONS) for object_name in OBJECTS
        },
    }


def validate_state(state):
    if set(state.get("agent_locations", {})) != set(AGENTS):
        raise InvalidStateError("Every agent must appear exactly once.")
    if set(state.get("object_states", {})) != set(OBJECTS):
        raise InvalidStateError("Every object must appear exactly once.")
    for agent, location in state["agent_locations"].items():
        if location not in LOCATIONS:
            raise InvalidStateError(f"{agent} has invalid location {location}.")
    valid_object_values = set(AGENTS) | set(LOCATIONS)
    for object_name, value in state["object_states"].items():
        if value not in valid_object_values:
            raise InvalidStateError(f"{object_name} has invalid state {value}.")


def who_is_carrying(state, object_name):
    value = state["object_states"][object_name]
    return value if value in AGENTS else "Nobody"


def effective_location(state, object_name):
    value = state["object_states"][object_name]
    return state["agent_locations"][value] if value in AGENTS else value


def pick_up_in_place(state, agent, object_name):
    """Internal pickup effect triggered only by an agent's arrival."""
    value = state["object_states"][object_name]
    if value in AGENTS:
        raise InvalidActionError(f"{object_name} is already carried by {value}.")
    if state["agent_locations"][agent] != value:
        raise InvalidActionError(
            f"{agent} and {object_name} are in different locations."
        )
    state["object_states"][object_name] = agent


def format_object_list(object_names):
    names = [f"the {name.lower()}" for name in object_names]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def apply_move(state, agent, destination):
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

    text = f"{agent} travels to {destination}."
    if len(pickups) == 1:
        text = (
            f"{agent} travels to {destination} and automatically picks up "
            f"the {pickups[0].lower()}."
        )
    elif len(pickups) > 1:
        text = (
            f"{agent} travels to {destination} and automatically picks up "
            f"{format_object_list(pickups)}."
        )

    return {
        "action": "move",
        "agent": agent,
        "destination": destination,
        "automatic_pickups": pickups,
        "text": text,
    }


def apply_drop(state, agent, object_name):
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if object_name not in OBJECTS:
        raise InvalidActionError(f"Unknown object: {object_name}")
    if state["object_states"][object_name] != agent:
        raise InvalidActionError(
            f"{agent} cannot drop {object_name}; {agent} is not carrying it."
        )

    location = state["agent_locations"][agent]
    state["object_states"][object_name] = location
    validate_state(state)  # Dropping deliberately does not trigger pickup.
    return {
        "action": "drop",
        "agent": agent,
        "object": object_name,
        "drop_location": location,
        "automatic_pickups": [],
        "text": f"{agent} drops the {object_name.lower()} on {location}.",
    }


def valid_drops(state):
    return [
        (value, object_name)
        for object_name, value in state["object_states"].items()
        if value in AGENTS
    ]


def generate_random_event(state, rng):
    drops = valid_drops(state)
    if drops and rng.random() < 0.30:
        agent, object_name = rng.choice(drops)
        return apply_drop(state, agent, object_name)

    agent = rng.choice(AGENTS)
    destinations = [
        location
        for location in LOCATIONS
        if location != state["agent_locations"][agent]
    ]
    return apply_move(state, agent, rng.choice(destinations))


def render_story(initial_state, events, question_object):
    lines = ["Initial state:"]
    for agent in AGENTS:
        lines.append(
            f"{agent} is on {initial_state['agent_locations'][agent]}."
        )
    for object_name in OBJECTS:
        verb = "are" if object_name in {"Sneakers", "Glasses"} else "is"
        lines.append(
            f"The {object_name.lower()} {verb} lying on "
            f"{initial_state['object_states'][object_name]}."
        )
    lines.extend(["", "Events:"])
    for event_number, event in enumerate(events, start=1):
        lines.append(f"{event_number}. {event['text']}")
    lines.extend(
        ["", f"Question: Who is carrying the {question_object.lower()}?"]
    )
    return "\n".join(lines)


def generate_candidate_story(story_seed):
    rng = random.Random(story_seed)
    initial_state = make_random_initial_state(rng)
    validate_state(initial_state)
    state = deepcopy(initial_state)
    trace = [{"time": 0, "state": deepcopy(state)}]
    events = []

    for event_id in range(1, rng.randint(MIN_EVENTS, MAX_EVENTS) + 1):
        event = generate_random_event(state, rng)
        event["event_id"] = event_id
        events.append(event)
        trace.append(
            {"time": event_id, "event_id": event_id, "state": deepcopy(state)}
        )

    question_object = rng.choice(OBJECTS)
    answer = who_is_carrying(state, question_object)
    return {
        "story_id": None,
        "seed": story_seed,
        "world_version": "1.0",
        "initial_state": initial_state,
        "events": events,
        "question": {
            "type": "object_carrier",
            "object": question_object,
            "text": f"Who is carrying the {question_object.lower()}?",
        },
        "answer": answer,
        "answer_details": {
            "carrier": answer,
            "effective_location": effective_location(state, question_object),
        },
        "story_length": len(events),
        "final_state": deepcopy(state),
        "state_trace": trace,
        "story_text": render_story(initial_state, events, question_object),
    }


def story_fingerprint(story):
    events = []
    for event in story["events"]:
        if event["action"] == "move":
            events.append(
                ("move", event["agent"], event["destination"])
            )
        else:
            events.append(("drop", event["agent"], event["object"]))
    return json.dumps(
        [story["initial_state"], events, story["question"]["object"]],
        sort_keys=True,
    )


def generate_balanced_dataset():
    master_rng = random.Random(MASTER_SEED)
    dataset = []
    fingerprints = set()
    pair_counts = Counter()
    answer_counts = Counter()
    object_counts = Counter()
    attempts = 0

    while len(dataset) < NUMBER_OF_STORIES:
        attempts += 1
        if attempts > 200_000:
            raise RuntimeError("Balanced generation exceeded the attempt limit.")

        candidate = generate_candidate_story(
            master_rng.randrange(1, 2**32)
        )
        answer = candidate["answer"]
        question_object = candidate["question"]["object"]
        pair = (question_object, answer)
        fingerprint = story_fingerprint(candidate)

        if pair_counts[pair] >= TARGET_PER_OBJECT_ANSWER_PAIR:
            continue
        if fingerprint in fingerprints:
            continue

        candidate["story_id"] = f"story_{len(dataset) + 1:04d}"
        dataset.append(candidate)
        fingerprints.add(fingerprint)
        pair_counts[pair] += 1
        answer_counts[answer] += 1
        object_counts[question_object] += 1

    return dataset, answer_counts, object_counts, attempts


def replay_story(story):
    state = deepcopy(story["initial_state"])
    validate_state(state)
    for stored_event in story["events"]:
        if stored_event["action"] == "move":
            replayed = apply_move(
                state, stored_event["agent"], stored_event["destination"]
            )
            if replayed["automatic_pickups"] != stored_event["automatic_pickups"]:
                raise AssertionError(
                    f"{story['story_id']} has inconsistent automatic pickups."
                )
        elif stored_event["action"] == "drop":
            replayed = apply_drop(
                state, stored_event["agent"], stored_event["object"]
            )
            if replayed["drop_location"] != stored_event["drop_location"]:
                raise AssertionError(
                    f"{story['story_id']} has an inconsistent drop location."
                )
        else:
            raise AssertionError(
                f"{story['story_id']} has unknown action {stored_event['action']}."
            )
    return state


def validate_dataset(dataset):
    for story in dataset:
        final_state = replay_story(story)
        if final_state != story["final_state"]:
            raise AssertionError(f"{story['story_id']} has a wrong final state.")
        object_name = story["question"]["object"]
        if who_is_carrying(final_state, object_name) != story["answer"]:
            raise AssertionError(f"{story['story_id']} has a wrong answer.")
        if (
            effective_location(final_state, object_name)
            != story["answer_details"]["effective_location"]
        ):
            raise AssertionError(
                f"{story['story_id']} has a wrong effective location."
            )


def location_statistics(dataset):
    counters = {
        "initial_agent_locations": Counter(),
        "initial_object_locations": Counter(),
        "movement_destinations": Counter(),
        "final_question_locations": Counter(),
    }
    for story in dataset:
        counters["initial_agent_locations"].update(
            story["initial_state"]["agent_locations"].values()
        )
        counters["initial_object_locations"].update(
            story["initial_state"]["object_states"].values()
        )
        for event in story["events"]:
            if event["action"] == "move":
                counters["movement_destinations"][event["destination"]] += 1
        counters["final_question_locations"][
            story["answer_details"]["effective_location"]
        ] += 1
    return counters


def print_counter(title, counter, labels):
    print(f"\n{title}")
    for label in labels:
        print(f"  {label}: {counter[label]}")


def main():
    print("Generating 300 balanced stories...")
    dataset, answer_counts, object_counts, attempts = generate_balanced_dataset()
    print("Validating every story...")
    validate_dataset(dataset)

    output_path = Path("data/world_stories_300.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nDataset successfully generated and validated.")
    print(f"Stories saved: {len(dataset)}")
    print(f"Candidate attempts: {attempts}")
    print(f"Output file: {output_path.resolve()}")
    print_counter("Answer counts", answer_counts, ANSWERS)
    print_counter("Question-object counts", object_counts, OBJECTS)
    for title, counter in location_statistics(dataset).items():
        print_counter(title.replace("_", " ").title(), counter, LOCATIONS)

    print("\nFirst three sample stories:")
    for story in dataset[:3]:
        print("\n" + "=" * 60)
        print(story["story_id"])
        print("=" * 60)
        print(story["story_text"])
        print(f"Answer: {story['answer']}")
    print("\nAll 300 stored answers agree with fresh simulation.")


if __name__ == "__main__":
    main()
