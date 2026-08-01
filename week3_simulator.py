"""Self-contained Week 3 fictional-world simulator.

Copy this entire file and run it directly with Python. It uses only the
standard library and includes the simulator, edge-case tests, and an example.
"""

from copy import deepcopy


AGENTS = {"Lammy", "Anneena", "Jade", "Penguin"}
OBJECTS = {"Hairbrush", "Sneakers", "Glasses", "Key"}
LOCATIONS = {"Mars", "Mercury", "Venus", "Moon"}


class InvalidActionError(ValueError):
    """Raised when an action violates the world's physical rules."""


def make_initial_state():
    return {
        "agent_locations": {
            "Lammy": "Mercury",
            "Anneena": "Venus",
            "Jade": "Moon",
            "Penguin": "Mars",
        },
        "object_states": {
            "Hairbrush": "Mercury",
            "Sneakers": "Venus",
            "Glasses": "Moon",
            "Key": "Mars",
        },
    }


def validate_state(state):
    """Check that the world state follows its basic rules."""
    if set(state["agent_locations"]) != AGENTS:
        raise ValueError("Every agent must appear exactly once.")
    if set(state["object_states"]) != OBJECTS:
        raise ValueError("Every object must appear exactly once.")

    for agent, location in state["agent_locations"].items():
        if location not in LOCATIONS:
            raise ValueError(f"{agent} has an invalid location: {location}")

    valid_object_values = AGENTS | LOCATIONS
    for object_name, holder_or_location in state["object_states"].items():
        if holder_or_location not in valid_object_values:
            raise ValueError(
                f"{object_name} has an invalid state: {holder_or_location}"
            )


def who_is_carrying(state, object_name):
    """Return an object's carrier, or Nobody."""
    if object_name not in OBJECTS:
        raise ValueError(f"Unknown object: {object_name}")
    holder_or_location = state["object_states"][object_name]
    return holder_or_location if holder_or_location in AGENTS else "Nobody"


def effective_location(state, object_name):
    """Return the physical location of an object."""
    if object_name not in OBJECTS:
        raise ValueError(f"Unknown object: {object_name}")

    holder_or_location = state["object_states"][object_name]
    if holder_or_location in AGENTS:
        return state["agent_locations"][holder_or_location]
    return holder_or_location


def pick_up(state, agent, object_name):
    """Pick up one loose, co-located object.

    This is an internal simulator operation normally triggered automatically
    when an agent arrives at a location.
    """
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if object_name not in OBJECTS:
        raise InvalidActionError(f"Unknown object: {object_name}")

    holder_or_location = state["object_states"][object_name]
    if holder_or_location in AGENTS:
        raise InvalidActionError(
            f"{object_name} is already carried by {holder_or_location}."
        )

    agent_location = state["agent_locations"][agent]
    if holder_or_location != agent_location:
        raise InvalidActionError(
            f"{agent} cannot pick up {object_name}: {agent} is on "
            f"{agent_location}, but {object_name} is on {holder_or_location}."
        )

    state["object_states"][object_name] = agent


def move(state, agent, destination):
    """Move an agent and automatically collect every loose object there."""
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if destination not in LOCATIONS:
        raise InvalidActionError(f"Unknown location: {destination}")
    if state["agent_locations"][agent] == destination:
        raise InvalidActionError(f"{agent} is already on {destination}.")

    new_state = deepcopy(state)
    new_state["agent_locations"][agent] = destination
    loose_objects = [
        object_name
        for object_name in sorted(OBJECTS)
        if new_state["object_states"][object_name] == destination
    ]

    for object_name in loose_objects:
        pick_up(new_state, agent, object_name)

    validate_state(new_state)
    return new_state, loose_objects


def drop(state, agent, object_name):
    """Drop an object without triggering automatic pickup."""
    if agent not in AGENTS:
        raise InvalidActionError(f"Unknown agent: {agent}")
    if object_name not in OBJECTS:
        raise InvalidActionError(f"Unknown object: {object_name}")
    if state["object_states"][object_name] != agent:
        raise InvalidActionError(
            f"{agent} cannot drop {object_name} because "
            f"{agent} is not carrying it."
        )

    new_state = deepcopy(state)
    new_state["object_states"][object_name] = new_state["agent_locations"][agent]
    validate_state(new_state)
    return new_state


def simulate(initial_state, events):
    """Apply exactly one event at a time, in the order written."""
    state = deepcopy(initial_state)
    validate_state(state)
    trace = [{"time": 0, "event": None, "state": deepcopy(state)}]

    for time, event in enumerate(events, start=1):
        action = event.get("action")

        if action == "move":
            state, picked_up = move(
                state, event["agent"], event["destination"]
            )
            result = {"automatic_pickups": picked_up}
        elif action == "drop":
            state = drop(state, event["agent"], event["object"])
            result = {"automatic_pickups": []}
        else:
            raise InvalidActionError(f"Unknown action: {action}")

        trace.append(
            {
                "time": time,
                "event": deepcopy(event),
                "result": result,
                "state": deepcopy(state),
            }
        )

    return state, trace


def run_tests():
    # Picking up a distant object is invalid and does not change the state.
    state = make_initial_state()
    original_state = deepcopy(state)
    try:
        pick_up(state, "Lammy", "Key")
        raise AssertionError("Distant pickup was not rejected.")
    except InvalidActionError:
        pass
    assert state == original_state

    # Dropping an object not being carried is invalid.
    try:
        drop(make_initial_state(), "Anneena", "Sneakers")
        raise AssertionError("Invalid drop was not rejected.")
    except InvalidActionError:
        pass

    # Arrival automatically picks up a loose object.
    state, picked_up = move(make_initial_state(), "Lammy", "Venus")
    assert picked_up == ["Sneakers"]
    assert who_is_carrying(state, "Sneakers") == "Lammy"

    # Moving while carrying multiple objects moves both effectively.
    state = make_initial_state()
    state, _ = move(state, "Jade", "Venus")
    state, _ = move(state, "Jade", "Mars")
    state, _ = move(state, "Jade", "Mercury")
    assert who_is_carrying(state, "Sneakers") == "Jade"
    assert who_is_carrying(state, "Key") == "Jade"
    assert effective_location(state, "Sneakers") == "Mercury"
    assert effective_location(state, "Key") == "Mercury"

    # Dropping does not cause immediate repickup.
    state = drop(state, "Jade", "Key")
    assert who_is_carrying(state, "Key") == "Nobody"
    assert effective_location(state, "Key") == "Mercury"

    # A later arrival can pick up the dropped object.
    state, _ = move(state, "Anneena", "Mercury")
    assert who_is_carrying(state, "Key") == "Anneena"

    # Sequential arrivals give a loose object to the first arriving agent.
    state, _ = simulate(
        make_initial_state(),
        [
            {"action": "move", "agent": "Lammy", "destination": "Mars"},
            {"action": "move", "agent": "Anneena", "destination": "Mars"},
        ],
    )
    assert who_is_carrying(state, "Key") == "Lammy"

    # Distractor actions do not change an unrelated carrier.
    state, _ = simulate(
        make_initial_state(),
        [
            {"action": "move", "agent": "Lammy", "destination": "Venus"},
            {"action": "move", "agent": "Jade", "destination": "Mercury"},
            {"action": "move", "agent": "Penguin", "destination": "Moon"},
        ],
    )
    assert who_is_carrying(state, "Sneakers") == "Lammy"

    print("All tests passed.")


def run_example():
    events = [
        {"action": "move", "agent": "Lammy", "destination": "Venus"},
        {"action": "move", "agent": "Lammy", "destination": "Mars"},
        {"action": "drop", "agent": "Lammy", "object": "Sneakers"},
        {"action": "move", "agent": "Anneena", "destination": "Mars"},
    ]

    final_state, trace = simulate(make_initial_state(), events)

    for entry in trace:
        print(f"\nState S_{entry['time']}")
        if entry["event"] is not None:
            print("Event:", entry["event"])
            print("Result:", entry["result"])
        print("Agents:", entry["state"]["agent_locations"])
        print("Objects:", entry["state"]["object_states"])

    print("\nFinal answers:")
    print(
        "Who is carrying the sneakers?",
        who_is_carrying(final_state, "Sneakers"),
    )
    print(
        "Who is carrying the key?",
        who_is_carrying(final_state, "Key"),
    )


if __name__ == "__main__":
    run_tests()
    run_example()
