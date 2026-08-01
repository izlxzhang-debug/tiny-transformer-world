# Week 3 Research Log: Standalone Simulator Verification

**Status:** Complete

**Last updated:** 1 August 2026

## Work completed

- Consolidated the Week 3 construction work into one self-contained Python
  file that can be copied and run directly.
- Kept automatic pickup as an arrival effect rather than a story action.
- Implemented sequential `move` and `drop` events.
- Implemented effective locations for carried objects.
- Rejected distant pickup, invalid drop, unknown entities, invalid locations,
  repeated same-location movement, and unsupported event types.
- Preserved the input state by returning a new state from every public state
  transition function.
- Included the requested edge-case checks and an executable example.

## Corrected pickup behavior

The first uploaded standalone version had an inconsistent `pick_up()` API: a
successful pickup mutated the supplied state and returned `None`, whereas
`move()` and `drop()` returned new states. The corrected implementation now:

1. validates the supplied state;
2. makes a deep copy;
3. applies pickup to the copy;
4. returns the new state; and
5. leaves the caller's original state unchanged.

Automatic pickup during `move()` uses a private in-place helper because the
move already operates on its own deep-copied state.

## Edge cases verified

- Picking up an object at another location is rejected.
- Dropping an object not carried by the acting agent is rejected.
- Multiple carried objects follow their carrier through effective location.
- Dropping does not cause immediate repickup.
- A later arrival can collect a dropped object.
- The first arrival receives a loose object when agents arrive sequentially.
- Distractor actions involving unrelated agents do not change the queried
  object's carrier.

## Reproduction

The complete implementation, checks, and example are all contained in
[`week3_simulator.py`](../../week3_simulator.py). No third-party libraries are
needed.

Run it with:

```bash
python3 week3_simulator.py
```

Expected opening output:

```text
All tests passed.
```

The example's final answers are:

```text
Who is carrying the sneakers? Anneena
Who is carrying the key? Lammy
```
