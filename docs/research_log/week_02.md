# Week 2 Research Log: Formal World Construction

**Status:** Complete

**Last updated:** 30 July 2026

## Work completed

- Formalized the world as a deterministic sequence of states and events.
- Implemented one-location-per-agent and one-state-per-object invariants.
- Implemented sequential `move` and `drop` events.
- Implemented automatic pickup of every loose object on arrival.
- Confirmed that dropping does not trigger immediate repickup.
- Added invalid-action validation.
- Created eleven hand-designed stories covering the required edge cases.
- Defined and generated the complete JSON dataset record.
- Added automated simulator and dataset tests.

## Formal representation

Let \(A\), \(O\), and \(L\) be the sets of agents, objects, and locations.
A state is:

\[
S_t=(P_t,U_t)
\]

where \(P_t:A\rightarrow L\) gives each agent's location and
\(U_t:O\rightarrow A\cup L\) gives each object's carrier or uncarried
location.

Each transition applies exactly one complete event:

\[
S_{t+1}=T(S_t,E_t)
\]

Automatic pickup is part of a move event and finishes before the next event
begins. This eliminates simultaneous-arrival ambiguity.

## Dataset record

Each JSON record stores:

- schema and world versions;
- story ID and seed;
- entity lists;
- complete initial and final states;
- ordered events and automatic pickups;
- the question and answer;
- story length and reasoning depth;
- evidence event IDs; and
- the complete state trace.

For the carrier question, an evidence event changes the queried object's
carrier-or-location value. Reasoning depth is the number of such events.

## Coverage

The stories cover:

- pickup of one object;
- pickup of several objects;
- movement while carrying several objects;
- dropping an object;
- another agent picking up a dropped object;
- a final answer of `Nobody`;
- an attempted invalid drop;
- several agents at one location;
- several objects at one location; and
- irrelevant distraction events.

## Reproduction

Generate the checked dataset with:

```bash
python scripts/generate_week2_dataset.py
```

Run the checks with:

```bash
pytest
```
