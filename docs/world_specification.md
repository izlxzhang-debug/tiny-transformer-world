# Controlled Fictional-World Specification

**Status:** Week 2 implemented specification

**Last updated:** 30 July 2026

This document defines the world that will be implemented by the rule-based
simulator and used to generate the transformer's stories. The rules must be
deterministic: the same initial state and action sequence must always produce
the same final answer.

## Agents

- **Lammy**, a sheep
- **Anneena**, a girl
- **Jade**, a girl
- **Penguin**, a penguin

The simulator treats all four characters as agents with the same movement,
carrying, and dropping abilities.

## Objects

- hairbrush
- sneakers
- glasses
- key

Sneakers and glasses are each treated as one object. Their individual parts
are not tracked.

## Celestial locations

- Mars
- Mercury
- Venus
- Moon

These are called **locations**, rather than planets, because the Moon is not a
planet.

## State representation

Let \(A\), \(O\), and \(L\) be the sets of agents, objects, and locations.
The simulator represents a state as:

\[
S_t=(P_t,U_t)
\]

where \(P_t:A\rightarrow L\) gives each agent's location and
\(U_t:O\rightarrow A\cup L\) gives each object's carrier or uncarried
location.

- If \(U_t(o)\in A\), that agent carries the object.
- If \(U_t(o)\in L\), the object lies uncarried at that location.

The object's physical location is derived from its carrier's location when it
is carried. This unified representation prevents an object from being both
carried and independently located somewhere.

## Actions

### Travel

An agent moves from their current location to a named destination.

Examples:

> Lammy travels to Mars.
>
> Anneena goes to Venus.
>
> Jade runs to Mercury.

Every object carried by the moving agent travels with them.

### Drop

An agent places a carried object at their current location.

Example:

> Lammy drops the sneakers on Venus.

This action is valid only if Lammy is on Venus and is carrying the sneakers.

### Automatic pickup

Pickup is not a separately chosen action. When an agent arrives at a location,
they automatically pick up every uncarried object currently there.

Automatic pickup occurs only on arrival. Dropping an object does not cause the
same agent to pick it up again immediately.

## State-transition rules

1. Every agent is in exactly one location at any time.
2. Every object is either uncarried at exactly one location or carried by
   exactly one agent.
3. Story actions occur one at a time and in the order written.
4. When an agent travels, every object they carry travels with them.
5. When an agent arrives, they automatically pick up all uncarried objects at
   the destination.
6. An agent may carry more than one object.
7. An agent may drop only an object they currently carry.
8. A dropped object remains at the agent's current location.
9. Dropping does not trigger automatic pickup by the agent who dropped the
   object.
10. An object cannot be both carried and independently located somewhere.
11. Two agents cannot carry the same object simultaneously.

Each event is applied by a single transition function:

\[
S_{t+1}=T(S_t,E_t)
\]

The event and all automatic pickups finish before the next event begins. The
index \(t\) is an event index rather than physical clock time, so simultaneous
arrivals are outside the model.

## Required initial state

Every generated story must begin from a complete valid state specifying:

- the starting location of every agent; and
- either the starting carrier or starting location of every object.

The initial state may be stored by the simulator even when the full state is
not written explicitly in the story text.

## Primary question

> Who is carrying the `[object]`?

The five possible answer classes are:

- Lammy
- Anneena
- Jade
- Penguin
- Nobody

`Nobody` is required because a dropped object may be lying uncarried at a
location.

## Valid example 1: automatic pickup

### Initial state

- Lammy is on Mercury.
- Anneena is on Venus.
- Jade is on the Moon.
- Penguin is on Mars.
- The sneakers are on Venus.
- The key is on Mars.

### Story

> Lammy travels to Venus.
>
> Lammy travels to Mars.

On arrival at Venus, Lammy automatically picks up the sneakers. The sneakers
then travel with Lammy to Mars. On arrival at Mars, Lammy also automatically
picks up the key.

Question: Who is carrying the sneakers?

Answer: **Lammy**

Question: Who is carrying the key?

Answer: **Lammy**

## Valid example 2: drop followed by automatic pickup

### Initial state

- Lammy is on Mars carrying the sneakers.
- Anneena is on Venus.

### Story

> Lammy drops the sneakers on Mars.
>
> Anneena travels to Mars.

After the first action, nobody carries the sneakers. When Anneena arrives on
Mars, she automatically picks them up.

Question: Who is carrying the sneakers?

Answer: **Anneena**

## Valid example 3: nobody carries the object

### Initial state

- Jade is on the Moon carrying the hairbrush.

### Story

> Jade drops the hairbrush on the Moon.

Question: Who is carrying the hairbrush?

Answer: **Nobody**

## Invalid actions and states

- Anneena cannot drop the sneakers if she is not carrying them.
- Jade cannot drop the hairbrush if she is not carrying it.
- Lammy cannot occupy Mars and Venus simultaneously.
- An agent cannot drop an object at a location where they are not present.
- Two agents cannot carry the glasses simultaneously.
- The key cannot be carried by Penguin while also lying independently on Mars.
- An object cannot be automatically collected from a different location.

There is no invalid action called "picking up an object in another location"
because pickup is automatic rather than a chosen action. The simulator instead
ensures that automatic pickup occurs only when an agent arrives at the object's
current location.

## Deterministic edge-case decisions

### Several objects at the destination

The arriving agent automatically picks up all uncarried objects there.

### Several agents at one location

Only an arrival event triggers pickup. Agents already at the location do not
pick up a newly dropped object unless they later leave and arrive again.

### A dropped object and its dropper

The object remains uncarried after being dropped. It is not immediately
returned to the same agent.

## Deferred extensions

The first version will not include:

- voluntary pickup choices;
- limits on the number of objects an agent can carry;
- giving objects directly to another agent;
- containers;
- object ownership;
- counterfactual questions;
- beliefs, intentions, emotions, or moral decisions.
