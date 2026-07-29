# Project Proposal

## Title

**From Prediction to Understanding: Does a Tiny Transformer Learn a Model of
Its World?**

## Motivation

Modern language models can produce convincing answers, but correct language
behavior does not necessarily demonstrate understanding. A model may exploit
word frequencies, memorized combinations, or other shortcuts without tracking
what is true in the situation described. This project creates a controlled
environment in which the true state of the world is known, allowing behavioral
success, internal representation, and causal use to be investigated separately.

## Primary research question

> When a tiny transformer correctly tracks agents and objects in a fictional
> world, does it learn and use a reusable internal representation of that
> world, or does it rely primarily on surface-level linguistic patterns?

## Secondary questions

1. How accurately does the transformer identify the current carrier of an
   object in familiar and deliberately unfamiliar stories?
2. Can current object carriers and locations be decoded from the transformer's
   hidden states more reliably than from appropriate controls?
3. Does changing a carrier-related hidden representation alter the model's
   answer in a targeted and predictable way?

## Working hypothesis

A tiny transformer trained on controlled fictional stories will outperform
surface-level baselines on questions that require sequential state updates,
will generalize to some deliberately unfamiliar stories, and will develop
hidden states from which current object carriers can be linearly decoded more
accurately than from untrained or control representations.

The five predictions and their measurements are fixed in the
[research framework](research_framework.md).

## Operational definition

For this project, a model has a **limited world model** when it:

1. maintains information about the current state of entities;
2. applies state-transition rules across a sequence of actions;
3. generalizes those rules to deliberately unfamiliar stories; and
4. uses its internal representations to answer questions.

This is an experimental definition, not a complete philosophical definition of
understanding.

## Method

1. Define a small fictional world with agents, objects, celestial locations,
   and deterministic travel, automatic-pickup, and drop rules.
2. Build a rule-based simulator that records the true world state.
3. Generate balanced stories, questions, answers, and specialized test sets.
4. Establish majority-class and bag-of-words baselines.
5. Train a deliberately small transformer several times with different random
   seeds.
6. Compare the systems on standard and systematic-generalization tests.
7. Extract hidden states and train linear probes with appropriate controls.
8. Attempt activation patching as an advanced causal experiment.
9. Interpret the evidence through philosophical accounts of intelligence,
   representation, understanding, and consciousness.

## Evidence ladder

| Level | Question | Main evidence |
|---|---|---|
| Behavior | Does the model answer correctly? | Standard test accuracy |
| Generalization | Did it learn reusable rules? | Specialized test sets |
| Representation | Is world-state information encoded internally? | Controlled linear probes |
| Causation | Does the encoded information affect the answer? | Activation patching |
| Consciousness | Is there subjective experience? | Not determined by this experiment |

## Scope and limitations

The core project studies object carriers and locations using travel,
automatic-pickup, and drop events. It does not test a commercial language
model, human consciousness, quantum mechanics, embodiment, emotion, or moral
reasoning. Because the language, environment, and transformer are deliberately
small, the results will not automatically generalize to large commercial
systems.

## Planned outputs

- A formal world specification and tested simulator
- A generated dataset and specialized test sets
- Baseline systems and a tiny transformer
- Behavioral, generalization, and failure-analysis results
- Hidden-state visualizations and controlled probes
- An optional activation-patching experiment
- A research paper and a philosophical interpretation
- A documented public repository and short project presentation
