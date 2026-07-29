# Research Question, Hypothesis, and Predictions

**Project:** From Prediction to Understanding  
**Status:** Week 1 research framework  
**Last updated:** 29 July 2026

This document fixes the main question and predictions before model training
begins. They should not be rewritten after seeing the results unless a revision
is clearly dated and explained in the research log.

## Primary research question

> When a tiny transformer correctly tracks agents and objects in a controlled
> fictional world, does it learn and use a reusable internal representation of
> that world, or does it rely primarily on surface-level linguistic patterns?

## Secondary research questions

1. **Behavior and generalization:** How accurately does the transformer identify
   the current carrier of an object in familiar stories and in deliberately
   unfamiliar stories containing new combinations, names, lengths, distractors,
   or paraphrases?
2. **Internal representation:** Can the current carrier and location of an
   object be decoded from the transformer's hidden states more reliably than
   from appropriate control representations?
3. **Causal use:** Does changing a carrier-related hidden representation alter
   the model's answer in a targeted and predictable way?

## Testable hypothesis

> A tiny transformer trained on controlled fictional stories will outperform
> surface-level baselines on questions that require sequential state updates,
> will generalize to some deliberately unfamiliar stories, and will develop
> hidden states from which current object carriers can be linearly decoded
> more accurately than from untrained or control representations.

The hypothesis will be evaluated using the mean result from at least three
independent transformer training runs.

## Predictions

### Prediction 1 — Performance above guessing

On a balanced five-answer task—Lammy, Anneena, Jade, Penguin, or Nobody—the
majority-class baseline should achieve approximately 20% accuracy. The trained
transformer should achieve clearly higher standard-test accuracy across the
three training runs.

### Prediction 2 — Advantage over surface-level word counts

The transformer should outperform a bag-of-words classifier on stories where
the answer depends on the order of travel, automatic pickup, and drop events
rather than the presence of a character or location name.

### Prediction 3 — Generalization has limits

The transformer should perform better on the standard test set than on the
longer-story and unfamiliar-paraphrase test sets. It may generalize more
successfully to unseen agent-object combinations than to unfamiliar sentence
forms.

### Prediction 4 — Carrier information appears in trained hidden states

A controlled linear probe should predict the current carrier of an object more
accurately from a trained transformer's hidden states than from:

- an untrained transformer's hidden states;
- randomly assigned carrier labels; and
- a simple input-only control.

### Prediction 5 — Targeted interventions are more specific than random ones

If activation patching is completed, replacing a carrier-related activation
with the corresponding activation from a matched comparison story should
change the predicted carrier more often than equally sized interventions at
random layers, tokens, or directions.

## Operational definition of a limited world model

For this project, a model has a **limited world model** when it satisfies all
four of the following experimental criteria:

1. **State maintenance:** It retains information about agent locations, object
   locations, and object carriers across a sequence of sentences.
2. **State updating:** It changes those states consistently after travel,
   automatic-pickup, and drop events.
3. **Rule generalization:** It applies the learned transition patterns to
   stories that differ deliberately from its training examples.
4. **Internal use:** Carrier information is detectable in its hidden states and
   there is evidence that this information contributes to its answers.

This definition applies only to the controlled fictional-world task. Meeting
it would support the claim that the model learned a limited task-specific world
model; it would not establish unrestricted human-like understanding.

## Measurement plan

| Question | Main measurement | Required comparison |
|---|---|---|
| Does the model answer correctly? | Accuracy by question and story type | Majority and bag-of-words baselines |
| Does it generalize? | Accuracy on specialized test sets | Standard-test accuracy |
| Does it represent carriers? | Linear-probe accuracy by layer | Untrained, random-label, and input controls |
| Does it use the representation? | Change in output after activation patching | Matched random interventions |

## Controlled world used in the experiment

The experiment uses four agents—Lammy, Anneena, Jade, and Penguin—four
objects—hairbrush, sneakers, glasses, and key—and four celestial locations:
Mars, Mercury, Venus, and the Moon.

Agents travel between locations and may drop objects they are carrying. When an
agent arrives at a location, they automatically collect every uncarried object
there. The main question is:

> Who is carrying the `[object]`?

The complete entities, transition rules, valid examples, invalid actions, and
edge-case decisions are recorded in the
[controlled fictional-world specification](world_specification.md).

## Revision rule

If the simulator or pilot data reveal that a question or prediction cannot be
tested as written, record the problem in the weekly research log before
revising this document. Preserve the earlier version in Git history and explain
why the revision was necessary.
