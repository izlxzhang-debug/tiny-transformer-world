#!/usr/bin/env python3
"""Causal activation patching for the Week 9 tiny transformer.

Creates counterfactual story pairs whose queried object's carrier differs by
one agent token, patches one [layer, token] activation at a time, and writes a
heatmap plus random-direction/token/layer and unrelated-property controls.

Run from the project root:
  python3 week13/intervene_hidden_states.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Install matplotlib: python3 -m pip install matplotlib") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from week12.extract_hidden_states import (  # noqa: E402
    AGENTS, LABELS, LOCATIONS, OBJECTS, TinyTransformerWithHiddenStates,
    choose_device, load_model, tokenize,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("results/tiny_transformer_training/seed_11/checkpoint_best.pt"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/activation_interventions"))
    p.add_argument("--object", choices=OBJECTS, default="Key")
    p.add_argument("--source-carrier", choices=AGENTS, default="Lammy")
    p.add_argument("--target-carrier", choices=AGENTS, default="Anneena")
    p.add_argument("--destination", choices=LOCATIONS, default="Venus")
    p.add_argument("--controls", type=int, default=50)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return p.parse_args()


def article_object(name: str) -> str:
    return name.lower()


def initial_state_for(query_object: str, destination: str) -> dict[str, Any]:
    """State where either selected agent can arrive and collect only query_object."""
    other_locations = [x for x in LOCATIONS if x != destination]
    agent_locations = {
        agent: other_locations[i % len(other_locations)]
        for i, agent in enumerate(AGENTS)
    }
    # Same starting location is legal and makes source/target pairs differ in
    # exactly one event-agent token rather than in their initial states.
    agent_locations["Lammy"] = other_locations[0]
    agent_locations["Anneena"] = other_locations[0]
    object_states = {query_object: destination}
    for i, obj in enumerate(x for x in OBJECTS if x != query_object):
        object_states[obj] = other_locations[(i + 1) % len(other_locations)]
    return {"agent_locations": agent_locations, "object_states": object_states}


def render_story(state: dict[str, Any], events: list[dict[str, Any]], query: str) -> str:
    lines = ["Initial state:"]
    for agent in AGENTS:
        lines.append(f"{agent} is on {state['agent_locations'][agent]}.")
    for obj in OBJECTS:
        lines.append(f"The {article_object(obj)} is lying on {state['object_states'][obj]}.")
    lines.extend(("", "Events:"))
    for number, event in enumerate(events, 1):
        if event["action"] == "move":
            sentence = f"{event['agent']} travels to {event['destination']}"
            if event.get("pickup"):
                sentence += f" and automatically picks up the {article_object(event['pickup'])}"
            lines.append(f"{number}. {sentence}.")
        else:
            lines.append(
                f"{number}. {event['agent']} drops the {article_object(event['object'])} "
                f"on {event['location']}."
            )
    lines.extend(("", f"Question: Who is carrying the {article_object(query)}?"))
    return "\n".join(lines)


def make_answer_pair(query: str, source_carrier: str, target_carrier: str,
                     destination: str) -> tuple[str, str]:
    """A minimal pair: only the arriving agent name changes."""
    state = initial_state_for(query, destination)
    source = render_story(state, [{"action": "move", "agent": source_carrier,
                                   "destination": destination, "pickup": query}], query)
    target = render_story(state, [{"action": "move", "agent": target_carrier,
                                   "destination": destination, "pickup": query}], query)
    return source, target


def make_unrelated_pair(query: str, query_carrier: str, destination: str) -> tuple[str, str]:
    """Pair changes who gets another object; queried carrier stays fixed."""
    state = initial_state_for(query, destination)
    unrelated = next(obj for obj in OBJECTS if obj != query)
    second_destination = next(
        loc for loc in LOCATIONS
        if loc != destination and loc != state["agent_locations"]["Jade"]
    )
    state["object_states"][unrelated] = second_destination
    common = {"action": "move", "agent": query_carrier,
              "destination": destination, "pickup": query}
    first = render_story(
        state,
        [common, {"action": "move", "agent": "Jade",
                  "destination": second_destination, "pickup": unrelated}],
        query,
    )
    second = render_story(
        state,
        [common, {"action": "move", "agent": "Penguin",
                  "destination": second_destination, "pickup": unrelated}],
        query,
    )
    return first, second


def encode(text: str, token_to_index: dict[str, int], capacity: int,
           device: torch.device) -> tuple[Tensor, list[str]]:
    tokens = ["<cls>"] + tokenize(text)
    if len(tokens) > capacity:
        raise ValueError(f"Story has {len(tokens)} tokens but capacity is {capacity}")
    unk = token_to_index["<unk>"]
    ids = torch.tensor([[token_to_index.get(t, unk) for t in tokens]],
                       dtype=torch.long, device=device)
    return ids, tokens


@torch.no_grad()
def clean_run(model: TinyTransformerWithHiddenStates, ids: Tensor
              ) -> tuple[Tensor, dict[str, Tensor]]:
    logits, states = model(ids, return_hidden_states=True)
    return logits[0], {k: v.clone() for k, v in states.items()}


@torch.no_grad()
def patched_run(model: TinyTransformerWithHiddenStates, ids: Tensor, layer: str,
                position: int, replacement: Tensor) -> Tensor:
    """Replace one residual-stream activation, then run remaining computation."""
    batch, length = ids.shape
    positions = torch.arange(length, device=ids.device).unsqueeze(0).expand(batch, length)
    x = model.embedding_dropout(
        model.token_embedding(ids) + model.position_embedding(positions)
    )
    padding_mask = ids.eq(model.pad_index)

    if layer == "embedding":
        x[:, position] = replacement
    for layer_number, block in enumerate(model.blocks, 1):
        x = block(x, padding_mask)
        if layer == f"layer_{layer_number}":
            x[:, position] = replacement
    normalized = model.final_norm(x)
    if layer == "final_norm":
        normalized[:, position] = replacement
    return model.classifier(normalized[:, 0])[0]


def margin(logits: Tensor, source_index: int, target_index: int) -> float:
    return float((logits[source_index] - logits[target_index]).item())


def normalized_effect(patched_logits: Tensor, target_margin: float,
                      denominator: float, source_index: int,
                      target_index: int) -> float:
    return (margin(patched_logits, source_index, target_index) - target_margin) / denominator


def token_labels(tokens: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    labels = []
    for i, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        labels.append(f"{i}:{token}")
    return labels


def save_heatmap(matrix: np.ndarray, layers: list[str], tokens: list[str],
                 path: Path) -> None:
    width = max(14, len(tokens) * 0.24)
    fig, ax = plt.subplots(figsize=(width, 4.8))
    finite_max = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    scale = max(finite_max, 0.05)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm",
                      vmin=-scale, vmax=scale)
    ax.set_yticks(range(len(layers)), layers)
    ax.set_xticks(range(len(tokens)), token_labels(tokens), rotation=90, fontsize=6)
    ax.set_xlabel("Token position in the target story")
    ax.set_ylabel("Patched activation stage")
    ax.set_title("Source-to-target activation-patching effect")
    fig.colorbar(image, ax=ax, label="normalized source-answer restoration")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "std": float(array.std()),
        "mean_absolute": float(np.abs(array).mean()),
        "maximum_absolute": float(np.abs(array).max()),
    }


def main() -> None:
    args = parse_args()
    if args.source_carrier == args.target_carrier:
        raise ValueError("Source and target carriers must differ")
    if args.controls <= 0 or args.top_k <= 0:
        raise ValueError("--controls and --top-k must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    model, token_to_index, metadata = load_model(args.checkpoint, device)
    capacity = metadata["position_capacity"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_text, target_text = make_answer_pair(
        args.object, args.source_carrier, args.target_carrier, args.destination
    )
    source_ids, source_tokens = encode(source_text, token_to_index, capacity, device)
    target_ids, target_tokens = encode(target_text, token_to_index, capacity, device)
    if source_tokens != target_tokens:
        differing = [i for i, (a, b) in enumerate(zip(source_tokens, target_tokens)) if a != b]
    else:
        differing = []
    if source_ids.shape != target_ids.shape:
        raise ValueError("Counterfactual stories must have aligned token positions")

    source_logits, source_states = clean_run(model, source_ids)
    target_logits, target_states = clean_run(model, target_ids)
    source_index = LABELS.index(args.source_carrier)
    target_index = LABELS.index(args.target_carrier)
    source_margin = margin(source_logits, source_index, target_index)
    target_margin = margin(target_logits, source_index, target_index)
    denominator = source_margin - target_margin
    if abs(denominator) < 1e-8:
        raise RuntimeError("Clean source and target have identical answer margins; choose another pair")

    layers = list(source_states)
    matrix = np.zeros((len(layers), len(target_tokens)), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for layer_i, layer in enumerate(layers):
        for position, token in enumerate(target_tokens):
            logits = patched_run(model, target_ids, layer, position,
                                 source_states[layer][:, position])
            effect = normalized_effect(logits, target_margin, denominator,
                                       source_index, target_index)
            matrix[layer_i, position] = effect
            rows.append({"control": "true_source", "layer": layer,
                         "position": position, "token": token, "effect": effect})

    flat_order = np.argsort(np.abs(matrix), axis=None)[::-1]
    selected_sites = [np.unravel_index(i, matrix.shape) for i in flat_order[:args.top_k]]
    controls: dict[str, list[float]] = {
        "random_direction": [], "random_layer": [], "random_token": [],
        "unrelated_property": [],
    }

    # Matched-norm random-direction controls at the strongest observed sites.
    for layer_i, position in selected_sites:
        layer = layers[layer_i]
        target_vector = target_states[layer][:, position]
        delta = source_states[layer][:, position] - target_vector
        delta_norm = float(delta.norm().item())
        for _ in range(args.controls):
            direction = torch.randn_like(delta)
            direction = direction / direction.norm().clamp_min(1e-12) * delta_norm
            logits = patched_run(model, target_ids, layer, position,
                                 target_vector + direction)
            controls["random_direction"].append(normalized_effect(
                logits, target_margin, denominator, source_index, target_index
            ))

    strongest_layer_i, strongest_position = selected_sites[0]
    strongest_layer = layers[strongest_layer_i]
    for _ in range(args.controls):
        random_layer = rng.choice(layers).item()
        logits = patched_run(model, target_ids, random_layer, strongest_position,
                             source_states[random_layer][:, strongest_position])
        controls["random_layer"].append(normalized_effect(
            logits, target_margin, denominator, source_index, target_index
        ))
        random_position = int(rng.integers(0, len(target_tokens)))
        logits = patched_run(model, target_ids, strongest_layer, random_position,
                             source_states[strongest_layer][:, random_position])
        controls["random_token"].append(normalized_effect(
            logits, target_margin, denominator, source_index, target_index
        ))

    # An unrelated pair changes who collects another object but keeps the key's
    # carrier fixed. Its source activations should not restore the main source answer.
    unrelated_a, unrelated_b = make_unrelated_pair(
        args.object, args.target_carrier, args.destination
    )
    unrelated_a_ids, unrelated_a_tokens = encode(
        unrelated_a, token_to_index, capacity, device
    )
    unrelated_b_ids, unrelated_b_tokens = encode(
        unrelated_b, token_to_index, capacity, device
    )
    _, unrelated_a_states = clean_run(model, unrelated_a_ids)
    _, unrelated_b_states = clean_run(model, unrelated_b_ids)
    common_length = min(len(unrelated_a_tokens), len(unrelated_b_tokens))
    for _ in range(args.controls):
        layer = rng.choice(layers).item()
        position = int(rng.integers(0, common_length))
        logits = patched_run(model, unrelated_b_ids, layer, position,
                             unrelated_a_states[layer][:, position])
        base_logits, _ = clean_run(model, unrelated_b_ids)
        # Raw change in the same source-vs-target answer margin, normalized by
        # the main counterfactual's clean margin difference.
        effect = (margin(logits, source_index, target_index) -
                  margin(base_logits, source_index, target_index)) / denominator
        controls["unrelated_property"].append(effect)

    save_heatmap(matrix, layers, target_tokens, args.output_dir / "patching_heatmap.png")
    with (args.output_dir / "patching_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.save(args.output_dir / "patching_effects.npy", matrix)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "pair": {
            "source_answer": args.source_carrier,
            "target_answer": args.target_carrier,
            "differing_token_positions": differing,
            "source_text": source_text,
            "target_text": target_text,
            "source_prediction": LABELS[int(source_logits.argmax())],
            "target_prediction": LABELS[int(target_logits.argmax())],
            "source_margin": source_margin,
            "target_margin": target_margin,
        },
        "top_sites": [
            {"layer": layers[li], "position": int(pos),
             "token": target_tokens[pos], "effect": float(matrix[li, pos])}
            for li, pos in selected_sites
        ],
        "controls": {name: summary(values) for name, values in controls.items()},
        "interpretation": (
            "Effect 1 means full restoration of the clean source-vs-target logit-margin "
            "difference; 0 means no change from the clean target. Negative and >1 effects "
            "are possible. Site ranking is exploratory because it is selected on this pair."
        ),
        "location_pair_warning": (
            "With this checkpoint, 'key in garden' versus 'key in library' has the same "
            "carrier answer (Nobody), and garden/library are outside the trained vocabulary "
            "and output classes. A carrier-changing pair is therefore used."
        ),
    }
    (args.output_dir / "intervention_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Activation intervention complete")
    print("Source clean prediction:", report["pair"]["source_prediction"])
    print("Target clean prediction:", report["pair"]["target_prediction"])
    print("Heatmap:", (args.output_dir / "patching_heatmap.png").resolve())
    print("Report:", (args.output_dir / "intervention_report.json").resolve())


if __name__ == "__main__":
    main()
