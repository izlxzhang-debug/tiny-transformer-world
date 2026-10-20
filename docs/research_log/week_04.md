# Week 4 Research Log: Balanced Dataset Generation

**Status:** Complete

**Last updated:** 3 August 2026

## Work completed

- Kept the complete simulator, generator, validation, statistics, and command
  entry point in one standalone file:
  [`Dataset/Week 4/generate_300_stories.py`](../../Dataset/Week%204/generate_300_stories.py).
- Randomized agent locations, loose-object locations, acting agents, movement
  destinations, drops, story lengths, and queried objects.
- Generated 300 nonduplicate stories with one to six sequential events each.
- Balanced all five answer labels at 60 stories each.
- Balanced all four queried objects at 75 stories each.
- Replayed every saved story and checked its final state, answer, automatic
  pickups, drop locations, and effective object location.
- Saved the validated dataset to
  [`Dataset/Week 4/dataset.json`](../../Dataset/Week%204/dataset.json).

## Reproduction

```bash
python3 'Dataset/Week 4/generate_300_stories.py'
```

The fixed master seed `42` makes the dataset reproducible.
