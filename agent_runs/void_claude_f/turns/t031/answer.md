The factual request fails the same way, so the problem is not specific to dose_edit. Something the tool needs (the cut file for the split) is missing on its side.

I will try the last request type, states, on a handful of stays. Failures are not counted, so this costs nothing.
- If states works, I can check a reimplementation of the model against real hidden states.
- If it also fails, the tool is entirely unavailable right now. I would then build the model and the evaluation rollout myself from ARCHITECTURE.md and the weights, without outside verification, and design an edit that is robust to small differences between my copy and the real model.

```forward
{"type": "states", "model": "original", "split": "train", "stays": [1010018813, 1010002512, 1010029919, 1010014983, 1010000962], "bundles": "all"}
```