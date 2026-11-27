The records support a next-hour insulin effect near −2 per rate unit and a norepinephrine effect near +3, but insulin’s longer-term persistence remains uncertain. The numerical-input probe failed; I’ll now test the dose-bin pathway reversibly. Stronger rollout responses with factual NLL still within tolerance would support pursuing this pathway.

```python
import inspect
import numpy as np
from pathlib import Path

print("=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in [
        "embed.bins", "encode_action", "z_post", "z_in",
        "attr_heads.value_bin", "embed_transition.value_bin", "last_numeric"
    ]):
        print(name, SHAPES[name])

print("\n=== RECONSTRUCTION HELPERS ===")
for helper_name in ["latent_context_at", "encode_single_debug", "reconstruct_reading_head"]:
    helper_obj = globals().get(helper_name)
    if callable(helper_obj):
        print(helper_name, inspect.signature(helper_obj))
        try:
            print(inspect.getsource(helper_obj)[:2400])
        except (OSError, TypeError):
            print("Referenced names:", helper_obj.__code__.co_names)
            print("Arguments/locals:", helper_obj.__code__.co_varnames)

print("\n=== REVERSIBLE BIN-GAIN PROBE ===")
assert not edited(), "Unexpected active edit; do not combine probes."
bin_probe_names = [
    n for n in NAMES
    if n in {"embed.bins.dose.weight", "embed.bins.rate.weight"}
]
assert len(bin_probe_names) == 2, bin_probe_names
for name in bin_probe_names:
    original = load_original(name)
    candidate_weight = original.clone() if hasattr(original, "clone") else original.copy()
    candidate_weight[1:] *= 2.0
    save_edit(name, candidate_weight)
    print(name, "non-absent rows doubled; absent row unchanged")
print("Active edits:", edited())
print("Tuning stay IDs:", sorted(tuning.stay_id.unique().tolist())[:100])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 49: reversible dose-bin gain probe\n"
            "- Doubled non-absent dose/rate bin embeddings, leaving numerical "
            "inputs and absent rows unchanged. This is an exploratory probe, "
            "not an accepted causal repair. Next require paired factual NLL "
            "and rollout validation. Seven forward passes used.\n")
```
