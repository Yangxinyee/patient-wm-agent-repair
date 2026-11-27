The architecture confirms that the current edit changes only the value-bin head; the recurrent dynamics and drug embeddings remain untouched. The recovery output was truncated before the stay IDs and causal notes. I’ll retrieve those compactly and preserve this trial. Its paired rollout should show whether the larger contrast actually strengthens insulin effects, including the halving response.

```python
import torch, hashlib, inspect
from pathlib import Path

ridge1_scale16_checkpoint = {
    n: torch.as_tensor(load(n)).detach().clone() for n in edited()
}
print("Saved ridge1_scale16_checkpoint.")
print("FIRST 100 TUNING IDS:", tuning_ids_recovery[:100])
print("SECOND 100 TUNING IDS:", tuning_ids_recovery[100:200])

print("\nEDIT HELPER SIGNATURES:")
for name in ("make_contrast_graft", "make_selective_contrast_graft", "cache_losses"):
    fn = globals().get(name)
    if callable(fn):
        print(name, inspect.signature(fn))

print("\nRELEVANT CONFIGURATIONS:")
for name in ("alt_contrast_config", "r3_choice", "r3_applied",
             "stronger6_checkpoint", "protected_separators",
             "r3_candidates"):
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, "keys:", list(obj.keys())[:20])
    else:
        print(name, repr(obj))

print("\nFIRST CAUSAL NOTES:")
print(notes_recovery[:5000])
print("\nCAUSAL SECTION HEADINGS:")
for line in notes_recovery.splitlines():
    if line.startswith("#") and any(
        s in line.lower() for s in ("causal", "discontinu", "identif", "norep", "rd", "insulin")
    ):
        print(line)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Ridge-1 factual validation; Python 214, forwards 22/50\n"
        "- Current r3 lambda=1, scale=16 trial: first-100 tuning GPU factual "
        "NLL 1.63803 versus original 1.63461, delta +0.00342.\n"
        "- Saved ridge1_scale16_checkpoint; stronger6_checkpoint remains the "
        "validated fallback. Rollout and broader factual validation are pending.\n"
    )
```
