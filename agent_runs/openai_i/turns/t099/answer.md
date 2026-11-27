The posterior is a two-layer network with only 192 hidden features, so a low-rank correction is possible in principle. The printed implementation is still incomplete, however. I’ll check its input conventions and the results of the earlier scaling probe. If absent fields or helper conventions explain the mismatch, I can fix the reconstruction rather than compensate with an incorrect weight edit.

```python
import json
from pathlib import Path

print("=== COMPACT HELPER CONTRACTS ===")
for n91 in ("complete_atom57", "latent_context", "latent_encode_batch",
            "latent_action_encode", "post_mean59", "context84"):
    fn91 = globals()[n91]
    print(n91, "defaults:", fn91.__defaults__,
          "globals:", fn91.__code__.co_names)

print("\n=== VOCAB CONVENTIONS ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
for n91 in ("ekind_ids57", "event_info57", "cat_ids57", "cat_maps57",
            "categorical_ids", "code_ids"):
    if n91 in globals():
        print(n91, str(globals()[n91])[:1400])

print("\n=== ABSENT-ATTRIBUTE EMBEDDING NORMS ===")
for n91 in NAMES:
    if n91.startswith(("embed.bins.", "embed.categorical.")) and n91.endswith(".weight"):
        w91 = load_original(n91)
        print(n91, tuple(w91.shape), "row0 norm:", float(w91[0].norm()))

print("\n=== CURRENT RECONSTRUCTION ARRAYS ===")
items91 = []
for n91, o91 in list(globals().items()):
    if n91.endswith(("84", "65")) and hasattr(o91, "shape"):
        items91.append((n91, type(o91).__name__, tuple(o91.shape)))
print(items91)

txt91 = Path("/work/notes.md").read_text()
print("\n=== EXACT-CASE NOTES ===")
for label91 in ("Python 57", "Python 58", "Python 59", "Python 60"):
    pos91 = txt91.find("## " + label91)
    if pos91 >= 0:
        end91 = txt91.find("\n## ", pos91 + 4)
        print(txt91[pos91:end91 if end91 >= 0 else pos91 + 600][:500])

print("\n=== PREVIOUS SCALING-PROBE RESULTS ===")
for k91 in (5, 6):
    p91 = Path(f"/work/fwd_{k91}/summary.json")
    obj91 = json.loads(p91.read_text())
    print("forward", k91, str(obj91)[:950])

print("\n=== CONTEXT RETURN CONTRACT ===")
try:
    result91 = context84(True, False)
    if isinstance(result91, tuple):
        print([(type(x91).__name__, getattr(x91, "shape", None)) for x91 in result91])
    else:
        print(type(result91).__name__, getattr(result91, "shape", None))
except Exception as exc91:
    print(type(exc91).__name__, str(exc91)[:200])

with open("/work/notes.md", "a") as f91:
    f91.write(
        "\n## Python 91; forward 6 — compact convention audit\n"
        "- Posterior has 192 hidden features, allowing a low-rank statistical correction.\n"
        "- Checking absent embeddings, helper contracts, and the earlier scaling probe before fitting it.\n"
        "- No weights changed.\n"
    )
```