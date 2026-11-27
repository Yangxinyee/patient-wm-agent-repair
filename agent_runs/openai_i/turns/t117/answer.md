The record-derived effects persist for several hours, so matching only the next reading would under-correct the rollout. I’ll recover the dose-decoding diagnostics and posterior helper implementations, then build a small recurrent candidate. It must preserve a dose signal after adjustment for pre-dose condition, rather than amplify the dosing policy.

```python
import inspect, numpy as np, pandas as pd

print("=== TARGETS AND DOSE DECODING ===")
for name in ["targets29", "diagnostics27", "decode_table28"]:
    obj108 = globals().get(name)
    if isinstance(obj108, pd.DataFrame):
        print(name)
        print(obj108.to_string(index=False))
print("\n=== LAG ESTIMATES ===")
print(irf_table.to_string(index=False))

print("\n=== KEY HELPER IMPLEMENTATIONS ===")
for name in ["latent_context", "latent_hidden", "latent_start_atom",
             "latent_action_encode", "posterior84", "context84", "drug_u99"]:
    fn108 = globals().get(name)
    if not inspect.isfunction(fn108):
        continue
    try:
        text108 = inspect.getsource(fn108)
    except Exception:
        text108 = ("names=" + str(fn108.__code__.co_names)
                   + "\nconstants=" + str(fn108.__code__.co_consts))
    print(name, text108[:1300])

print("\n=== ALIGNMENT AND CURRENT FEATURES ===")
for name in ["latent_bundles", "action_cases50", "contexts50",
             "hidden50", "post_hidden50", "dose_innovations50",
             "dose_models", "innovation_cache", "latent_empty_e"]:
    obj108 = globals().get(name)
    if isinstance(obj108, dict):
        print(name, "keys", list(obj108.keys())[:8])
    elif hasattr(obj108, "shape"):
        print(name, type(obj108).__name__, tuple(obj108.shape))
    elif isinstance(obj108, list):
        print(name, "length", len(obj108), "first", str(obj108[:1])[:300])
print("first bundle:", str(latent_bundles[:1])[:350])
print("latent z SD:", np.std(latent_state[:, -32:], axis=0).mean())
print("edited:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 108 — persistent-effect target and feature audit\n"
            "- Controlled record estimates show multi-hour drug-effect tails; proximal slopes alone are insufficient.\n"
            "- Recovering dose decoding and posterior helper definitions before constructing the candidate.\n"
            "- No weights changed.\n")
```