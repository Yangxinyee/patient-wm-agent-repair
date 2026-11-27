The inventory was truncated before the useful diagnostics, so it adds no evidence for an edit. I’ll restrict the output to the head caches and replay helpers. This should reveal whether we already have enough validated computations to test a compensated dose pathway.

```python
import inspect, numpy as np, torch
print("FOCUSED CACHE OBJECTS:")
for name, obj in sorted(list(globals().items())):
    if not name.startswith(("hp_", "he_", "ce_", "rx_", "repair_")):
        continue
    if isinstance(obj, dict):
        print(name, "dict", len(obj), str(list(obj)[:12])[:300])
    elif not isinstance(obj, (np.ndarray, torch.Tensor)) and not inspect.isfunction(obj):
        print(name, type(obj).__name__, str(obj)[:180])
print("\nACTION / LATENT TENSOR LAYOUT:")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_in", "z_head", "encode_action", "embed.numeric")):
        print(name, SHAPES[name])
print("\nHEAD BUILDER:")
rx_show_function("ce_make_heads", limit=3600)
print("\nREPLAY HELPER GLOBALS:")
for name in ("rx_layer2", "rx_cross133", "rx_linear", "rx_mlp"):
    obj = globals().get(name)
    if inspect.isfunction(obj):
        print(name, "names:", obj.__code__.co_names, "constants:", str(obj.__code__.co_consts)[:500])
print("\nPAIRED HEAD CACHE SUMMARY:")
for name in ("hp_phi0", "hp_dphi", "hp_mean0", "hp_meanp", "hp_nll0", "hp_nllp", "he_numeric_original", "he_numeric_probe"):
    obj = globals().get(name)
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        x = obj.detach().cpu().numpy() if torch.is_tensor(obj) else obj
        print(name, x.shape, "mean/std/range:", float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 150\nRun 149 inventory exceeded output limit; no new validation or edit. Restricting output to head caches, action/latent tensors, and bounded helper internals. Counts: 150 Python runs, 8 forward passes.\n")
```