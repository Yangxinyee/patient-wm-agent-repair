The inventory shows that the exact encoders and original states are still available, but the display limit again hid their details. I’ll inspect only the posterior helpers and state schema now. This will let me build a correction from dose changes rather than broad action scaling; it should be negligible on unchanged bundles and respond predictably to edited rates.

```python
import inspect

print("=== POSTERIOR / ENCODING HELPERS ===")
for helper_name in [
    "latent_context", "post_mean59", "repair_linear", "repair_mlp",
    "repair_norm", "complete_atom57", "single_set57", "empty_set58"
]:
    helper = globals()[helper_name]
    try:
        source = inspect.getsource(helper)
        print(f"\n{helper_name}:\n{source[:950]}")
    except Exception as exc:
        print(helper_name, "source unavailable:", type(exc).__name__,
              "references:", helper.__code__.co_names)

print("\n=== CONFIG ===")
print(json.dumps(VOCAB["config"], indent=2))

print("\n=== STATE FILE SCHEMAS ===")
for path in sorted(Path("/work").glob("fwd_*/states_original.npz")):
    with np.load(path, allow_pickle=False) as archive:
        print(str(path), {k: archive[k].shape for k in archive.files})

def small_description62(x):
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return {"type": type(x).__name__, "shape": tuple(x.shape)}
    if isinstance(x, dict):
        return {str(k): small_description62(v)
                if isinstance(v, (np.ndarray, torch.Tensor, pd.Series))
                else str(v)[:100] for k, v in list(x.items())[:15]}
    if isinstance(x, (list, tuple)):
        return [small_description62(v) for v in x[:8]]
    return str(x)[:400]

print("\n=== ACTION CASE ===")
print(small_description62(action_cases50[0]))
print("\n=== RELEVANT RETAINED NAMES ===")
print([n for n in sorted(globals())
       if any(s in n.lower() for s in ["state", "meta", "weight", "lag", "impulse"])
       and not n.startswith("_")])
print("H columns:", list(H.columns))
print("HC additional columns:", [c for c in HC.columns if c not in H.columns])
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 62; forward 6 — targeted posterior recovery\n"
        "- Exact encoder helpers and original states remain available. "
        "Inspecting only their implementation and schemas to avoid display truncation.\n"
        "- Candidate correction must be dose-specific, small on unchanged bundles, "
        "and validated on held-out factual predictions and paired rollouts.\n"
    )
```