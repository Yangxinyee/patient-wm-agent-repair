The architecture confirms that doses affect the latent posterior and recurrent state, not just the value head. The 5× edit is still only a fallback. I’ll recover the earlier causal estimates and state caches; agreement across lagged record analyses would support an effect target, while disagreement would argue against forcing a large response.

```python
r162_notes = open("/work/notes.md").read()
print("=== EARLY RECORD EVIDENCE ===")
print(r162_notes[:6800])
print("=== RELEVANT LIVE OBJECTS ===")
r162_keys = [k for k in globals() if any(s in k.lower() for s in
    ["effect", "causal", "impulse", "lag", "target", "beta", "coef", "panel", "hp_", "he_", "probe", "pair"])]
for k in r162_keys[-55:]:
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:8])
    elif isinstance(v, (int,float,str)):
        print(k, repr(v)[:140])
    elif inspect.isfunction(v):
        print(k, str(inspect.signature(v)))
print("=== STATE FILES ===")
for r162_path in ["/work/fwd_2/states_original.npz",
                  "/work/fwd_4/states_original.npz",
                  "/work/fwd_11/states_edited.npz"]:
    if os.path.exists(r162_path):
        with np.load(r162_path) as zf:
            print(r162_path, [(k, zf[k].shape) for k in zf.files])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 162\nArchitecture recovered: doses enter action encoder, posterior, prelude, and recurrent state; all heads share out. Recovering early causal evidence and exact-state caches to avoid another unvalidated replay. Counts: 162 Python runs, 11 forwards.\n")
```