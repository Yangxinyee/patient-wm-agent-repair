The cross-fitted checks support an insulin effect near −2 per rate unit and a norepinephrine effect near +3.3, with similar estimates after hour 12. I’ll recover their longer-horizon estimates and the relevant tensor layout. Stable impulse responses would support using them as rollout targets; instability would require wider uncertainty.

```python
print("=== REMAINING EARLY CONCLUSIONS ===")
end63 = notes62.find("## Restart / Run 14")
print(notes62[max(0, end63 - 3600):end63])

print("\n=== RELEVANT TENSORS ===")
prefixes63 = (
    "embed.numeric", "embed.code", "embed.bins", "query",
    "type_emb", "last_", "z_head", "core.z_prior", "core.gap",
    "gap.", "attr_heads.value_bin", "embed_transition.value_bin",
    "core.out_norm", "core.flow"
)
for n63 in NAMES:
    if n63.startswith(prefixes63):
        print(n63, SHAPES[n63])

print("\n=== FORWARD FILES ===")
for k63 in range(1, 8):
    d63 = Path(f"/work/fwd_{k63}")
    print(k63, [p.name for p in d63.iterdir()] if d63.exists() else [])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for k63 in VOCAB:
    if any(s63 in k63.lower() for s63 in ["normal", "kind", "type", "status"]):
        print(k63, str(VOCAB[k63])[:1400])

print("\n=== SELECTED PERSISTENT OBJECTS ===")
for k63, v63 in list(globals().items()):
    if k63.startswith(("wr_", "rh_", "rc_", "trial_", "gated_", "gate_", "head_")):
        if isinstance(v63, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(k63, type(v63).__name__, tuple(v63.shape))
        elif isinstance(v63, dict):
            print(k63, "dict keys", list(v63)[:12])
print("Other dictionary names:",
      [k for k, v in list(globals().items())
       if isinstance(v, dict) and not k.startswith("_")])
```