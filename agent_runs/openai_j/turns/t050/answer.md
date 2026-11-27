The probe did not improve doubled insulin and weakened doubled norepinephrine, despite preserving factual fit. I’ll reject it and restore the original input weights. Next I’ll inspect the available state and readout features for a more selective correction. A useful feature must respond to dose changes without substantially disturbing factual predictions.

```python
drop_edit("embed.numeric.weight")

print("CURRENT EDITS:", edited())
print("\n=== RELEVANT TENSORS ===")
prefixes = (
    "embed.numeric", "embed.bins", "encode_action",
    "attr_heads.value_bin", "last_numeric", "last_status",
    "core.z_", "core.z_head", "core.z_in"
)
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
print("top-level keys:", list(VOCAB.keys()))
print("codes:", VOCAB.get("codes"))
print("config:", VOCAB.get("config"))
for key, value in VOCAB.items():
    if isinstance(value, dict) and key not in ("codes", "config"):
        print(key, "keys:", list(value.keys())[:15])

print("\n=== CACHED ARRAY / TABLE VARIABLES ===")
shown = 0
for key, value in list(globals().items()):
    if key.startswith("_"):
        continue
    if isinstance(value, (np.ndarray, pd.DataFrame)):
        print(key, type(value).__name__, value.shape)
        shown += 1
    elif isinstance(value, dict) and any(
        term in key.lower() for term in ("kernel", "state", "feature", "target", "policy", "param")
    ):
        print(key, "dict keys:", list(value.keys())[:12])
        shown += 1
    if shown >= 85:
        break

with notes_path.open("a") as f:
    f.write(
        "\n## Forward 7 and run 42: rejected input-gain probe\n"
        "- Original/edited insulin x2: -3.680/-3.638; half: +2.018/+2.380. "
        "NE x2: +0.465/+0.342; half: -0.871/-1.090.\n"
        "- Scaling numerical dose/rate columns did not repair the main response "
        "and weakened NE doubling. Restored original embed.numeric.weight.\n"
        "- Next: investigate selective algebraic readout/state corrections, "
        "with factual preservation and dose sensitivity checked separately.\n"
        "- Python runs: 42. Forward passes: 7. No active edits.\n"
    )
```