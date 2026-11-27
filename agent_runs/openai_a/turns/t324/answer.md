The smoother gate failed: doubling fell to −4.71 and halving to +0.71. I’ll reject it and test the cached sharper gate, which had comparable local contrasts and slightly lower first-sample factual cost. Only a preserved GPU response would support that alternative.

```python
from pathlib import Path

for n, v in validated_gated8_checkpoint.items():
    save_edit(n, np.asarray(v, dtype=np.float32))
assert all(np.array_equal(np.asarray(load(n)), np.asarray(v))
           for n, v in validated_gated8_checkpoint.items())

sharp240_checkpoint = {
    n: np.asarray(v, dtype=np.float32).copy()
    for n, v in gate_candidates274[(240.0, 1.0)].items()
}
for n, v in sharp240_checkpoint.items():
    assert v.shape == tuple(SHAPES[n]) and np.isfinite(v).all()
    save_edit(n, v)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Gate150 rejected; sharper-gate trial; Python 278, forwards 34/50\n"
        "- Gate150 exact rollout: insulin x2 -4.709, half +.705; worse than gated8.\n"
        "- Restored and verified validated_gated8_checkpoint before the next trial.\n"
        "- Installed cached strength240/amplitude1 candidate. CPU local contrasts unchanged;\n"
        "  first NLL difference -.001662, tuning +.000106 versus validated gated8.\n"
        "- Sharp240 is UNACCEPTED pending GPU rollout and factual checks.\n"
    )

print("Sharp240 trial installed; validated fallback protected.")
print("\nEARLY RECORD-IDENTIFICATION NOTES:")
print(Path("/work/notes.md").read_text()[:6000])
```
