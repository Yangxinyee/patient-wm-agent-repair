The head update is a low-dimensional shear, so I can halve it without changing its directions. I’ll install strength 0.10. If factual damage falls enough and the rollout benefit remains measurable, this would support a smaller repair; otherwise this pathway is unlikely to help.

```python
head_strength76 = 0.10
head_reduced76 = np.eye(4) + (
    head_strength76 / head_strength73
) * (head_reduced73 - np.eye(4))

apply_lowrank_right_transform(
    "attr_heads.value_bin.0.weight", head_basis73, head_reduced76
)
pilot76_ops = [
    dict(o, vector_array=np.load(Path("/work") / o["vector"]).copy())
    for o in ops()
]
print("Installed strength:", head_strength76)
print("Operation count:", len(ops()))
print("Reduced transform:\n", head_reduced76)

print("\n=== PRIOR VALIDATION AND TARGETS ===")
_note76 = Path("/work/notes.md").read_text()
_lines76 = [
    line for line in _note76.splitlines()
    if any(s in line.lower() for s in
           ("target", "factual nll", "rollout", "rejected", "gain 4"))
]
print("\n".join(_lines76[:65]))

with Path("/work/notes.md").open("a") as _f76:
    _f76.write(
        "\n## Python run 76: half-strength recalibrated pilot\n"
        "- Installed numeric gain 4 plus the same head shear at strength 0.10.\n"
        "- Strength 0.20 failed factual fit by +0.00897; testing a weaker version.\n"
        "- Factual and rollout behavior at this strength remain untested. Forwards: 28.\n"
    )
```