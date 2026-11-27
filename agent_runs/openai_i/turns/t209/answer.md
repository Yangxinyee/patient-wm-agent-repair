The actual factual NLL increased by 0.00695, so the full-strength candidate fails the limit. I’ll reduce it to 60% before testing rollouts and review the record-based effect estimates. The reduced edit is useful only if factual fit passes and its dose responses move toward those estimates.

```python
gain187 = 0.6
for name, delta in [
    ("attr_heads.value_bin.2.weight", delta_w186),
    ("attr_heads.value_bin.2.bias", delta_b186),
]:
    save_edit(name, np.asarray(load_original(name), dtype=np.float32) + gain187 * delta)
print("Reduced candidate proxy:", evaluate_head183(gain187 * vector186))
print("Active edits:", edited())

print("\n--- data documentation ---")
print(Path("/data/README.md").read_text())

notes187 = Path("/work/notes.md").read_text()
print("\n--- earliest causal-analysis notes ---")
print(notes187[:6500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 187; forward 14 — reduce direct-head edit\n"
            "- Actual tuning factual NLL: 1.63155 original, 1.63850 edited; +.00695 fails the .005 safeguard.\n"
            "- Reduced candidate186 to gain .6. Full-strength candidate is rejected.\n"
            "- Next: actual matched rollout response and reduced-candidate factual validation.\n")
```