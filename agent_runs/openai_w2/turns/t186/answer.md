This is the same failed run; it provides no new result, and the weights are unchanged. I’ll simplify the reporting code and retrieve the evidence first. Agreement between the threshold and history-adjusted estimates would support the current causal calibration.

```python
wanted162 = (
    "## Run 5:", "## Run 6:", "## Run 10:", "## Run 11:",
    "## Run 12:", "## Run 22:", "## Run 33:", "## Run 38:",
    "## Run 39:"
)
text162 = "\n\n".join(
    section for section in sections160
    if section.startswith(wanted162)
)
print("=== RECORD-ONLY IDENTIFICATION AND PERSISTENCE ===")
print(text162[:8500])
print("\nEvidence excerpt length:", len(text162))
print("Current edits:", edited())
```
