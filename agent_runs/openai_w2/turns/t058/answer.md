Scale 12 worsened factual NLL by only 0.00155 nats on these 100 stays, within the limit, but still missed the estimated insulin effect. The fit constraint leaves some room; the dose pathway appears to be the bottleneck. I’ll recover the earlier decoder experiments and action tensors before editing, looking for a direction that changes dose responses without shifting ordinary predictions.

```python
print("=== NOTEBOOK SIZE AND SECTION HEADINGS ===")
print("characters", len(notes_text))
print("\n".join(s for s in notes_text.splitlines() if s.startswith("## ")))
print("=== MIDDLE NOTES ===")
print(notes_text[13500:19500])
print("=== FUNCTIONS ===")
print([k for k, v in list(globals().items())
       if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("=== ACTION / LATENT / VALUE TENSORS ===")
for n in NAMES:
    if any(s in n for s in ["encode_action", "z_post", "attr_heads.value", "embed.bins", "embed.numeric"]):
        print(n, SHAPES[n])
print("=== VOCAB STRUCTURE ===")
print("keys", list(VOCAB))
print("codes", VOCAB.get("codes"))
print("numeric_specs", numeric_specs)
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Runs 46–48: second context recovery\n"
            "Scale-12 factual NLL on benchmark: original 1.574995, edited 1.576541 "
            "(delta +0.001546, within budget). Forward 9: insulin x2/half -3.252/+2.597; "
            "NE x2/half +2.521/-3.393. Numeric scaling saturates and cannot alone "
            "reach record-supported effects. Inspecting action and decoder pathways "
            "before a targeted, reversible regression-derived edit.\n")
```
