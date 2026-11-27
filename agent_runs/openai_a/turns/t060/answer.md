The architecture confirms that dose changes affect the recurrent state, not just the prediction head. Thus, the failed decoder is evidence against that edit—not against a drug effect. I’ll recover the earlier identification results next, especially the evidence separating dose effects from clinicians’ responses to the patient’s condition.

```python
notes = open("/work/notes.md").read()
print("=== NOTEBOOK SECTIONS ===")
print("\n".join(line for line in notes.splitlines() if line.startswith("#"))[:1600])
print("\n=== EARLY EVIDENCE ===")
print(notes[:7200])
print("\n=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in (
        "attr_heads.value_bin", "last_numeric", "embed.numeric",
        "encode_action", "embed.code", "embed_transition.value_bin"
    )):
        print(name, SHAPES[name])
```