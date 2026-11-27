The architecture confirms that predictions depend on time flow, sampled latents, and a last-reading register—several places where reconstruction can go wrong. The output was truncated before the current calibration objects appeared. I’ll retrieve those and the remaining identification checks; robust record-derived effects would justify GPU calibration, while weak identification would require revisiting the targets.

```python
notes = open("/work/notes.md").read()
print("=== IDENTIFICATION CHECKS ===")
print(notes[4200:10000])
print("=== NOTEBOOK HEADINGS ===")
print("\n".join(line for line in notes.splitlines() if line.startswith("## ")))
print("=== KERNEL NAMES ===")
print(" ".join(sorted(k for k in globals() if not k.startswith("__"))))
print("=== SAVED FORWARD FILES ===")
for d in sorted(os.listdir("/work")):
    if d.startswith("fwd_"):
        print(d, ", ".join(os.listdir("/work/" + d)))
```
