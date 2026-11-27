The recovered notes show substantial adjusted one-hour effects, but those estimates still need the later confounding checks. The output was truncated before the retained fits appeared. I’ll retrieve those compactly; stable effects across the later checks would support a correction, while placebo failures would weaken the causal interpretation.

```python
print("=== NOTEBOOK LENGTH ===", len(notes_text))
print("=== EARLIER ANALYSES, CONTINUED ===")
print(notes_text[5200:11000])
print("=== RETAINED VARIABLE NAMES ===")
names_now = [n for n in globals() if not n.startswith("_")]
print(", ".join(names_now))
print("=== CANDIDATE/FIT OBJECTS ===")
for line in inventory:
    if any(s in line.lower() for s in
           ["trace", "offset", "innovation", "decoder", "candidate", "effect",
            "causal", "ridge", "bundle", "state", "action", "train", "tune",
            "target", "ctx", "feature"]):
        print(line)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 53\n"
            "Run 52 output was truncated before retained-object inventory. "
            "Recovering later causal checks and candidate fits compactly. No edit.\n"
            "Counts: 53 Python runs, 6 forward passes.\n")
```