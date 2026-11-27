The architecture has an event-and-previous-bin lookup where this correction can be applied exactly. The recovered record estimates support immediate effects of about −1.99 per insulin rate unit and +3.29 per norepinephrine rate unit; the cumulative targets remain provisional. I’ll verify the correction’s indexing and sign before installing it. Reproducing the earlier screen would confirm the implementation.

```python
import dis

print("=== EARLY RECORD ANALYSIS ===")
print(r255_notes[:5800])

print("\n=== TRANSITION LOOKUP ===")
r256_transition_names = [n for n in NAMES if n.startswith("embed_transition.value_bin")]
print([(n, SHAPES[n]) for n in r256_transition_names])
print("codes:", VOCAB["codes"])
r256_example_key = next(iter(r252_tables[80.0]))
print("table example:", r256_example_key,
      np.asarray(r252_tables[80.0][r256_example_key]).round(6))

print("\n=== NLL CORRECTION SIGN ===")
dis.dis(r252_nll)

print("\n=== COMPACT CANDIDATE INTERFACES ===")
for n, v in sorted(r255_globals.items()):
    if n.startswith("r246_") and callable(v):
        print(n, inspect.signature(v))
for split, vals in r252_logits.items():
    print("logits", split, [tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                           for v in vals])
print("train cache:", type(c210_train).__name__)
print("train event counts:", c210_train.event.value_counts().to_dict())
print("train stay identifiers:", r252_train_ids[:8], "n=", len(r252_train_ids))

print("\n=== TUNING COHORT IDENTIFIERS ===")
for n, v in sorted(r255_globals.items()):
    if re.search(r"(tune|tuning).*(ids|stays)|(?:ids|stays).*(tune|tuning)", n, re.I):
        if isinstance(v, (list, tuple, np.ndarray)) and len(v):
            print(n, "n=", len(v), "first=", list(v[:5]), "last=", list(v[-3:]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 256: transition-correction implementation recovery\n"
            "Checking event/previous-bin lookup indexing and NLL correction sign, "
            "and recovering early record-derived target evidence. No weights changed.\n")
```