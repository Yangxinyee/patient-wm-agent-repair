The insulin detector is asymmetric, but a previous fully balanced correction harmed factual fit. That rules out repeating it unchanged. I’ll recover its exact scale and validation results to assess a much smaller correction. It is worth testing only if it increases half-dose sensitivity while adding little factual-state disturbance.

```python
print("=== BALANCED-CANDIDATE VALIDATION ===")
notes218 = open("/work/notes.md").read()
for sec218 in re.split(r"(?=^## )", notes218, flags=re.M):
    heading218 = sec218.split("\n", 1)[0]
    if re.search(r"\b(64|65|66|148|149|150|151|152|153|154)\b", heading218):
        print(sec218[:700])

print("\n=== ARCHIVE FIELDS ===")
for fn218 in ["insulin_candidate58.npz", "insulin_candidate63.npz",
              "ne_candidate151.npz"]:
    with np.load("/work/" + fn218, allow_pickle=False) as a218:
        print(fn218)
        for key218 in a218.files:
            x218 = a218[key218]
            print(key218, x218.shape,
                  x218.tolist() if x218.size <= 6 else
                  float(np.linalg.norm(x218)) if
                  np.issubdtype(x218.dtype, np.number) else str(x218.dtype))

print("\n=== FORWARD SUMMARY CONTRACTS ===")
for k218 in [7, 9, 20, 21, 23]:
    sm218 = json.load(open("/work/fwd_%d/summary.json" % k218))
    print(k218, str(sm218)[:950])

print("\n=== RECENT TRAINING-CONTRAST BUFFERS ===")
buf218 = []
for name218, obj218 in list(globals().items()):
    if re.search(r"(18[8-9]|19[0-9])$", name218):
        if isinstance(obj218, (np.ndarray, torch.Tensor, pd.DataFrame)):
            buf218.append((name218, type(obj218).__name__, tuple(obj218.shape)))
        elif inspect.isfunction(obj218):
            buf218.append((name218, "function", obj218.__code__.co_names))
        elif isinstance(obj218, dict):
            buf218.append((name218, "dict", list(obj218)[:8]))
print(buf218[:65])

with open("/work/notes.md", "a") as f218:
    f218.write(
        "\n## Run 218: assess partial insulin contrast balancing\n"
        "Full balancing was previously attempted and cannot be reused blindly. "
        "Recovering candidate scales, measured factual losses, and fresh-training "
        "contrast buffers to assess a small correction rather than the rejected "
        "full edit. No weights changed. Python runs: 218; forwards: 23.\n"
    )
```
