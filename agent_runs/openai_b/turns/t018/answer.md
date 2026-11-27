The architecture matches the observed exact invariance: dose changes enter through the edited magnitude embeddings, while reading values remain untouched. I’ll now check the available dose variation and retained causal-analysis objects, and identify the remaining tuning batches. Strong nonlinear dose associations after adjustment would argue against submitting a zero-effect model.

```python
print("=== ANALYSIS OBJECTS ===")
snapshot = list(globals().items())
for name, obj in snapshot:
    if name.startswith("_"):
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape)
    elif isinstance(obj, dict) and any(
        s in name.lower() for s in ("resid", "causal", "prediction", "encoding", "model")
    ):
        print(name, {
            str(k): (
                {"type": type(v).__name__, "shape": str(v.shape)}
                if hasattr(v, "shape")
                else {"type": type(v).__name__,
                      "keys": list(v.keys()) if isinstance(v, dict) else None}
            )
            for k, v in obj.items()
        })
    elif callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print("function:", name)

print("\n=== CAUSAL RESULTS ===")
print(causal_results)
print("\n=== EDIT INVARIANCE CHECK ===")
W = load("embed.numeric.weight")
print("numeric shape:", tuple(W.shape))
print("max absolute dose/rate magnitude weights:",
      float(W[:, [2, 4]].abs().max()))
for field in ("dose", "rate"):
    B = load(f"embed.bins.{field}.weight")
    print(field, "max nonmissing-bin difference:",
          float((B[1:] - B[1]).abs().max()))

print("\n=== VOCAB KEYS ===")
print(list(VOCAB))
for key in VOCAB:
    if any(s in key.lower() for s in ("norm", "code", "config")):
        print(key, VOCAB[key])

tuning_ids = np.sort(
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique()
)
print("\n=== REMAINING TUNING IDS ===")
print("batch 3:", tuning_ids[200:300].tolist())
print("batch 4:", tuning_ids[300:400].tolist())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 12: architecture audit\n"
        "Dose edits change dose/rate numeric scores and fine-bin embeddings. "
        "The three edited tensors remove those magnitude paths while "
        "preserving code, presence, missingness, and measurement inputs. "
        "Reading-head registers are per measurement code, so drug-rate "
        "registers do not directly affect reading attributes. Inspecting "
        "residual-analysis objects for nonlinear checks and identifying "
        "the two remaining tuning batches.\n"
    )
```