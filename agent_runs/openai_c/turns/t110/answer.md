The head pilot adds almost nothing: insulin x2 changes from −2.503 to −2.561, while norepinephrine does not improve. I’ll remove it and test a different hypothesis: fixed drug-identity embeddings may suppress dose sensitivity. This test removes only those identity terms, retaining dose, rate, kind, and unit inputs. Improved rollouts with safe NLL would support that hypothesis.

```python
numeric_axis77 = pilot76_ops[0]["vector_array"].copy()
clear_ops()
rank1("embed.numeric.weight", numeric_axis77, 3.0, side="in")

print("Drug numeric axis:", numeric_axis77)
print("Vocabulary codes:", VOCAB["codes"])
print("Code embedding shape:", SHAPES["embed.code.weight"])

def code_index77(label):
    table = VOCAB["codes"]
    if isinstance(table, dict):
        if label in table:
            entry = table[label]
            if isinstance(entry, dict):
                return int(entry.get("id", entry.get("index")))
            return int(entry)
        return int(next(k for k, v in table.items() if v == label))
    for i, entry in enumerate(table):
        if entry == label:
            return i
        if isinstance(entry, dict) and entry.get("name", entry.get("code")) == label:
            return int(entry.get("id", i))
    raise ValueError("Code not found: " + label)

for drug77 in ("insulin", "norepinephrine"):
    row77 = np.zeros(SHAPES["embed.code.weight"][0])
    row77[code_index77(drug77)] = 1.0
    rank1("embed.code.weight", row77, -1.0, side="out")

drug_identity_pilot77_ops = [
    dict(o, vector_array=np.load(Path("/work") / o["vector"]).copy())
    for o in ops()
]
print("Installed operations:", ops())

with Path("/work/notes.md").open("a") as f77:
    f77.write(
        "\n## Forwards 29–30 / Python run 77: head pilot rejected; identity pilot\n"
        "- Head strength 0.10 factual increase +0.00303 passed the probe constraint.\n"
        "- Rollouts: insulin -2.561/+1.697; NE +0.100/+0.016. Negligible benefit.\n"
        "- Removed head shear; retained numeric gain 4.\n"
        "- New exploratory test zeros only insulin and NE rows of embed.code.weight.\n"
        "- Numeric doses/rates, fine bins, kinds and units remain available.\n"
        "- Hypothesis: fixed treatment-identity components mask useful dose variation.\n"
        "- Factual safety and rollout benefit remain untested. Forward passes: 30.\n"
    )
```