The weaker fallback also fails on stays 100–199: factual NLL rises by about 0.00760. It therefore cannot be accepted as-is. I’ll build a factual-head cache for this separate tuning cohort and compare it with the GPU results; agreement will let me screen smaller corrections without relying on the narrow training cache.

```python
import dis
import io

print("=== FACTUAL CSV SCHEMA ===")
for k197 in (26, 27):
    df197 = pd.read_csv(f"/work/fwd_{k197}/factual_edited.csv")
    print(k197, df197.columns.tolist(), df197.head(1).to_dict("records"))

print("=== ORIGINAL STATE SCHEMA ===")
for k197, v197 in r123_states.items():
    print(k197, np.asarray(v197).shape, np.asarray(v197)[:3]
          if k197 != "x" else np.asarray(v197)[:1, :5])

print("=== HEAD-CACHE HELPER BYTECODE ===")
for n197 in ("r129_queries", "rh_register", "r85_bins"):
    o197 = globals().get(n197)
    if not inspect.isfunction(o197):
        continue
    print("\n", n197, inspect.signature(o197))
    buf197 = io.StringIO()
    dis.dis(o197, file=buf197)
    print(buf197.getvalue()[:3200])

print("=== RELEVANT VOCAB ===")
print("VOCAB KEYS", list(VOCAB.keys()))
print("CODES", VOCAB["codes"])
print("REGISTER / PRIOR SHAPES", {
    n: SHAPES[n] for n in NAMES
    if n.startswith(("last_numeric", "last_status", "z_prior", "type_emb",
                     "embed_transition.value", "embed.bins.value"))
})

with open("/work/notes.md", "a") as f197:
    f197.write(
        "\n## Forward 27 / Run 197: weaker fallback fails another cohort\n"
        "Checked combined fallback factual NLL on tuning 100..199: "
        "1.62804 vs 1.62044 original, delta approximately +0.00760. "
        "Neither combined candidate is accepted across tested cohorts. "
        "Next: reconstruct an independent tuning factual-head cache from original states "
        "and validate it against GPU per-reading predictions before screening revisions.\n"
    )
```