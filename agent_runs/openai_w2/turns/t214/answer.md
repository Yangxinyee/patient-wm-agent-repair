The new cohort’s factual loss increased by 0.00617 nats, above the allowed margin on that cohort. The pooled estimate may still pass, but this is evidence against strengthening the current edit. I’ll check whether the loss is broadly distributed and review how the decoder was fitted. Consistent losses would favor weakening it; unstable outliers could point to a generalization problem.

```python
paired184 = []
for k in [20, 22, 23]:
    o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    assert len(o) == len(e)
    for col in ["stay_id", "hour", "event"]:
        if col in o:
            assert o[col].equals(e[col])
    nllcol184 = next(c for c in o.columns if "nll" in c.lower())
    p = o[[c for c in ["stay_id", "hour", "event"] if c in o]].copy()
    p["delta_nll"] = e[nllcol184].to_numpy() - o[nllcol184].to_numpy()
    p["cohort"] = k
    paired184.append(p)
paired184 = pd.concat(paired184, ignore_index=True)
cluster184 = paired184.groupby("stay_id")["delta_nll"].agg(["sum", "count", "mean"])
mean184 = paired184["delta_nll"].mean()
influence184 = cluster184["sum"] - mean184 * cluster184["count"]
se184 = np.sqrt(len(cluster184) / (len(cluster184) - 1) *
                np.square(influence184).sum()) / len(paired184)
print("POOLED FACTUAL", {"stays": len(cluster184), "readings": len(paired184),
                         "delta_nll": mean184, "cluster_se": se184})
print("\nBY COHORT AND SENSOR")
print(paired184.groupby(["cohort", "event"])["delta_nll"]
      .agg(["mean", "count"]).to_string())
print("\nSTAY-LEVEL CHANGE QUANTILES")
print(cluster184["mean"].quantile([0, .1, .25, .5, .75, .9, 1]).to_string())

print("\nDECODER CONSTRUCTION NOTES")
for i in [47, 48, 50, 56, 63]:
    s = sections180[i]
    print(s[:700])

print("\nAVAILABLE LOCAL MODEL/FEATURE FUNCTIONS")
functions184 = []
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and getattr(obj, "__module__", "") == "__main__" and hasattr(obj, "__code__"):
        try:
            functions184.append(name + " " + str(inspect.signature(obj))[:125])
        except Exception:
            pass
print("\n".join(functions184)[:3200])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Forward23 / run184: factual margin narrows\n"
            "Current validated_B176 on 100 untouched tuning stays raises factual NLL "
            "by +0.00617, exceeding the budget on this cohort. Pooled 299-stay "
            f"delta {mean184:.6f}, cluster SE {se184:.6f}. "
            "Do not strengthen current gains without improving generalization or "
            "demonstrating a safer candidate. Weights unchanged during audit.\n")
```