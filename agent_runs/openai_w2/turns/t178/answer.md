The architecture confirms that the current edit changes recurrent processing of dose information. The output was truncated before the statistical evidence, so I’ll retrieve that evidence separately and pool the two factual tests. Stable, pre-dose-adjusted estimates would support the proposed causal targets; failed timing or confounding checks would weaken them.

```python
notes154 = Path("/work/notes.md").read_text()
start154 = notes154.find("## Run 3\n")
stop154 = notes154.find("## Restart, run 13")
evidence154 = notes154[start154:stop154]
print("=== EARLY CAUSAL EVIDENCE ===")
print(evidence154[:6500])
print("Evidence characters:", len(evidence154))

paired154 = []
for k154 in (16, 18):
    o154 = pd.read_csv(f"/work/fwd_{k154}/factual_original.csv")
    e154 = pd.read_csv(f"/work/fwd_{k154}/factual_edited.csv")
    print("\nForward", k154, "columns:", o154.columns.tolist())
    assert len(o154) == len(e154)
    for c154 in ("stay_id", "hour", "event"):
        if c154 in o154:
            assert o154[c154].equals(e154[c154]), (k154, c154)
    nll_candidates154 = [c for c in o154.columns if "nll" in c.lower()]
    nll_col154 = "nll" if "nll" in o154 else nll_candidates154[0]
    p154 = o154.copy()
    p154["delta_nll"] = e154[nll_col154].to_numpy() - o154[nll_col154].to_numpy()
    p154["edited_nll"] = e154[nll_col154].to_numpy()
    p154["forward"] = k154
    paired154.append(p154)
    print("NLL original/edited/difference:",
          o154[nll_col154].mean(), e154[nll_col154].mean(), p154.delta_nll.mean())

pooled154 = pd.concat(paired154, ignore_index=True)
cluster154 = pooled154.groupby("stay_id").delta_nll.agg(["sum", "size"])
mean154 = pooled154.delta_nll.mean()
influence154 = cluster154["sum"] - mean154 * cluster154["size"]
se154 = np.sqrt(len(cluster154) / (len(cluster154)-1) * np.sum(influence154**2)) / len(pooled154)
print("\n=== POOLED FACTUAL VALIDATION ===")
print("Stays/readings:", len(cluster154), len(pooled154))
print("NLL increase:", mean154, "stay-cluster SE:", se154,
      "approximate upper 95% bound:", mean154 + 1.96*se154)
print(pooled154.groupby("event").delta_nll.agg(["size", "mean"]).to_string())

with open("/work/notes.md", "a") as f154:
    f154.write(
        f"\n## Run 154: pooled factual validation\n"
        f"Initial-safe [-10,+12] candidate across {len(cluster154)} tuning stays "
        f"and {len(pooled154)} readings: paired NLL increase {mean154:.6f}, "
        f"stay-cluster SE {se154:.6f}. Reviewing earlier causal identification "
        f"and persistence evidence before further changes.\n"
    )
```
