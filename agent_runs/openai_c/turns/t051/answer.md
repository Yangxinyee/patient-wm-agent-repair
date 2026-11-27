The head pilot passes the factual limit (+0.00170 nats) but adds almost no dose response, so I’ll remove it. I’ll test routing the drug’s numeric dose signal through the better-trained numeric-value embedding; measurement inputs remain unchanged. A larger rollout response with little factual damage would support this pathway, while unchanged or wrong-signed responses would reject it.

```python
clear_ops()
replay_ops(repair_candidates["numeric_gain4"])

numeric_mix_k = -2.0
numeric_mix_basis = np.eye(6)[:, [0, 2, 4]]
numeric_mix_transform = np.eye(3)
numeric_mix_transform[0, 1:] = numeric_mix_k / np.sqrt(2.0)
apply_lowrank_right_transform(
    "embed.numeric.weight", numeric_mix_basis, numeric_mix_transform
)
repair_candidates["numeric_value_mix_m2"] = freeze_ops(ops())

print("Applied dose/rate-to-value numeric mixing:", numeric_mix_k)
print("Operation count:", len(ops()))
print("Numeric normalization entries for drugs:")
numeric_entries = VOCAB["numeric"]
if isinstance(numeric_entries, list):
    print([x for x in numeric_entries
           if str(x.get("code", "")) in {"insulin", "norepinephrine"}])
else:
    print(str(numeric_entries)[:2200])

for n in [9, 16]:
    p = Path(f"/work/fwd_{n}/summary.json")
    if p.exists():
        j = json.loads(p.read_text())
        print("factual", n, {
            m: r.get("value_nll_per_value")
            for m, r in j.get("results", {}).items()
        })

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 34: numeric-value mixing pilot\n"
            "- Head NE pilot factual NLL 1.636318 vs original 1.634615 (+0.001703), but rollout benefit negligible; removed.\n"
            "- Retained numeric gain 4 and added a drug-dose/rate-to-numeric-value input shear with coefficient -2.\n"
            "- Measurement numeric inputs are unchanged; signs and factual safety require forward validation.\n"
            "- Candidate saved as numeric_value_mix_m2. Python runs: 34; forward passes: 17.\n")
```
