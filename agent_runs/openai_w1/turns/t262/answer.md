The rejected edit used very large output weights, which could amplify small factual carrier errors. The cached record estimates suggest delayed insulin-negative and norepinephrine-positive effects, but those estimates still depend on the adjustment assumptions. I’ll test one-tenth of the added output pathway. If factual loss falls sufficiently and dose effects remain useful, this route may be salvageable; otherwise I’ll abandon it.

```python
print("RECORD ESTIMATES")
for x in causal_estimates:
    print(str(x)[:450])
print("\nASSOCIATIONS")
for x in association_results:
    print(str(x)[:350])

baseline232 = {n: arr231(v).copy() for n, v in candidate228_baseline.items()}
print("\nOutput baseline consistency:",
      np.linalg.norm(baseline232["core.prelude.0.attn.out.weight"] - candidate228_out0))
scale232 = 0.10
for n, v in rejected230.items():
    a = arr231(v).copy()
    if n == "core.prelude.0.attn.out.weight":
        b = baseline232[n]
        a = b + scale232 * (a - b)
    save_edit(n, torch.as_tensor(a, dtype=torch.float32))
print("Installed output-path scale:", scale232)
print("Factual test stays:", [int(s) for s in candidate229_stays])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 232\nInstalled reversible marker-output shrinkage t=.10 from "
            "rejected229 toward validated relay baseline, retaining marker inputs. "
            "Large output norms may amplify small factual leakage; GPU factual and rollout tests "
            "will determine whether shrinking is viable. Counts: 232 Python runs, 21 forwards.\n")
```