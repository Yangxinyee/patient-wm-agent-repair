The refits trade factual-state matching against dose-contrast preservation; none is validated yet. The generated baseline is not near a glucose ceiling, so ceiling saturation is unlikely to explain the weak half-dose effect. I’ll install the moderate-penalty refit reversibly and test factual loss first. It is acceptable only if the loss increase stays within 0.005 nats.

```python
import json
import hashlib

print("=== CURRENT VERSUS REFITTED LOCAL RESPONSES ===")
comparison221 = {}
for label221, cand221 in [
    ("current", candidate206),
    ("refit_penalty_1", refit220[1.]),
    ("refit_penalty_10", refit220[10.])
]:
    baseE221 = prelude155(tensor153(tokens206)[i169], overrides=cand221)[0]
    bp221 = predict_E219(baseE221)
    row221 = {
        "baseline_minus_original": (bp221 - orig_pred219).mean(axis=0).tolist()
    }
    for key221 in [(0, .5), (0, 2.), (1, .5), (1, 2.)]:
        ep221 = prelude155(tensor153(r214_tokens[key221])[i169],
                           overrides=cand221)[0]
        row221[str(key221)] = (predict_E219(ep221) - bp221).mean(axis=0).tolist()
    comparison221[label221] = row221
    print(label221)
    for key221, value221 in row221.items():
        print(" ", key221, np.round(value221, 5).tolist())

checkpoint221 = {name221: tensor153(load(name221)).clone()
                 for name221 in edited()}
candidate221 = {name221: tensor153(value221).clone()
                for name221, value221 in refit220[1.].items()}
for name221, value221 in candidate221.items():
    assert tuple(value221.shape) == tuple(SHAPES[name221])
    assert torch.isfinite(value221).all()
for name221, value221 in candidate221.items():
    save_edit(name221, value221)

hash221 = hashlib.sha256()
for name221 in sorted(candidate221):
    hash221.update(name221.encode())
    hash221.update(arr183(candidate221[name221]).tobytes())
print("\nINSTALLED REFIT", hash221.hexdigest()[:12])
ids221 = sorted(int(x) for x in r213_join["stay_id"].unique())
assert 0 < len(ids221) <= 100
print("NEXT FACTUAL REQUEST")
print(json.dumps({"type": "factual", "model": "both",
                  "split": "tuning", "stays": ids221}))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 221; forwards 14. Generated baseline is not near a glucose ceiling. Installed reversible current-feature compensation refit with paired-dose contrast penalty 1; checkpoint221 retains the prior candidate206. Edit hash " + hash221.hexdigest()[:12] + ". Next: authoritative factual test on the same tuning stays; accept only <=0.005 nats increase before rollout testing.\n")
```