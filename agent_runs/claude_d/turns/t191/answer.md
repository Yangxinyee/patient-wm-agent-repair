The four cheapest units cost +0.0003 NLL when ablated together. Their individual savings do not add, but the cost is small enough to accept.

A linear MAP-vs-glucose gate on with_code separates the families with some margin, though not a large one under z noise:

- MAP rows: minimum 0.15, 1st percentile 0.56;
- glucose rows: maximum -0.19, 99th percentile -0.56.

With a gate multiplier of about 40, every MAP row sits above +5.9 (linear GELU regime) and every glucose row below -7.6 (silenced). The reference unit cancels the gate term exactly.

The two insulin-surprise definitions are identical where both exist (r = 1.0). Their lag-1 deficits differ only because one version dropped 750 rows, so I will use about -1.4 and let the rollout calibration decide.

Rough budget at full counterfactual scaling:

| part | NLL cost |
|---|---|
| unit repurposing | +0.0003 |
| insulin, lags 1 and 2 | +0.0023 |
| NE, lag 1 | +0.008 |

So NE will probably need partial scaling (about half the response for about +0.001). Next I look at the bin-centre table, so I can fit per-family logit tilt vectors that give a uniform mean shift.

```python
vc = VOCAB['value_bin_centres']
print(type(vc).__name__, (list(vc.keys()) if isinstance(vc, dict) else len(vc)))
print(VOCAB['value_bin_note'][:300])
k0 = list(vc.keys())[0] if isinstance(vc, dict) else 0
print(np.round(np.array(vc[k0]),1) if isinstance(vc, dict) else np.round(np.array(vc[0]),1))
print('cen rows equal within code?', {c: int(len(np.unique(np.round(cen[evc==c],3),axis=0))) for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']})
```