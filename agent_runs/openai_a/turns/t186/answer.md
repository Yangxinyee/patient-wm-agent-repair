Explicit register protection still produced no candidate with both correct local signs and acceptable factual fit. Increasing the gate strength did not help. I’ll now measure the correction actually reaching each event’s logits, rather than its raw decoder score, to determine whether this is a calibration problem or a genuine sensitivity–fit tradeoff.

```python
print("=== EVALUATOR OPERATIONS, GROUPED BY SOURCE LINE ===")
line = None
parts = []
for ins in dis.get_instructions(sc_evaluate):
    if ins.starts_line is not None and parts:
        print(line, " ; ".join(parts)[:290])
        parts = []
    if ins.starts_line is not None:
        line = ins.starts_line
    if ins.opname not in ("CACHE", "RESUME"):
        parts.append(ins.opname + (" " + ins.argrepr if ins.argrepr else ""))
if parts:
    print(line, " ; ".join(parts)[:290])

print("\n=== ACTUAL SIGNAL AND EVENT-SPECIFIC NLL ===")
cache = sc_fact["tuning"]
x = torch.from_numpy(cache["x"])
gate = torch.as_tensor(cache["gate"])
base_logits = torch.as_tensor(cache["logits"])
bins = torch.as_tensor(cache["bins"], dtype=torch.long)
meta = sc_meta["tuning"]
original_p = torch.softmax(base_logits, -1).mean(0)
baseline_nll = -torch.log(original_p[torch.arange(len(bins)), bins].clamp_min(1e-30)).numpy()

for label, w, b, scale in (
    ("retained", sc_old_w, sc_old_b, sc_old_scale),
    ("register_lambda_0.1", r2_candidates[(.1, 1.)]["w"], r2_candidates[(.1, 1.)]["b"], 1.),
    ("register_lambda_1", r2_candidates[(1., 1.)]["w"], r2_candidates[(1., 1.)]["b"], 1.),
):
    raw = x @ torch.as_tensor(w, dtype=torch.float32) + float(b)
    applied = torch.nn.functional.gelu(raw + gate) - torch.nn.functional.gelu(gate)
    p = torch.softmax(base_logits + float(scale)*applied[...,None]*torch.as_tensor(sc_rank), -1).mean(0)
    nll = -torch.log(p[torch.arange(len(bins)), bins].clamp_min(1e-30)).numpy()
    check = sc_evaluate(cache, w, b, scale)
    print(label, "reconstructed NLL", float(nll.mean()),
          "helper NLL", check["nll"])
    for event in meta.event.unique():
        ix = np.asarray(meta.event == event)
        a = applied[:,ix].numpy()
        g = gate[:,ix].numpy()
        r = raw[:,ix].numpy()
        print(event, "n", int(ix.sum()),
              "NLL_delta_vs_base", round(float((nll-baseline_nll)[ix].mean()),6),
              "gate_q", np.round(np.quantile(g,[.01,.5,.99]),2),
              "raw_q", np.round(np.quantile(r,[.01,.5,.99]),3),
              "applied_q", np.round(np.quantile(a,[.01,.5,.99]),3),
              "active_fraction", round(float(np.mean(np.abs(a)>1e-4)),4))
```
