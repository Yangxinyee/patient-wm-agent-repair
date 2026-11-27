The threshold analysis gives a tuning insulin target of −10.71 for doubling and +5.35 for halving. The current edit is too weak but has factual headroom, so I’ll preserve it and test twice the contrast strength. This remains a hypothesis: stronger rollouts would help only if GPU factual NLL still stays within 0.005.

```python
print("=== CONTRAST CONSTRUCTION PARAMETERS ===")
for k in ["sc_nodes", "sc_gate_b", "sc_rank", "sc_bias_comp", "sc_names"]:
    if k in globals():
        print(k, globals()[k])
print("gate weight norm:", float(np.linalg.norm(sc_gate_w)))
print("=== COMPACT GRAFT BYTECODE ===")
for ins in dis.Bytecode(make_contrast_graft):
    if ins.opname not in {"RESUME", "CACHE", "PRECALL", "PUSH_NULL"}:
        print(ins.opname, ins.argrepr)

gpu13_checkpoint = {n: load(n).clone() for n in edited()}
scale_trial_checks = {}
for scale in [1.0, 1.5, 2.0]:
    checks = {}
    for label, cache in sc_fact.items():
        ev = sc_evaluate(cache, trial_r3["w"], trial_r3["b"], scale=scale)
        checks[label] = ev["nll"] - cache["original_nll"]
    scale_trial_checks[scale] = checks
    print("scale", scale, "cached factual NLL deltas", checks)

trial_scale = 2.0
trial_contrast_weights = make_contrast_graft(
    trial_r3["w"], trial_r3["b"], scale=trial_scale)
for n, array in trial_contrast_weights.items():
    save_edit(n, torch.as_tensor(array, dtype=torch.float32))
print("Installed reversible contrast-strength trial:", trial_scale)
print("Edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Contrast-strength trial; Python 183, forwards 13/50\n"
            "- Recovered strong insulin RD evidence: beta=-1.928709, rho=0.831840; "
            "training x2 target -11.136, first-100 tuning target -10.708, half +5.354.\n"
            "- NE flexible estimates approximately +3.3 remain observational; "
            "do not treat their agreement as identification.\n"
            "- Preserved GPU13 scale=1 checkpoint. Installed scale=2 trial through "
            "the same four value-head tensors; NOT accepted pending GPU factual/rollout checks.\n"
            f"- Cached scale checks: {scale_trial_checks}\n")
```
