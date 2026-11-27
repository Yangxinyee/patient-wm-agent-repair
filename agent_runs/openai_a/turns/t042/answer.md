The output confirms linear, clipped normalization and an eight-field register, but it does not test the reconstructed predictions. I’ll now compare register scaling and activation choices numerically. A substantial improvement would support an input-layout issue; unchanged errors would point elsewhere.

```python
torch.set_num_threads(8)
import torch.nn.functional as F

head_parts_tune = cpu_head_parts(reference_meta, state_files[3])
head_q = ln(
    head_parts_tune["base"]
    + lin(head_parts_tune["prior"][:, :32], "z_head")
    + W["type_emb.weight"][1],
    "query_norm"
)
last_z_tensor = torch.tensor(reference_meta.last_z.to_numpy(), dtype=torch.float32)
has_last_tensor = torch.tensor(reference_meta.has_last.to_numpy(), dtype=torch.float32)
diagnostic_results = []
for activation_name, activation in [
    ("gelu", F.gelu), ("silu", F.silu), ("relu", F.relu)
]:
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        wc = (head_q + head_parts_tune["additive"]
              + (scale - 1) * last_z_tensor[:, None]
              * W["last_numeric.weight"][:, 2][None, :])
        hidden = activation(lin(wc, "attr_heads.value_bin.0"))
        logits = lin(hidden, "attr_heads.value_bin.2") + head_parts_tune["transition"]
        probs = logits.softmax(-1)
        means = (probs * head_centres).sum(-1).numpy()
        nll = -probs[torch.arange(len(probs)), head_obsbins].clamp_min(1e-20).log().numpy()
        err = means - head_gpu_mean
        diagnostic_results.append({
            "activation": activation_name, "z_scale": scale,
            "rmse": float(np.sqrt(np.mean(err**2))),
            "nll": float(nll.mean()),
            "glucose_rmse": float(np.sqrt(np.mean(
                err[reference_meta.event.eq("glucose_finger").to_numpy()]**2))),
            "map_rmse": float(np.sqrt(np.mean(
                err[reference_meta.event.eq("map_arterial").to_numpy()]**2)))
        })
print("=== INPUT/ACTIVATION CHECK ===")
print(pd.DataFrame(diagnostic_results).sort_values("rmse").head(9).to_string(index=False))
print("last_numeric column norms:",
      np.round(torch.linalg.vector_norm(W["last_numeric.weight"], dim=0).numpy(), 3))
print("\n=== PREVIOUS CAUSAL CHECKS: TEXT CONCLUSIONS ONLY ===")
for title in ["## Factual baseline", "## Sensor-contrast", "## Dynamic identification",
              "## Representation inspection", "## Kernel recovery"]:
    start = notes_text.find(title)
    if start >= 0:
        end = notes_text.find("\n## ", start + 4)
        segment = notes_text[start:end if end >= 0 else None]
        print("\n".join(line for line in segment.splitlines()
                        if line.startswith(("##", "- "))))
print("\n=== GPU BASELINE ===")
print("factual NLL:", float(factual_baseline.nll.mean()))
print(dose_baseline.groupby("edit")["shift"].mean().to_string())
with open("/work/notes.md", "a") as f:
    f.write("\n## Head recovery after Python 36\n"
            "- Current normalization matches vocab: (x-median)/iqr, clipped [-10,10]. Actual last_numeric has 8 inputs.\n"
            "- Retained objects: state_files[3] = original 100 tuning states, state_files[4] = original 100 train states;\n"
            "  reference_meta/head_gpu_mean/head_gpu_nll/head_centres/head_obsbins align with 4751 tuning readings.\n"
            "- cpu_head_parts returns base/prior/additive/transition; cpu_head_probs evaluates them.\n"
            "- P (30000,26,6), TP (400,26,6); event_col gives columns; train_ids/tuning_ids retained.\n"
            "- Original rollout: insulin x2 -1.847, half +0.777; NE x2 -0.498, half -0.600.\n"
            "- Insulin RD beta -1.929, rho .83184; sustained training x2 estimate -11.136, first100 tuning -10.708.\n"
            "- No edits. Testing CPU head input scale/activation before a targeted edit.\n")
    f.write(json.dumps(sorted(diagnostic_results, key=lambda x: x["rmse"])[:3]) + "\n")
```
