The enriched baseline still shows weak responses: −1.77 and +0.82 for insulin, and +0.05 and −0.48 for norepinephrine. I’ll first test whether the value head’s hidden features encode dose innovations. If they do, a low-rank value-head edit could be safer than changing the recurrent state, because teacher-forced predictions from the other heads would remain unchanged.

```python
print("=== ALIGNMENT AND RETAINED HEAD RECONSTRUCTION ===")
print(head_records.head(3).to_string(index=False))
print(aligned_ref.head(3).to_string(index=False))
print("HM rows:", HM[:5].tolist())
print("best register:", register_search[:3])
print("best view:", view_search[:3])
print("num_best:", num_best[:3].numpy().tolist())
print("repair index:", repair_s[:18].tolist(), repair_j[:18].tolist())
print("repair_lp summaries:")
for key, value in repair_lp.items():
    print(key, {k: (v.shape if isinstance(v, np.ndarray) else
                    repr(v)[:900]) for k, v in value.items()})
print("representative training IDs:", repair_state_ids_1)

# Cache a deterministic reconstruction for the already aligned 4,745 readings.
TF = torch.nn.functional
repair_base = hx[:, :192] + norm(hflow.mean(1), "core.out_norm")
repair_gap = mlp((torch.log(hdt.clamp_min(1.0))/10)[:, None], "gap", "silu")
repair_context = repair_base + repair_gap
repair_prior = mlp(repair_context, "z_prior", "gelu")[:, :32]
repair_view = repair_context + lin(repair_prior, "z_head")
repair_q = norm(repair_view + W["type_emb.weight"][1], "query_norm")
repair_num = torch.zeros((len(HM), 8))
repair_num[:, 0] = torch.tensor(HM[:, 3], dtype=torch.float32)
repair_num[:, 3] = torch.log1p(torch.tensor(HM[:, 4], dtype=torch.float32))/10
repair_num[:, 6] = torch.tensor(HM[:, 5], dtype=torch.float32)
repair_register = (lin(repair_num, "last_numeric") +
                   W["embed.bins.value.weight"][hlastbin] +
                   W["embed.bins.rate.weight"][0] +
                   W["last_status.weight"][0])
repair_withcode = repair_q + W["embed.code.weight"][hcode] + repair_register
repair_phi = TF.relu(lin(repair_withcode, "attr_heads.value_bin.0"))
repair_logits = (lin(repair_phi, "attr_heads.value_bin.2") +
                 W["embed_transition.value_bin.weight"][hcode*17 + hlastbin])
repair_probs = torch.softmax(repair_logits, -1)
repair_predmean = (repair_probs*hcentres).sum(-1).numpy()
repair_prednll = -torch.log(
    repair_probs[torch.arange(len(HM)), htruebin].clamp_min(1e-12)).numpy()
print("Reconstruction RMSE mean/NLL:",
      float(np.sqrt(np.mean((repair_predmean-refmean)**2))),
      float(np.sqrt(np.mean((repair_prednll-refnll)**2))))
print("Reconstructed/original NLL:", repair_prednll.mean(), refnll.mean())

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 5 / Run 24\nEnriched original tuning rollout: insulin x2=-1.773, "
            "half=+0.816 (665 exposed hours); NE x2=+0.048, half=-0.481 "
            "(571 exposed hours). Exploring innovation-decoding in value-head hidden features "
            "to preserve teacher-forced non-value outputs exactly. No edits.\n"
            "Counts: 24 Python runs, 5 forward passes.\n")
```
