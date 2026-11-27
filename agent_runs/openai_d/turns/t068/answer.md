The audit stopped before producing a result because I used the wrong tensor prefix. It also revealed a different cached type embedding than an earlier assumption. I’ll correct the tensor name, rerun the latent-noise test, and measure sensitivity to the type embedding; either could explain why the local checks missed the failure.

```python
q_draws = []
q_alt_draws = []
for draw in range(len(eps_t)):
    zzcheck = mu_t + sd_t*eps_t[draw]
    viewcheck = base_t + rc_lin(zzcheck, "z_head")
    q_draws.append(rc_norm(viewcheck + const_t, "query_norm"))
    q_alt_draws.append(rc_norm(viewcheck + type_t[1], "query_norm"))
q_draws = torch.stack(q_draws)
q_alt_draws = torch.stack(q_alt_draws)
q_mean_check = q_draws.mean(0)
wc_draws = as_t(repair_wc_mean)[None] + q_draws-q_mean_check[None]
wc_alt_draws = wc_draws + q_alt_draws-q_draws

mc_code_gate = (wc_draws @ as_t(rec_gate_v) + float(rec_gate_b)).numpy()
mc_dose_signal = (wc_draws @ as_t(trial_dv) + float(trial_db)).numpy()
mc_pre = wc_draws @ rW1[replaced_rows].T + rb1[replaced_rows]
mc_correction = F.gelu(mc_pre) @ rW2[:, replaced_rows].T

print("=== PER-DRAW VARIABILITY ===")
for ec in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    maskc = head_meta.event.to_numpy() == ec
    print(ec,
          "gate 1/50/99 percentiles",
          np.round(np.quantile(mc_code_gate[:,maskc],[.01,.5,.99]),3).tolist(),
          "dose within-row SD 10/50/90 percentiles",
          np.round(np.quantile(mc_dose_signal[:,maskc].std(0),[.1,.5,.9]),3).tolist(),
          "correction RMS logits 10/50/90/99 percentiles",
          np.round(np.quantile(torch.sqrt(mc_correction[:,maskc].square().mean(-1)).numpy(),
                               [.1,.5,.9,.99]),3).tolist())

prob_o = torch.zeros((len(head_meta),17))
prob_r = torch.zeros_like(prob_o)
prob_alt_o = torch.zeros_like(prob_o)
for draw in range(len(wc_draws)):
    inp = wc_draws[draw]
    lo = F.gelu(inp@oW1.T+ob1)@oW2.T+ob2+as_t(wr_trans)
    lr = F.gelu(inp@rW1.T+rb1)@rW2.T+rb2+as_t(wr_trans)
    alt = wc_alt_draws[draw]
    la = F.gelu(alt@oW1.T+ob1)@oW2.T+ob2+as_t(wr_trans)
    prob_o += torch.softmax(lo,-1)/len(wc_draws)
    prob_r += torch.softmax(lr,-1)/len(wc_draws)
    prob_alt_o += torch.softmax(la,-1)/len(wc_draws)

mo = (prob_o*as_t(wr_centres)).sum(-1).numpy()
mr = (prob_r*as_t(wr_centres)).sum(-1).numpy()
ma = (prob_alt_o*as_t(wr_centres)).sum(-1).numpy()
print("\nOriginal MC reconstruction RMS:",
      np.sqrt(np.mean((mo-head_meta.tool_mean.to_numpy())**2)))
print("Alternative type original reconstruction RMS:",
      np.sqrt(np.mean((ma-head_meta.tool_mean.to_numpy())**2)))
for ec in head_meta.event.unique():
    maskc = head_meta.event.to_numpy() == ec
    print(ec, "edited mean shift 0/50/100 percentiles",
          np.round(np.quantile((mr-mo)[maskc],[0,.5,1]),3).tolist(),
          "type change in decoder mean:",
          float(((wc_alt_draws-wc_draws)@as_t(trial_dv)).mean(0)[maskc].mean()))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 59–60: corrected latent audit\n"
            "Run 59 stopped at an incorrect tensor prefix; z_head is top-level. "
            "Run 60 tests individual-draw sensitivity and the type-embedding mismatch. "
            "No weights edited.\n")
```