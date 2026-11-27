The raw mean inputs also give the intended code gate, so simple projection or scaling errors look unlikely. I’ll now evaluate the replacement neurons on individual latent draws rather than their average. If that reproduces the factual damage, the decoder is too noise-sensitive; otherwise, the reconstructed inputs still differ from the real head’s.

```python
torch.set_num_threads(8)

print("=== QUERY COMPONENTS ===")
print([n for n in NAMES if "z_head" in n or "query_norm" in n])
const_t = as_t(wr_const)
type_t = as_t(W0["type_emb.weight"])
print("Nearest type embedding:", int(torch.argmin(
    torch.linalg.vector_norm(type_t-const_t[None], dim=1))))

base_t = as_t(wr_flowbase) + as_t(wr_gap)
mu_t, sd_t = as_t(wr_mu), as_t(wr_logsd).exp()
eps_t = as_t(wr_eps)
q_draws = []
for draw in range(len(eps_t)):
    zzcheck = mu_t + sd_t*eps_t[draw]
    qqcheck = rc_norm(base_t + rc_lin(zzcheck, "core.z_head") + const_t,
                      "query_norm")
    q_draws.append(qqcheck)
q_draws = torch.stack(q_draws)
q_mean_check = q_draws.mean(0)
wc_draws = as_t(repair_wc_mean)[None] + q_draws-q_mean_check[None]

mc_code_gate = (wc_draws @ as_t(rec_gate_v) + float(rec_gate_b)).numpy()
mc_dose_signal = (wc_draws @ as_t(trial_dv) + float(trial_db)).numpy()
mc_pre = wc_draws @ rW1[replaced_rows].T + rb1[replaced_rows]
mc_correction = F.gelu(mc_pre) @ rW2[:, replaced_rows].T
print("\n=== PER-DRAW VARIABILITY ===")
for ec in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    maskc = head_meta.event.to_numpy() == ec
    print(ec,
          "gate quantiles", np.round(np.quantile(mc_code_gate[:,maskc],
                                                 [.01,.5,.99]),3).tolist(),
          "dose-signal within-row SD quantiles",
          np.round(np.quantile(mc_dose_signal[:,maskc].std(0),
                               [.1,.5,.9]),3).tolist(),
          "correction logit RMS quantiles",
          np.round(np.quantile(torch.sqrt(mc_correction[:,maskc].square().mean(-1)).numpy(),
                               [.1,.5,.9,.99]),3).tolist())

prob_o = torch.zeros((len(head_meta),17))
prob_r = torch.zeros_like(prob_o)
for draw in range(len(wc_draws)):
    inp = wc_draws[draw]
    lo = F.gelu(inp@oW1.T+ob1)@oW2.T+ob2+as_t(wr_trans)
    lr = F.gelu(inp@rW1.T+rb1)@rW2.T+rb2+as_t(wr_trans)
    prob_o += torch.softmax(lo,-1)/len(wc_draws)
    prob_r += torch.softmax(lr,-1)/len(wc_draws)
print("\nOriginal MC mean reconstruction RMS:",
      float(torch.sqrt(torch.mean(
          ((prob_o*as_t(wr_centres)).sum(-1)
           -as_t(head_meta.tool_mean.to_numpy()))**2))))
print("Edited MC mean shift quantiles:",
      np.round(np.quantile(((prob_r-prob_o)*as_t(wr_centres)).sum(-1).numpy(),
                           [0,.1,.5,.9,1]),3).tolist())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 59: latent-draw audit\n"
            "Projection omission was refuted. Testing the rejected edit on individual "
            "latent draws while keeping their mean with-code input fixed, to distinguish "
            "latent noise amplification from inaccurate reconstructed inputs. No edits.\n")
```