The record-based estimates are stable across dose ranges: roughly −2 glucose units per insulin-rate unit at the next hour, and +3.3 MAP units per norepinephrine-rate unit. Longer effects remain less certain. I’ll now measure what the original head already captures and whether its features can decode the missing dose-innovation response.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

torch.set_num_threads(8)
wr_meta = head_meta.copy().reset_index(drop=True)
wr_x = torch.as_tensor(
    states0["x"][wr_meta.state_idx.to_numpy()], dtype=torch.float32)
wr_out = wr_x[:, :192]
wr_h = wr_x[:, 192:960].reshape(-1, 4, 192)
wr_dt = torch.tensor(wr_meta.dt.to_numpy(), dtype=torch.float32)[:, None]
wr_codes = torch.tensor(
    [code_ids[e] for e in wr_meta.event], dtype=torch.long)
wr_lb = torch.tensor(wr_meta.last_bin.to_numpy(), dtype=torch.long)

wr_const_rows = (rh_fixed
    - W0["embed.code.weight"][rh_codes]
    - W0["embed.bins.value.weight"][rh_lb])
wr_const = wr_const_rows.mean(0)
print("Fixed-register constant reconstruction max error:",
      float((wr_const_rows - wr_const).abs().max()))
print("Normalisation example:", rh_norm[("glucose_finger", "value")])
print("Panel index:", wide.index.names, wide.index[:3].tolist())

wr_nf = torch.zeros((len(wr_meta), 8))
wr_nf[:, 0] = torch.tensor(wr_meta.has_last.to_numpy(), dtype=torch.float32)
wr_nf[:, 1] = torch.log1p(torch.tensor(
    wr_meta.age.to_numpy() / 3600, dtype=torch.float32)) / 10
wr_nf[:, 3] = wr_nf[:, 0]
wr_nf[:, 6] = torch.tensor(wr_meta.last_z.to_numpy(), dtype=torch.float32)
wr_fixed = (W0["embed.code.weight"][wr_codes]
            + W0["embed.bins.value.weight"][wr_lb] + wr_const
            + rc_lin(wr_nf, "last_numeric"))
wr_centres = torch.tensor(
    [[0.0] + list(VOCAB["value_bin_centres"][e]) for e in wr_meta.event],
    dtype=torch.float32)
wr_trans = W0["embed_transition.value_bin.weight"][wr_codes * 17 + wr_lb]
wr_flow = wr_h * torch.exp(
    -wr_dt[:, None, :] * torch.exp(W0["core.flow.a"])[None, None, :])
wr_gap = rc_mlp(torch.log1p(wr_dt / 3600) / 10, "gap", "gelu")
wr_base = wr_out + rc_norm(wr_flow.mean(1), "core.out_norm") + wr_gap
wr_prior = rc_mlp(wr_base, "z_prior", "gelu")
wr_mu, wr_logsd = wr_prior[:, :32], wr_prior[:, 32:].clamp(-7, 2)
wr_gen = torch.Generator().manual_seed(813)
wr_eps_half = torch.randn((16, len(wr_meta), 32), generator=wr_gen)
wr_eps = torch.cat([wr_eps_half, -wr_eps_half], 0)
wr_phi = torch.zeros((len(wr_meta), 192))
wr_p = torch.zeros((len(wr_meta), 17))
wr_gain = torch.zeros(len(wr_meta))
wr_score = torch.arange(17, dtype=torch.float32)
wr_score[0] = 8.0
for j in range(len(wr_eps)):
    zj = wr_mu + wr_logsd.exp() * wr_eps[j]
    qj = rc_norm(wr_base + rc_lin(zj, "z_head")
                 + W0["type_emb.weight"][1], "query_norm")
    phij = rc_act(rc_lin(qj + wr_fixed, "attr_heads.value_bin.0"), "gelu")
    pj = torch.softmax(rc_lin(phij, "attr_heads.value_bin.2")
                       + wr_trans, -1)
    mj = (pj * wr_centres).sum(1)
    wr_phi += phij / len(wr_eps)
    wr_p += pj / len(wr_eps)
    wr_gain += ((pj * wr_centres * wr_score).sum(1)
                - mj * (pj * wr_score).sum(1)) / len(wr_eps)
wr_mean = (wr_p * wr_centres).sum(1).numpy()
print("Full-sample local mean RMSE versus tool:",
      float(np.sqrt(np.mean((wr_mean - wr_meta.tool_mean.to_numpy())**2))))

wr_instruments = {}
wr_dose_index = pd.MultiIndex.from_arrays(
    [wr_meta.stay_id.to_numpy(), wr_meta.hour.to_numpy() - 1],
    names=wide.index.names)
for drug, event in [("insulin", "glucose_finger"),
                    ("norepinephrine", "map_arterial")]:
    dc = dml_cache[drug]
    ds = pd.Series(dc["rd"], index=wide.index[dc["inds"]])
    rd_aligned = ds.reindex(wr_dose_index).to_numpy()
    valid = (wr_meta.event.to_numpy() == event) & np.isfinite(rd_aligned)
    ii_valid = np.flatnonzero(valid)
    rr = rd_aligned[valid]
    yy = wr_meta.observed.to_numpy()[valid]
    pp = wr_meta.tool_mean.to_numpy()[valid]
    gg = wr_meta.stay_id.to_numpy()[valid]
    print("\n", drug, "paired readings:", len(rr),
          "innovation SD:", round(float(rr.std()), 4))
    print("Record slope:", residual_effect(yy, rr, gg))
    print("Original-mean slope:", residual_effect(pp, rr, gg))
    print("Missing-response slope:", residual_effect(yy - pp, rr, gg))

    ff = wr_phi.numpy()[valid]
    scale = ff.std(0).clip(0.1)
    fs = (ff - ff.mean(0)) / scale
    folds_local = gg % 5
    for alpha in [10.0, 100.0, 1000.0]:
        pred = np.zeros(len(rr))
        for fold in range(5):
            tr = folds_local != fold
            te = ~tr
            reg = Ridge(alpha=alpha).fit(fs[tr], rr[tr])
            pred[te] = reg.predict(fs[te])
        print("Decoder ridge", alpha, "held-out R2",
              round(float(r2_score(rr, pred)), 4),
              "SD", round(float(pred.std()), 4))
    wr_instruments[drug] = {"valid": valid, "rd": rd_aligned,
                            "indices": ii_valid}
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 27–31: recovery and head-instrument diagnostics\n"
            "Recovered architecture, aligned states and DML caches. "
            "No edits. Building antithetic latent-averaged reading-head features "
            "and measuring residual-dose response plus held-out innovation decodability.\n")
```
