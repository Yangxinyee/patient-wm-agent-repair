The intercept is the first coefficient; that convention recovers the small factual residuals. I’ll use the corrected conversion and measure how changes at the direct recurrent input affect glucose and MAP predictions. Accurate reconstruction and consistent sensitivity across bundles would support a low-rank edit; weak or erratic sensitivity would refute it.

```python
import time
import torch

raw_fits119 = {}
for fam119 in ["ffn", "E"]:
    fit119 = prelude_fits112[fam119][0.3]
    w119 = fit119["B"][1:] / fit119["sd"][:, None]
    b119 = fit119["B"][0] - fit119["mu"] @ w119
    raw_fits119[fam119] = (w119, b119)
w119, b119 = raw_fits119["E"]
E_linear_dose119 = w119 + np.outer(constant_vec117 / constant117, b119)
print("Correct E conversion error:",
      np.max(np.abs(prelude_E111 @ E_linear_dose119 -
                    (prelude_E111 @ w119 + b119))))

best_core119 = min(core_search104, key=lambda r: r["h_rmse"])
print("Best recurrent convention:", best_core119)
mem119 = memory104(include_current=best_core119["include_current"])
heads119 = best_core119["heads"]
age119 = best_core119["age_mode"]
h119 = repair_hflow101[core_pick104]
e119 = repair_e101[core_pick104]
u119 = repair_u101[core_pick104]
z119 = repair_z101[core_pick104]
tok119 = torch.cat([
    h119 + Wrepair["core.slots"],
    e119[:, None], u119[:, None], lin104(z119, "z_in")[:, None]
], dim=1)
E119 = norm104(layer104(tok119, "core.prelude.0", heads119)[:, :4],
              "core.prelude_norm")
base119 = h119 + E119
rec119 = base119
for pref119 in ["core.blocks.0.0", "core.blocks.0.1"]:
    rec119 = layer104(rec119, pref119, heads119,
                     memory=mem119[0], memory_mask=mem119[1],
                     memory_age=mem119[2], age_mode=age119)
delta119 = F104.softplus(Wrepair["core.delta_in"])
abar119 = torch.exp(-delta119 * torch.exp(Wrepair["core.a_in"]))
hpre119 = (abar119 * h119 + delta119 * lin104(E119, "core.b_in")
           + rec119 - base119)

S_before119 = S["x"].copy()
S["x"] = state_array110.copy()

def read_hpre119(hp119, return_states=False):
    hn119 = layer104(lin104(hp119, "core.c_out"),
                    "core.coda.0", heads119)
    on119 = norm104(hn119.mean(dim=1), "core.out_norm")
    S["x"][core_rows104, :192] = on119.numpy()
    S["x"][core_rows104, 192:960] = hn119.numpy().reshape(-1, 768)
    rr119 = local_value_head(probe_head110, repair_head_cfg,
                            zmode="prior_mc", nsamples=8, seed=729)
    return (rr119, hn119, on119) if return_states else rr119["mean"]

try:
    head_base119, hn119, on119 = read_hpre119(hpre119, True)
    err_h119 = float(torch.sqrt(torch.mean(
        (hn119 - _actual_h104) ** 2)))
    err_out119 = float(torch.sqrt(torch.mean(
        (on119 - _actual_out104) ** 2)))
    print("Reconstruction RMSE h/out:", err_h119, err_out119)
    assert err_h119 < 0.002, "Stop: recurrent reconstruction is not accurate."
    J119 = np.zeros((len(probe_head110), 192), dtype=np.float64)
    eps119 = 1.0
    t119 = time.time()
    for d119 in range(192):
        perturb119 = torch.zeros_like(hpre119)
        perturb119[:, :, d119] = eps119
        mplus119 = read_hpre119(hpre119 + perturb119)
        mminus119 = read_hpre119(hpre119 - perturb119)
        J119[:, d119] = (mplus119 - mminus119) / (2 * eps119)
    print("Finite-difference elapsed seconds:", round(time.time()-t119, 1))
    for event119 in ["glucose_finger", "map_arterial"]:
        jj119 = J119[probe_head110.event.eq(event119).to_numpy()]
        print(event119, "n", len(jj119),
              "mean-direction norm", np.linalg.norm(jj119.mean(axis=0)),
              "median individual norm", np.median(np.linalg.norm(jj119, axis=1)))
    print("delta range:", float(delta119.min()), float(delta119.max()))
finally:
    S["x"] = S_before119

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 113–119: restart recovery and direct-state sensitivity\n"
            "No edits. Recovered prelude fits: their intercept is FIRST, not last. "
            "Using the wrong convention inflated residual estimates. "
            "Testing an analytical rank-two core.b_in injection using fitted E dose residuals "
            "and finite-difference next-reading sensitivities; no gradients or weight optimizer.\n")
```
