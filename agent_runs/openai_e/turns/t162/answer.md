The architecture confirms that a `b_in` change reaches the reading head through a known linear map followed by the coda. There are direct next-reading queries for many drug bundles, so an output-constrained regression is feasible. I’ll check the dose units, null-space projection and sensitivity mapping; mismatches would invalidate the proposed fit.

```python
import numpy as np
import torch
from pathlib import Path

def arr141(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

Binput141 = arr141(basis100).astype(np.float64)
EF141 = arr141(E_fact95).astype(np.float64)
cfrows141 = np.asarray(drug_rows57, dtype=int)
Fslot141 = EF141.reshape(-1, 192) @ Binput141
Fmean141 = EF141.mean(axis=1) @ Binput141
Dmean141 = {}
print("FEATURE ALIGNMENT")
print("input-basis orthogonality error",
      float(np.max(np.abs(Binput141.T @ Binput141 - np.eye(188)))))
print("factual XF95 RMSE",
      float(np.sqrt(np.mean((Fslot141 - arr141(XF95))**2))))
for key141 in E_edit95:
    direct141 = (
        arr141(E_edit95[key141]) - EF141[cfrows141]
    ).reshape(-1, 192) @ Binput141
    Dmean141[key141] = direct141.reshape(-1, 4, 188).mean(axis=1)
    print(key141, "DF103 RMSE",
          float(np.sqrt(np.mean((direct141 - arr141(DF103[key141]))**2))),
          "contrast RMS", float(np.sqrt(np.mean(direct141**2))))
print("slot feature variation / feature RMS",
      float(np.sqrt(np.mean((EF141 - EF141.mean(axis=1, keepdims=True))**2))),
      float(np.sqrt(np.mean(EF141**2))))

Braw141 = arr141(basis136).astype(np.float64)
Praw141 = arr141(prebasis136).astype(np.float64)
delta141 = np.logaddexp(0.0, arr141(load_original("core.delta_in")).astype(np.float64))
Cout141 = arr141(load_original("core.c_out.weight")).astype(np.float64)
Pcalc141 = Cout141 @ (delta141.reshape(-1, 1) * Braw141)
print("\nSENSITIVITY-TO-WEIGHT ALIGNMENT")
print("prebasis relative error",
      float(np.linalg.norm(Pcalc141 - Praw141) / max(np.linalg.norm(Praw141), 1e-12)))
U141, s141, Vt141 = np.linalg.svd(Braw141, full_matrices=False)
rank141 = int(np.sum(s141 > s141[0] * 1e-6))
Bdir141 = U141[:, :rank141]
Tdir141 = Vt141[:rank141].T / s141[:rank141]
Jm141 = arr141(Jmean136).astype(np.float64) @ Tdir141
Jl141 = np.einsum("rbd,dk->rbk", arr141(Jlogit136).astype(np.float64), Tdir141)
print("direction singular values", np.round(s141, 4), "retained", rank141)

print("\nRAW DOSE CHECK")
ne_mask141 = cf_frame["has_norepinephrine"].to_numpy(dtype=bool)
for factor141 in [2.0, 0.5]:
    dose141 = arr141(dose_contrasts46[("norepinephrine", factor141)])
    print("NE", factor141, "active contrast quantiles",
          np.round(np.quantile(dose141[ne_mask141, 1], [0, .25, .5, .75, 1]), 5))
if len(joined115) == len(cfrows141):
    rate141 = joined115["norepinephrine"].to_numpy(dtype=float)
    double141 = arr141(dose_contrasts46[("norepinephrine", 2.0)])[:, 1]
    valid141 = np.isfinite(rate141) & ne_mask141
    print("doubling contrast versus recorded NE rate max error",
          float(np.max(np.abs(double141[valid141] - rate141[valid141]))))

print("\nREADOUT CENTRES")
defs141 = readout121.__defaults__ or ()
print("default shapes", [
    tuple(x.shape) if isinstance(x, (np.ndarray, torch.Tensor)) else type(x).__name__
    for x in defs141
])
centres141 = arr141(defs141[-1]).astype(np.float64)
prob141 = arr141(prob_base136).astype(np.float64)
if centres141.shape == prob141.shape:
    print("centre mean agreement",
          float(np.max(np.abs((prob141 * centres141).sum(axis=1) - arr141(mean_base136)))))
    sd141 = np.sqrt(np.maximum(
        (prob141 * centres141**2).sum(axis=1) - (prob141 * centres141).sum(axis=1)**2, 1.0
    ))
    print("reading SD quantiles", np.round(np.quantile(sd141, [.1, .5, .9]), 3))

src141 = head_meta["state_row"].to_numpy(dtype=int)
lookup141 = np.full(len(EF141), -1, dtype=int)
lookup141[cfrows141] = np.arange(len(cfrows141))
cfhead141 = lookup141[src141]
fit_read141 = (head_meta["stay_id"].to_numpy(dtype=np.int64) % 5) != 4
print("\nTRAIN / HELD-OUT DIRECT NE QUERIES")
for event141 in ["map_arterial", "map_cuff", "glucose_finger"]:
    use141 = head_meta["event"].eq(event141).to_numpy() & (cfhead141 >= 0)
    use141 &= ne_mask141[np.maximum(cfhead141, 0)]
    print(event141, int(np.sum(use141 & fit_read141)), int(np.sum(use141 & ~fit_read141)))

with open("/work/notes.md", "a") as f141:
    f141.write(
        "\n## Run 141: verify output-regression alignment\n"
        "Audited admission-null features, corrected dose contrasts, raw NE units, "
        "orthonormalized output directions and the exact b_in-to-coda linear map. "
        "Preparing stay-held-out distributional and dose-response constraints. "
        "No weights changed. Python runs: 141; forwards: 15.\n"
    )
```