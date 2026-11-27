The recovered analysis estimates a positive norepinephrine effect using history-conditioned dose innovations, though that identification assumption remains important. The current head underreacts locally, especially to doubling. I’ll fit small ridge regressions that target those dose contrasts while penalizing changes to factual predictions. Stronger held-out responses within the factual-loss budget would support a new edit; training-only gains would not.

```python
import scipy.linalg as la208
import pandas as pd
import torch.nn.functional as F

# Statistical regressions on fixed model outputs; no model gradients or edits.
beta208 = 3.55456
mapmask208 = head_meta.event.str.startswith("map").to_numpy()
tunemap208 = wm_tune_meta145.event.str.startswith("map").to_numpy()
fitstays208 = set(head_meta.loc[cal190_fitmask, "stay_id"])
fitall208 = head_meta.stay_id.isin(fitstays208).to_numpy()
cvall208 = ~fitall208
fitmap208 = fitall208 & mapmask208

calweights208 = head177_make(map192_cal)
bt208 = head177_eval(wm_train145, calweights208)
bu208 = head177_eval(wm_tune145, calweights208)
bc208 = {k: head177_eval(mc, calweights208)
         for k, mc in wm_cf146.items()}
v208 = torch.as_tensor(wm_v151, dtype=torch.float32)
cent208 = wm_train145["centres"]
tcent208 = wm_tune145["centres"]

xx208 = wm_train145["inputs"][:, fitmap208]
center208 = xx208.mean((0, 1)).numpy().astype(np.float64)
scale208 = xx208.std((0, 1)).clamp_min(0.05).numpy().astype(np.float64)
xs208 = ((xx208.numpy().astype(np.float64) - center208) / scale208)
xs208 = np.concatenate([xs208, np.ones((*xs208.shape[:2], 1))], axis=2)
pf208 = bt208["probs"][:, fitmap208]
vv208 = ((pf208 * v208.square()).sum(-1)
         - (pf208 * v208).sum(-1).square()).clamp_min(1e-5).numpy()
xf208 = xs208.reshape(-1, 193)
q208 = xf208.T @ (xf208 * vv208.reshape(-1, 1)) / len(xf208)
q208 = (q208 + q208.T) / 2

def response_design208(mc, pred):
    p = pred["probs"]
    md = (p * cent208).sum(-1)
    vd = (p * v208).sum(-1)
    cov = (p * (cent208 * v208)).sum(-1) - md * vd
    gx = (cov[..., None] * mc["inputs"]).mean(0).numpy().astype(np.float64)
    gb = cov.mean(0).numpy().astype(np.float64)
    return np.column_stack([(gx - gb[:, None]*center208)/scale208, gb])

gf208 = response_design208(wm_train145, bt208)
stat208 = {}
for k208, mc208 in wm_cf146.items():
    sel208 = fitmap208 & np.asarray(mc208["active"], dtype=bool)
    gd208 = response_design208(mc208, bc208[k208]) - gf208
    original_shift208 = bc208[k208]["mean"] - bt208["mean"]
    if k208[0] == "norepinephrine":
        y208 = beta208*np.asarray(mc208["delta_rate"]) - original_shift208
    else:
        # Preserve the baseline's cross-drug MAP contrasts.
        y208 = np.zeros(len(head_meta))
    g208 = gd208[sel208]
    stat208[k208] = (g208.T @ g208 / max(len(g208), 1),
                    g208.T @ y208[sel208] / max(len(g208), 1))

def scalar_eval208(mc, baseline, coef, mask, centres):
    cc = torch.as_tensor(coef, dtype=torch.float32)
    tau = F.linear(mc["inputs"], cc[:192], cc[-1])
    tau = tau * torch.as_tensor(mask, dtype=torch.float32)[None, :]
    pm = torch.softmax(
        baseline["logits"] + tau[..., None]*v208, dim=-1
    ).mean(0)
    target = torch.as_tensor(mc.get("target", wm_train145["target"]),
                             dtype=torch.long)
    nll = -pm.gather(1, target[:, None]).squeeze(1).clamp_min(1e-30).log()
    mean = (pm * centres).sum(-1)
    return nll.numpy(), mean.numpy()

def raw_coef208(cs):
    c = np.empty(193)
    c[:192] = cs[:192]/scale208
    c[-1] = cs[-1] - np.dot(c[:192], center208)
    return c

map208_candidates = {}
rows208 = []
k2, b2 = stat208[("norepinephrine", 2.0)]
kh, bh = stat208[("norepinephrine", 0.5)]
ki = sum(stat208[("insulin", f)][0] for f in (0.5, 2.0))/2

for ridge208 in (0.003, 0.03):
    for fw208 in (0.1, 0.3, 1.0, 3.0, 10.0):
        for hw208 in (0.0, 0.25, 1.0):
            mat208 = k2 + hw208*kh + 0.25*ki + fw208*q208
            mat208 += ridge208*np.trace(q208)/193*np.eye(193)
            cs208 = la208.solve(mat208, b2+hw208*bh, assume_a="pos")
            add208 = raw_coef208(cs208)
            key208 = (fw208, hw208, ridge208)
            map208_candidates[key208] = map192_cal + add208
            nn208, mm208 = scalar_eval208(
                wm_train145, bt208, add208, mapmask208, cent208)
            tn208, _ = scalar_eval208(
                wm_tune145, bu208, add208, tunemap208, tcent208)
            row208 = dict(fw=fw208, half_weight=hw208, ridge=ridge208,
                          cv_dNLL=float((nn208-wm_train145["nll"])[cvall208].mean()),
                          tune_dNLL=float((tn208-wm_tune145["nll"]).mean()),
                          norm=float(np.linalg.norm(map192_cal+add208)))
            for f208 in (0.5, 2.0):
                mc208 = wm_cf146[("norepinephrine", f208)]
                _, cm208 = scalar_eval208(mc208, bc208[("norepinephrine", f208)],
                                         add208, mapmask208, cent208)
                for ev208, tag208 in (("map_arterial", "A"), ("map_cuff", "C")):
                    ss208 = cvall208 & (head_meta.event.to_numpy()==ev208)
                    ss208 &= np.asarray(mc208["active"], dtype=bool)
                    dr208 = np.asarray(mc208["delta_rate"])[ss208]
                    dy208 = (cm208-mm208)[ss208]
                    row208[f"{tag208}{f208}_slope"] = float(
                        np.dot(dr208, dy208)/max(np.dot(dr208, dr208), 1e-12))
            rows208.append(row208)

map208_table = pd.DataFrame(rows208)
map208_table.to_csv("/work/map_double_response_frontier.csv", index=False)
print("=== DOUBLING-FOCUSED FRONTIER: LOCAL, NOT ROLLOUT ===")
print(map208_table.query("cv_dNLL < .005 and tune_dNLL < .0047")
      .sort_values("A2.0_slope", ascending=False).head(15).to_string(index=False))
print("\nFactual-loss ranges:")
print(map208_table[["cv_dNLL", "tune_dNLL", "A2.0_slope",
                    "A0.5_slope"]].agg(["min", "max"]).to_string())
with open("/work/notes.md", "a") as n208:
    n208.write("\n## Run 208: doubling-focused MAP regression\n"
               "- Current candidate confirmed unchanged; local NE response remains weak.\n"
               "- Fit closed-form ridge regressions on fixed head mean sensitivities, "
               "with factual-distribution and cross-drug preservation penalties.\n"
               "- Training uses 60 stays; screening uses 40 held-out training stays "
               "and tuning. No weights installed; 21 forward passes used.\n")
```
