The numerical response fit can move glucose and MAP separately, but the most precise directions require large state perturbations. I’ll use the smaller, more regularized directions and test exact local responses before submitting weights. I’ll also include zero, half, 1.5× and 3× doses when fitting the features. Roughly proportional dose responses and small factual shifts would support an official trial.

```python
import numpy as np, torch

r245_cases = {("factual", 1.0): base228}
for drug245 in ["insulin", "norepinephrine"]:
    r245_cases[drug245, 2.0] = original_cf231[drug245]
    for factor245 in [0.0, 0.5, 1.5, 3.0]:
        u245 = action230(drug245, factor245)
        r245_cases[drug245, factor245] = core228(
            original_token231(u245), u245
        )

r245_keys = [k for k in r245_cases if k[0] != "factual"]
r245_Dparts, r245_Yparts = [], []
for drug245, factor245 in r245_keys:
    de245 = np242(r245_cases[drug245, factor245]["E"]) - r243_baseE
    y245 = np.zeros((len(de245)*4, 2))
    col245 = ["insulin", "norepinephrine"].index(drug245)
    y245[:, col245] = np.repeat(
        (factor245-1)*r238_doses[:, col245], 4
    )
    r245_Dparts.append(de245.reshape(-1, 192))
    r245_Yparts.append(y245)
r245_D = np.concatenate(r245_Dparts).astype(np.float64)
r245_Y = np.concatenate(r245_Yparts)
r245_Cd = r245_D.T @ r245_D / len(r245_D)
r245_rhs = r245_D.T @ r245_Y / len(r245_D)
r245_fits = {}
print("=== MULTIPLIER-GENERAL RATE FEATURES ===")
for penalty245 in [1., 10., 100.]:
    V245 = np.linalg.solve(
        r245_Cd + penalty245*r243_Ca + 1e-5*np.eye(192), r245_rhs
    )
    r245_fits[penalty245] = V245
    cp245 = r245_D @ V245
    slopes245 = np.sum(cp245*r245_Y, 0)/np.sum(r245_Y**2, 0)
    print("penalty", penalty245,
          "slopes", slopes245.round(4),
          "slope-normalised factual RMS",
          (np.sqrt(np.mean((r243_A@V245)**2, 0))/slopes245).round(4))

r245_V = r245_fits[10.]
r245_dirs = r244_dirs[1.]
r245_R = np.stack([
    ((np242(r245_cases[d, 2.]["E"])-r243_baseE) @ r245_V).mean((0, 1))
    for d in ["insulin", "norepinephrine"]
])
r245_original_means = {}
for drug245 in ["insulin", "norepinephrine"]:
    case245 = r245_cases[drug245, 2.]
    r245_original_means[drug245] = (
        np242(r49_readouts(case245["out"], case245["h"], indices151))
        - np242(pbase242)
    ).mean(0)
r245_need = np.array([
    [-2*r238_doses[:, 0].mean()-r245_original_means["insulin"][0],
     -r245_original_means["insulin"][2]],
    [-r245_original_means["norepinephrine"][0],
     3*r238_doses[:, 1].mean()-r245_original_means["norepinephrine"][2]]
])
r245_response = np.einsum("nkj,jd->nkd", r244_J, r245_dirs).mean(0)[[0, 2]]
r245_Q = np.linalg.solve(r245_R, r245_need @ np.linalg.inv(r245_response).T)
r245_G = r245_dirs @ r245_Q.T
r245_delta = np242(torch.nn.functional.softplus(W0["core.delta_in"]))
r245_C = (r245_G @ r245_V.T) / r245_delta[:, None]

def r245_prediction(case, amplitude):
    features = np242(case["E"]) @ r245_V
    injection = features @ r245_G.T
    pre = case["pre"] + torch.as_tensor(amplitude*injection, dtype=torch.float32)
    return np242(r244_read_pre(pre, indices151))

print("\n=== EXACT LOCAL CANDIDATE RESPONSE ===")
for amplitude245 in [0., 0.25, 0.5, 1.]:
    factual245 = r245_prediction(base228, amplitude245)
    print("amplitude", amplitude245,
          "factual reading-shift RMS",
          np.sqrt(np.mean((factual245-np242(pbase242))**2, 0)).round(4))
    for drug245 in ["insulin", "norepinephrine"]:
        for factor245 in [2., 0.5]:
            pred245 = r245_prediction(r245_cases[drug245, factor245], amplitude245)
            print(drug245, factor245,
                  (pred245-factual245).mean(0).round(4))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 245: multiplier-general low-rank recurrent candidate\n"
            "- Numerical routing works locally; highly precise routes have large norms.\n"
            "- Chose more regularised readout directions and fitted rate features across x0/.5/1.5/2/3.\n"
            "- Computed a rank-two core.b_in candidate calibrated to record-estimated one-hour effects.\n"
            "- Measuring exact one-step factual and dose responses before submission.\n"
            "- No weights changed. Python runs: 245; forward passes: 17.\n")
```