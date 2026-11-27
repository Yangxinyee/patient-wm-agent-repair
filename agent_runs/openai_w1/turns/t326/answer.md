Using exported posterior means makes the factual tokens match the verified original replay exactly. Dose changes also produce meaningful prelude contrasts, with zero contrast on inactive bundles. I’ll fit small dose projections with factual-leakage and feedback constraints. Low leakage on held-out stays and approximately proportional dose responses would justify a cautious GPU trial; poor results would rule it out.

```python
from scipy.linalg import null_space

# Local injection directions were calibrated earlier without gradients.
B286 = np.asarray(B257, dtype=np.float64)
d286 = F.softplus(W282["core.delta_in"]).numpy().astype(np.float64)
Jimm286 = d286[:, None] * B286

# Approximate transformed feedback: coda response, then next prelude,
# holding the next exported latent fixed. This is a stability constraint,
# not a claim that the posterior reconstruction has been solved.
eps286 = 0.1
next286 = np.asarray(pick276, dtype=int) + 1
valid286 = (
    (next286 < len(E285))
    & (S280["stay_id"][np.minimum(next286, len(E285)-1)]
       == S280["stay_id"][pick276])
)
ixnext286 = next286[valid286]
feedback_rows286 = []
for j in range(2):
    yp, _ = coda269(torch.as_tensor(eps286 * B286[:, j],
                                    dtype=torch.float32))
    ym, _ = coda269(torch.as_tensor(-eps286 * B286[:, j],
                                    dtype=torch.float32))
    dy = (yp - ym)[valid286] / (2 * eps286)
    decay_next = torch.exp(
        -torch.as_tensor(dt277, dtype=torch.float32)[ixnext286, None, None]
        * torch.exp(flow_a281)
    )
    rawp = raw285[ixnext286].clone()
    rawm = rawp.clone()
    rawp[:, :4] += eps286 * dy * decay_next
    rawm[:, :4] -= eps286 * dy * decay_next
    ep = prelude277(rawp, W282)
    em = prelude277(rawm, W282)
    feedback_rows286.append(
        ((ep - em) / (2 * eps286)).numpy().reshape(-1, 192)
    )
feedback_matrix286 = np.concatenate(feedback_rows286, axis=0).astype(np.float64)
_, sv286, vt286 = np.linalg.svd(feedback_matrix286, full_matrices=False)
print("Feedback singular values:", sv286[:10])
print("Fraction of feedback energy in first four directions:",
      float(np.sum(sv286[:4]**2) / np.sum(sv286**2)))

unique286 = np.unique(S280["stay_id"])
train286 = np.isin(S280["stay_id"], unique286[:75])
test286 = ~train286
print("Fit/held-out stays:", int(train286.sum()), int(test286.sum()),
      len(unique286[:75]), len(unique286[75:]))

def moments286(mask):
    ff = E285[mask].reshape(-1, 192).astype(np.float64)
    fg = ff.T @ ff / len(ff)
    dg = np.zeros((192, 192), dtype=np.float64)
    dc = np.zeros((192, 2), dtype=np.float64)
    count = 0
    for (drug, factor), ecf in Es285.items():
        j = 0 if drug == "insulin" else 1
        active = mask & (ds_rates[:, j] > 0)
        dd = (ecf[active] - E285[active]).reshape(-1, 192).astype(np.float64)
        yy = np.zeros((len(dd), 2), dtype=np.float64)
        yy[:, j] = np.repeat(ds_rates[active, j] * (factor - 1), 4)
        dg += dd.T @ dd
        dc += dd.T @ yy
        count += len(dd)
    return fg, dg / count, dc / count, ff.mean(0)

stats286 = {
    "train": moments286(train286),
    "all": moments286(np.ones(len(E285), dtype=bool))
}
fits286 = {}
fits286_train = {}

def fit286(stats, alpha, mode):
    fg, dg, dc, mean = stats
    cols = [Jimm286, mean[:, None]]
    if mode == "feedback4":
        cols.append(vt286[:4].T)
    jj = np.concatenate(cols, axis=1)
    nn = null_space(jj.T)
    gg = dg + alpha * fg + 1e-4 * np.eye(192)
    pp = nn @ np.linalg.solve(nn.T @ gg @ nn, nn.T @ dc)
    return pp

def metrics286(pp):
    leak = E285[test286].reshape(-1, 192).astype(np.float64) @ pp
    ratios = []
    for drug, factor in (
        ("insulin", 2.0), ("insulin", 0.5),
        ("norepinephrine", 2.0), ("norepinephrine", 0.5)
    ):
        j = 0 if drug == "insulin" else 1
        active = test286 & (ds_rates[:, j] > 0)
        xx = (Es285[(drug, factor)][active] - E285[active]).mean(1)
        pred = xx @ pp[:, j]
        truth = ds_rates[active, j] * (factor - 1)
        ratios.append(float(np.sum(pred) / np.sum(truth)))
    return np.sqrt(np.mean(leak**2, axis=0)), ratios

print("\nmode alpha | held-out factual RMS [ins, NE] | dose-recovery ratios [x2, half, x2, half]")
for mode in ("immediate", "feedback4"):
    for alpha in (1.0, 10.0, 100.0):
        pp = fit286(stats286["train"], alpha, mode)
        fits286_train[(mode, alpha)] = pp
        fits286[(mode, alpha)] = fit286(stats286["all"], alpha, mode)
        leak, ratios = metrics286(pp)
        print(mode, alpha, np.round(leak, 5), np.round(ratios, 4),
              "P column RMS", np.round(np.sqrt(np.mean(pp**2, axis=0)), 3))

print("\n=== LOCAL UNIT RESPONSE CHECK ===")
rzero286 = np.asarray(direct_readout257(np.zeros(192)))
print("Readout shape:", rzero286.shape)
for j in range(2):
    rp = np.asarray(direct_readout257(B286[:, j]))
    rm = np.asarray(direct_readout257(-B286[:, j]))
    response = (rp - rm) / 2
    print("Channel", j, "mean local response",
          np.mean(response, axis=0) if response.ndim > 1 else response)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 286\nExact factual prelude features confirmed via exported z and original replay. Fitted analytical rank-two dose projections across x0/.5/1.5/2/3 with factual penalties, immediate-injection null constraints and optional four-direction approximate transformed-feedback nulling; evaluated on 25 held-out stays. No weights changed. Posterior dose response remains a limitation requiring GPU testing. Counts: 286 Python runs, 30 forwards.\n")
```
