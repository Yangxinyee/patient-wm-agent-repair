The paired training states show a factual penalty of about 0.00276 nats, concentrated in glucose. I’ll test small code-and-last-bin offsets fitted to restore the original distributions, with whole stays held out. Lower held-out divergence and NLL would support calibration; deterioration would reject it.

```python
groups176 = (train_meta162.code.to_numpy().astype(int)*17 +
             train_meta162.last_bin.to_numpy().astype(int))
ng176 = W0["embed_transition.value_bin.weight"].shape[0]
codes176 = train_meta162.code.to_numpy().astype(int)
stays176 = train_meta162.stay_id.to_numpy()
unique176 = np.unique(stays176)
rng176 = np.random.default_rng(176)
rng176.shuffle(unique176)
fold176 = np.isin(stays176, unique176[:len(unique176)//2])
delta_logits176 = (lt_orig175 - lt_edit175).astype(np.float64)
delta_logits176 -= delta_logits176.mean(axis=1, keepdims=True)
glucose_codes176 = np.unique(codes176[train_meta162.event.str.startswith("glucose")])
calibration_fits176 = {}

def fit_offsets176(mask, strength):
    counts = np.bincount(groups176[mask], minlength=ng176).astype(float)
    sums = np.zeros((ng176, 17))
    np.add.at(sums, groups176[mask], delta_logits176[mask])
    correction = np.zeros_like(sums)
    for code in glucose_codes176:
        rows = mask & (codes176 == code)
        code_mean = delta_logits176[rows].mean(0)
        ix = np.arange(code*17, (code+1)*17)
        prior = code_mean * (rows.sum()/(rows.sum()+100.))
        correction[ix] = (sums[ix] + strength*prior)/(counts[ix,None]+strength)
    return correction

def offset_metrics176(correction, mask):
    p = pt_edit175.astype(np.float64)*np.exp(correction[groups176])
    p /= p.sum(1, keepdims=True)
    original = pt_orig175.astype(np.float64)
    labels = yt175
    nll = -np.log(p[np.arange(len(p)), labels].clip(1e-12))
    nll_original = -np.log(original[np.arange(len(p)), labels].clip(1e-12))
    kl = np.sum(original*np.log((original+1e-12)/(p+1e-12)), axis=1)
    return {
        "NLL_delta": float((nll-nll_original)[mask].mean()),
        "KL": float(kl[mask].mean()),
        "offset_RMS": float(np.sqrt(np.mean(correction[groups176[mask]]**2)))
    }

print("UNCALIBRATED", offset_metrics176(np.zeros((ng176,17)), np.ones(len(yt175), bool)))
for strength176 in (10., 50., 200., 1000.):
    summaries176 = []
    for trainmask176 in (fold176, ~fold176):
        correction176 = fit_offsets176(trainmask176, strength176)
        summaries176.append(offset_metrics176(correction176, ~trainmask176))
    calibration_fits176[strength176] = fit_offsets176(np.ones(len(yt175), bool), strength176)
    print("strength", strength176, "held-out halves", summaries176)

print("\nINSULIN BASELINE HISTORY")
for marker176 in ("## Run 58:", "## Run 63:", "## Run 64:", "## Run 65:", "## Run 68:"):
    pos176 = nt170.find(marker176)
    if pos176 >= 0:
        end176 = nt170.find("\n## ", pos176+3)
        print(nt170[pos176:end176 if end176 >= 0 else None][:650])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 176: cross-validated factual-distribution offsets\n"
            "Corrected paired training MC NLL: original=1.64055, insulin baseline=1.64332 (+.00276). "
            "Testing closed-form code/last-bin logit-residual averages with shrinkage and held-out stays, "
            "restricted to glucose codes. No weights changed. Python runs: 176; forwards: 22.\n")
```