The capture worked: it exposes all four sensors’ final-layer features, and the scratch weights are still original. I’ll now build paired original/doubled-dose features plus factual features. These can support a closed-form output-layer correction that penalizes factual changes; held-out dose-response recovery will determine whether it merits an official test.

```python
import numpy as np, torch, time
from pathlib import Path

r179_start = time.time()
r179_indices = np.asarray(indices151, dtype=int)
r179_drug = np.asarray(drug156, dtype=int)
r179_dc = np.asarray(dc156, dtype=float)
assert len(r179_indices) == len(r179_drug) == len(r179_dc) == len(ucf156)
assert np.isfinite(r179_dc).all() and (r179_dc > 0).all()

def r179_capture_head(out, h, indices):
    hidden, probs = [], []
    old_lin, old_softmax = lin0, torch.softmax
    def capture_lin(x, prefix):
        if prefix == "attr_heads.value_bin.2":
            hidden.append(x.detach().clone())
        return old_lin(x, prefix)
    def capture_softmax(x, dim, *args, **kwargs):
        result = old_softmax(x, dim, *args, **kwargs)
        if x.ndim == 3 and x.shape[-1] == 17:
            probs.append(result.detach().clone())
        return result
    try:
        globals()["lin0"] = capture_lin
        torch.softmax = capture_softmax
        means = r49_readouts(out, h, indices)
    finally:
        globals()["lin0"] = old_lin
        torch.softmax = old_softmax
    assert len(hidden) == len(probs) == 4, (len(hidden), len(probs))
    result = {}
    for j, event in enumerate(r49_sensors):
        H = hidden[j].numpy().astype(np.float64)
        H = np.concatenate([H, np.ones((*H.shape[:-1],1))], axis=-1)
        P = probs[j].numpy().astype(np.float64)
        centres = r49_centres[j].numpy().astype(np.float64)
        M = (P * centres).sum(-1)
        assert np.max(np.abs(M.mean(1)-means[:,j].numpy())) < 1e-3
        result[event] = {"H": H, "P": P, "C": centres, "M": M}
    return result

r179_u0 = torch.as_tensor(rr33_u_all[r179_indices], dtype=torch.float32)
r179_o0, r179_h0 = r59_core(r179_indices, r179_u0)
r179_oc, r179_hc = r59_core(r179_indices, ucf156)
r179_rms = float(np.sqrt(np.mean(
    (r179_o0.numpy()-sx[r179_indices,:192])**2)))
print("Original local-state reconstruction RMS:", r179_rms)
assert r179_rms < 1e-3

r179_head0 = r179_capture_head(r179_o0, r179_h0, r179_indices)
r179_headc = r179_capture_head(r179_oc, r179_hc, r179_indices)

r179_rng = np.random.default_rng(179)
r179_findices = r179_rng.choice(len(sx), min(1600,len(sx)), replace=False)
r179_headf = r179_capture_head(
    torch.tensor(sx[r179_findices,:192], dtype=torch.float32),
    torch.tensor(sx[r179_findices,192:960].reshape(-1,4,192), dtype=torch.float32),
    r179_findices)

r179_tilt = np.r_[0., np.linspace(-2.,2.,16)]
def r179_sensitivity(data):
    H,P,C,M = (data[k] for k in ("H","P","C","M"))
    sensitivity = (P*(C[None,None,:]-M[:,:,None])*r179_tilt).sum(-1)
    G = (H*sensitivity[:,:,None]).mean(1)
    Et = (P*r179_tilt).sum(-1)
    Vt = (P*r179_tilt**2).sum(-1)-Et**2
    return G,Vt

r179_G0, r179_Gc, r179_Gf, r179_Vf = {},{},{},{}
for event in r49_sensors:
    r179_G0[event],_ = r179_sensitivity(r179_head0[event])
    r179_Gc[event],_ = r179_sensitivity(r179_headc[event])
    r179_Gf[event],r179_Vf[event] = r179_sensitivity(r179_headf[event])
    delta = r179_headc[event]["M"].mean(1)-r179_head0[event]["M"].mean(1)
    print(event, "original paired slope per rate unit:",
          {int(d): float(np.dot(r179_dc[r179_drug==d],delta[r179_drug==d]) /
                         np.dot(r179_dc[r179_drug==d],r179_dc[r179_drug==d]))
           for d in np.unique(r179_drug)})
print("Drug labels and dose-change quartiles:")
for d in np.unique(r179_drug):
    print(int(d), int((r179_drug==d).sum()),
          np.quantile(r179_dc[r179_drug==d],[0,.25,.5,.75,1]).round(4))
print("Elapsed seconds", round(time.time()-r179_start,1))

with open("/work/notes.md","a") as f:
    f.write("\n## Run 179: build paired and factual output-layer features\n"
            "- Captured original/doubled-dose hidden features and probabilities for "
            "all four sensors, plus 1,600 factual contexts.\n"
            "- Original local-core reconstruction checked against official states.\n"
            "- Preparing a no-gradient rank-one bin-tilt edit, with factual Fisher "
            "penalty and stay-held-out dose-response validation.\n"
            "- Python runs: 179; forward passes: 10. Saved edits unchanged.\n")
```
