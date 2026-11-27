The recurrent replay now matches the saved states closely, and the latent extractor reproduces the intended payload to about 10⁻⁶. Those are useful checks, but not rollout validation. I’ll fit the final attention decoder and calibrate two injection directions for the record-derived effects. Correct scaled-dose contrasts and small cross-drug responses would support a pilot.

```python
import numpy as np, torch
import torch.nn.functional as F
torch.set_num_threads(8)

print("Latent-head attention, min/mean:",
      float(r169[3][:,3,:,6].min()), float(r169[3][:,3,:,6].mean()))
print("Carrier-only prelude disturbance RMS:",
      float(torch.mean((r169[0]-orig169[0])**2).sqrt()))

# Decode the two physical-dose innovations from the reserved head's values.
Vmat170 = zero169["core.prelude.0.attn.kv.weight"][336:384]
V170 = as_numpy(ln169 @ Vmat170.T).astype(np.float64)
target170 = as_numpy(signal169).astype(np.float64)
stay170 = S["stay_id"][idx81]
unique170 = np.unique(stay170)
rng170 = np.random.default_rng(170)
rng170.shuffle(unique170)
fit170 = np.isin(stay170, unique170[:int(.8*len(unique170))])
decoder_candidates170 = []
for lam in [1e-9, 1e-7, 1e-5, 1e-3]:
    B = np.linalg.solve(V170[fit170].T @ V170[fit170] + lam*np.eye(48),
                        V170[fit170].T @ target170[fit170])
    err = np.sqrt(np.mean((V170[~fit170] @ B-target170[~fit170])**2, axis=0))
    decoder_candidates170.append((float(np.mean(err)), lam, B, err))
_, lam170, decoder170, error170 = min(decoder_candidates170, key=lambda x:x[0])
print("Decoder holdout RMS / ridge / norm:", error170, lam170,
      np.linalg.norm(decoder170, axis=0))

active_idx170 = np.flatnonzero(active154)
print("Action alignment:", len(active_idx170), len(u_cases153[0]),
      float(np.sqrt(np.mean((as_numpy(U154)[active_idx170]-as_numpy(u_cases153[0]))**2))))
case_signals170 = []
for case in range(len(u_cases153)):
    post = posterior153(C154[active_idx170], E154[active_idx170],
                        u_cases153[case], post154)[:, :32]
    raw = F.linear(post, tensor153(zin154), tensor153(zb154))
    norm = F.layer_norm(raw, (192,), W["core.prelude.0.norm1.weight"],
                        W["core.prelude.0.norm1.bias"])
    case_signals170.append(as_numpy(norm @ Vmat170.T @ tensor153(decoder170)))
case_signals170 = np.asarray(case_signals170)
for case in range(1, len(case_signals170)):
    desired = y_cases153[case,:,:2] - y_cases153[0,:,:2]
    actual = case_signals170[case] - case_signals170[0]
    print("Case", case, "dose contrast mean", np.mean(desired,axis=0),
          "decoded mean", np.mean(actual,axis=0),
          "contrast RMS", np.sqrt(np.mean((actual-desired)**2,axis=0)))

def view170(h, out, dt):
    dt = tensor153(dt).reshape(-1)
    hf = h * torch.exp(-dt[:,None,None] * torch.exp(W["core.flow.a"]))
    ctx = out + layer_norm(hf.mean(1), "core.out_norm")
    ctx = ctx + mlp(torch.log(dt.clamp_min(1))[:,None]/10, "gap")
    return ctx + linear(mlp(ctx, "z_prior")[:,:32], "z_head")

codes170 = np.asarray(read_codes105, dtype=int)
events170 = {int(c): str(readrows121.iloc[np.flatnonzero(codes97==c)[0]]["event"])
             for c in codes170}
print("Calibration codes:", events170)
def phys170(view):
    pred = readout134(view[:,None,:])
    return torch.stack([pred[:,np.asarray(codes97)==c].mean(1)
                        for c in codes170], dim=1)

basis170 = tensor153(np.linalg.qr(rng170.standard_normal((192,64)))[0])
J170 = np.zeros((4,64))
eps170 = .02
for start in range(0,64,8):
    n = min(8,64-start)
    injection = basis170[:,start:start+n].T.repeat_interleave(16,0)[:,None,:]
    predictions = []
    for sign in [-1,1]:
        E = prelude155(tokens_carrier155[positions155].repeat(n,1,1),
                       zero169, sign*eps170*injection)[0]
        h, out = state_update82(hf169.repeat(n,1,1), E, mem169.repeat(n,1,1),
                               mask169.repeat(n,1), age_emb169.repeat(n,1,1), 0)
        predictions.append(as_numpy(phys170(view170(h,out,next_dt105.repeat(n))))
                           .reshape(n,16,4).mean(1))
    J170[:,start:start+n] = ((predictions[1]-predictions[0])/(2*eps170)).T

effect170 = {"glucose_finger":(-1.94,0), "glucose_lab":(-1.91,0),
             "map_arterial":(0,3.20), "map_cuff":(0,2.90)}
T170 = np.asarray([effect170[events170[c]] for c in codes170])
coef170 = J170.T @ np.linalg.solve(J170 @ J170.T + 1e-5*np.eye(4), T170)
directions170 = as_numpy(basis170) @ coef170
print("Jacobian singular values:", np.linalg.svd(J170,compute_uv=False))
print("Injection direction norms:", np.linalg.norm(directions170,axis=0))
print("Predicted immediate effect:", J170 @ coef170)
with open("/work/notes.md", "a") as f:
    f.write("\n\nPython run 170; forwards 10. Recurrent reconstruction validated: "
            "h RMS .00203, out RMS 1.48e-5. Latent extractor reconstructs intended "
            "physical-dose innovations to about 1e-6. Fitting attention-value decoder "
            "and numerical finite-difference physiological injection; no gradients, "
            "and no additional weight edits in this run.\n")
```
