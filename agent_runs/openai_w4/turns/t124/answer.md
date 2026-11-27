The recurrent-input probes found a direction that produces roughly −2 glucose units per unit signal, with much smaller MAP changes. It remains reasonably linear over the tested range. Next I’ll connect the attention signal to that direction using a fitted decoder, checking held-out accuracy, non-action leakage, and transfer at larger signals before saving any weights.

```python
import sys, time
t111 = time.time()

def capture_ep111(indices):
    captured = {}
    previous_profile = sys.getprofile()
    def profile111(frame, event, arg):
        if event == "return" and frame.f_code is transition73.__code__:
            for key in ("ep", "zk", "hf"):
                captured[key] = frame.f_locals[key].detach().cpu().numpy().copy()
    sys.setprofile(profile111)
    try:
        states = transition73(indices)
    finally:
        sys.setprofile(previous_profile)
    return states, captured

def capture_q111(indices, qvalue):
    global rp_mlp
    old_mlp = rp_mlp
    axis = torch.as_tensor(latent_axis104, dtype=torch.float32)
    def mlp111(x, prefix):
        y = old_mlp(x, prefix)
        if prefix == "z_post":
            y = y.clone()
            current = y[..., :32] @ axis
            y[..., :32] += (float(qvalue)-current)[..., None] * axis
        return y
    rp_mlp = mlp111
    try:
        return capture_ep111(indices)
    finally:
        rp_mlp = old_mlp

try:
    rW = rW_candidate108
    _, base_cap111 = capture_ep111(indices102)
    plus_h111 = transition_with_b110(probe_indices95, v110)
    minus_h111 = transition_with_b110(probe_indices95, -v110)
finally:
    rW = rW_original108

# Select a channel direction with little factual activity or self-feedback.
Xbase111 = base_cap111["ep"].reshape(-1, 192).astype(np.float64)
gamma111 = np.asarray(rW["core.prelude_norm.weight"], dtype=np.float64)
dh111 = (np.asarray(plus_h111[:,192:960]) -
         np.asarray(minus_h111[:,192:960])).reshape(-1,192)/2
flow111 = np.asarray(rW["core.flow.a"], dtype=np.float64)
feedback111 = np.concatenate([
    dh111 * gamma111 * np.exp(-dt*np.exp(flow111))
    for dt in (0., 300., 3600.)
], axis=0)
_, _, vh111 = np.linalg.svd(feedback111, full_matrices=False)
G111 = Xbase111.T @ Xbase111 / len(Xbase111)
G111 += 100.0 * (vh111[:20].T @ vh111[:20])
eigen111, vectors111 = np.linalg.eigh(G111)
channel_direction111 = vectors111[:,0] / gamma111
channel_direction111 -= channel_direction111.mean()
channel_direction111 /= np.linalg.norm(channel_direction111)

stay111 = np.asarray(repair_bank["stay_id"])[indices102]
fit_bundle111 = (stay111 % 5) < 3
q_original111 = base_cap111["zk"] @ latent_axis104
has_insulin111 = np.asarray(repair_bank["has_insulin"])[indices102].astype(bool)
control_indices111 = indices102[::16]
control_fit111 = (np.asarray(repair_bank["stay_id"])[control_indices111] % 5) < 3
print("posterior signal SD:", float(q_original111.std()),
      "non-action max:", float(np.max(np.abs(q_original111[~has_insulin111]))))
channel_fits111 = {}

for amplitude111 in (20., 80., 320.):
    weights111 = dict(rW_candidate108)
    out111 = weights111["core.prelude.0.attn.out.weight"].clone()
    out111[:, head104*width103] = torch.as_tensor(
        amplitude111*channel_direction111, dtype=torch.float32)
    weights111["core.prelude.0.attn.out.weight"] = out111
    try:
        rW = weights111
        _, cap111 = capture_ep111(indices102)
        XX111 = [cap111["ep"].reshape(-1,192).astype(np.float64)]
        yy111 = [np.repeat(cap111["zk"] @ latent_axis104, 4)]
        ff111 = [np.repeat(fit_bundle111,4)]
        controls111 = {}
        for qtest111 in (-6., -2., 0., 2., 6.):
            _, cc111 = capture_q111(control_indices111, qtest111)
            controls111[qtest111] = cc111["ep"].reshape(-1,192).astype(np.float64)
            XX111.append(controls111[qtest111])
            yy111.append(np.full(len(controls111[qtest111]), qtest111))
            ff111.append(np.repeat(control_fit111,4))
    finally:
        rW = rW_original108
    XX111 = np.concatenate(XX111)
    yy111 = np.concatenate(yy111)
    ff111 = np.concatenate(ff111)
    decoder111 = np.linalg.solve(
        XX111[ff111].T@XX111[ff111] + 1e-4*np.eye(192),
        XX111[ff111].T@yy111[ff111])
    factual_pred111 = (
        cap111["ep"].reshape(-1,192) @ decoder111
    ).reshape(-1,4)
    factual_target111 = cap111["zk"] @ latent_axis104
    test111 = ~fit_bundle111
    rmse111 = np.sqrt(np.mean(
        (factual_pred111[test111]-factual_target111[test111,None])**2))
    leakage111 = np.sqrt(np.mean(
        factual_pred111[test111 & ~has_insulin111]**2))
    print("amplitude", amplitude111,
          "decoder norm", round(float(np.linalg.norm(decoder111)),3),
          "held-out factual RMSE", round(float(rmse111),5),
          "non-action RMS", round(float(leakage111),5))
    for qtest111 in (-6., -2., 0., 2., 6.):
        pp111 = (controls111[qtest111] @ decoder111).reshape(-1,4)
        print("  q", qtest111,
              "held-out mean", round(float(pp111[~control_fit111].mean()),4),
              "RMSE", round(float(np.sqrt(np.mean(
                  (pp111[~control_fit111]-qtest111)**2))),4))
    channel_fits111[amplitude111] = {
        "weights": weights111, "decoder": decoder111,
        "factual_rmse": float(rmse111), "leakage": float(leakage111)
    }

print("elapsed seconds:", round(time.time()-t111,2))
print("saved edits:", edited())
```
