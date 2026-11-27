The first decoder is not selective enough: it reports substantial signal on bundles without insulin. Large synthetic signals transfer, but the factual leakage could create unwanted feedback. I’ll check the channel’s intermediate output and exclude layer-normalization invariances when choosing its direction. A successful decoder should need moderate coefficients and stay near zero on non-action bundles.

```python
import scipy.linalg

print("--- diagnose previous channel geometry ---")
last_controls112 = controls111
diff112 = last_controls112[2.] - last_controls112[0.]
print("previous q=2 versus q=0 E change RMS:",
      float(np.sqrt(np.mean(diff112**2))))
print("previous decoder gain:", float(np.mean(diff112 @ decoder111)/2))

# Exclude normalization invariances and leading injected-memory directions.
beta112 = np.asarray(rW["core.prelude_norm.bias"], dtype=np.float64)
constraints112 = np.vstack([
    1/gamma111, Xbase111.mean(axis=0), beta112, vh111[:20]
])
null112 = scipy.linalg.null_space(constraints112, rcond=1e-8)
gram112 = Xbase111.T@Xbase111 / len(Xbase111)
eig112, ev112 = np.linalg.eigh(null112.T@gram112@null112)
reader_direction112 = null112 @ ev112[:,0]
channel_direction112 = gamma111*reader_direction112
channel_direction112 -= channel_direction112.mean()
channel_direction112 /= np.linalg.norm(channel_direction112)
print("constrained subspace dimension:", null112.shape[1],
      "baseline reader RMS:",
      float(np.sqrt(np.mean((Xbase111@reader_direction112)**2))))

# Also retain tokens to verify the actual normalized attention input.
def capture_ep112(indices):
    captured = {}
    previous_profile = sys.getprofile()
    def profile112(frame, event, arg):
        if event == "return" and frame.f_code is transition73.__code__:
            for key in ("ep", "zk", "hf", "tok"):
                captured[key] = frame.f_locals[key].detach().cpu().numpy().copy()
    sys.setprofile(profile112)
    try:
        states = transition73(indices)
    finally:
        sys.setprofile(previous_profile)
    return states, captured

channel_fits112 = {}
for amplitude112 in (40., 80., 160.):
    weights112 = dict(rW_candidate108)
    out112 = weights112["core.prelude.0.attn.out.weight"].clone()
    out112[:, head104*width103] = torch.as_tensor(
        amplitude112*channel_direction112, dtype=torch.float32)
    weights112["core.prelude.0.attn.out.weight"] = out112
    try:
        rW = weights112
        _, cap112 = capture_ep112(indices102)
        X112 = [cap112["ep"].reshape(-1,192).astype(np.float64)]
        y112 = [np.repeat(cap112["zk"] @ latent_axis104,4)]
        f112 = [np.repeat(fit_bundle111,4)]
        controls112 = {}
        for qt112 in (-6., -2., 0., 2., 6.):
            _, cc112 = capture_q111(control_indices111, qt112)
            controls112[qt112] = cc112["ep"].reshape(-1,192).astype(np.float64)
            X112.append(controls112[qt112])
            y112.append(np.full(len(controls112[qt112]),qt112))
            f112.append(np.repeat(control_fit111,4))
    finally:
        rW = rW_original108
    X112, y112, f112 = map(np.concatenate,(X112,y112,f112))
    dec112 = np.linalg.solve(
        X112[f112].T@X112[f112]+1e-4*np.eye(192),
        X112[f112].T@y112[f112])
    pred112 = (cap112["ep"].reshape(-1,192)@dec112).reshape(-1,4)
    q112 = cap112["zk"]@latent_axis104
    te112 = ~fit_bundle111
    rmse112 = float(np.sqrt(np.mean((pred112[te112]-q112[te112,None])**2)))
    leak112 = float(np.sqrt(np.mean(pred112[te112 & ~has_insulin111]**2)))
    print("amplitude", amplitude112, "decoder norm",
          round(float(np.linalg.norm(dec112)),3),
          "factual RMSE", round(rmse112,5),
          "non-action RMS", round(leak112,5))
    print("controlled held-out means:",
          [(q, round(float((controls112[q]@dec112).reshape(-1,4)
                           [~control_fit111].mean()),4))
           for q in (-6.,-2.,0.,2.,6.)])
    channel_fits112[amplitude112] = {
        "weights": weights112, "decoder": dec112,
        "factual_rmse": rmse112, "leakage": leak112,
        "capture": cap112
    }

print("self-block constants:", self_block67.__code__.co_consts)
print("saved edits:", edited())
```
