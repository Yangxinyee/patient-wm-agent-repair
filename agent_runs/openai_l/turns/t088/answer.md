The factual action transition matches the exported output to about 1.5×10⁻⁵ RMS, and action re-encoding is also accurate. I’ll now measure how doubling insulin changes the next-reading features. A mismatch with the record-based −1.95 next-hour response would identify what the head correction needs to supply.

```python
torch.set_num_threads(8)
drug_names81 = np.asarray(drugs65)
rate_state81 = {
    d: np.bincount(owners72,
                   weights=np.where(drug_names81 == d, rates65, 0.0),
                   minlength=len(sx63)).astype(np.float32)
    for d in ["insulin", "norepinephrine"]
}
cf_features81 = {}

def make_cf81(drug, factor, batch_size=96):
    rr = rates65.copy()
    dd = doses65.copy()
    changed = drug_names81 == drug
    rr[changed] *= factor
    dd[changed] *= factor
    uu = pool_actions49(action_embedding67(drugs65, rr, dd),
                        owners72, counts72)
    reading_rows = np.flatnonzero(rate_state81[drug][pi73] > 0)
    state_indices = np.unique(pi73[reading_rows])
    state_lookup = np.full(len(sx63), -1, dtype=np.int64)
    state_lookup[state_indices] = np.arange(len(state_indices))
    out_cf = torch.empty(len(state_indices), 192)
    h_cf = torch.empty(len(state_indices), 4, 192)
    for start in range(0, len(state_indices), batch_size):
        ids = state_indices[start:start+batch_size]
        ctx, hf, memory, ages, padding = bundle_inputs74(ids)
        posterior = mlp63(torch.cat(
            [ctx, e_valid60[ids], uu[ids]], dim=-1), "z_post")
        oo, hh = transition64(hf, e_valid60[ids], uu[ids],
                             posterior[:, :32], memory, ages, ~padding)
        out_cf[start:start+len(ids)] = oo
        h_cf[start:start+len(ids)] = hh
    pos = state_lookup[pi73[reading_rows]]
    ctx_read, _ = context64(out_cf[pos], h_cf[pos], dtread73[reading_rows])
    means, probs, hidden, logits = decode67(
        ctx_read, const_train67[reading_rows], trans_train67[reading_rows],
        centres_train67[reading_rows], epsread73[:, reading_rows],
        return_features=True)
    ans = {
        "rows": reading_rows, "hidden": hidden, "logits": logits,
        "means": means, "probs": probs,
        "delta_rate": (factor-1)*rate_state81[drug][pi73[reading_rows]]
    }
    cf_features81[(drug, factor)] = ans
    return ans

for factor in [1.0, 2.0]:
    c = make_cf81("insulin", factor)
    r = c["rows"]
    shift = (c["means"]-means_fact80[:, r]).mean(0).numpy()
    print("\ninsulin factor", factor, "readings", len(r))
    for ev in np.unique(events74):
        m = events74[r] == ev
        if m.any():
            print(ev, "mean shift", float(shift[m].mean()),
                  "RMS shift", float(np.sqrt(np.mean(shift[m]**2))))
            if factor != 1:
                dr = c["delta_rate"][m]
                print("  dose-weighted slope",
                      float(np.dot(dr, shift[m])/np.dot(dr, dr)))

print("\n=== TRAIN CSV HEADER ===")
print(pd.read_csv("/data/train.csv", nrows=2).to_string(index=False))
```