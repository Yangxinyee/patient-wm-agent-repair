The saved states cover all 100 training stays, so I can fit and hold out entire stays. I’ll try a value-head-only correction constrained by the record-derived slopes at several dose multipliers. This keeps teacher-forced states and non-value heads unchanged. A useful candidate must reproduce the dose response without exceeding the factual NLL budget.

```python
t59 = time.time()
rng59 = np.random.default_rng(5901)
groups59 = {sid: np.flatnonzero(rr33_stays == sid)
            for sid in np.unique(rr33_stays)}
position59 = np.zeros(len(sx), dtype=int)
for rows59 in groups59.values():
    position59[rows59] = np.arange(len(rows59))

def r59_core(indices, u=None):
    indices = np.asarray(indices, int)
    prev = indices - 1
    assert np.all(rr33_stays[prev] == rr33_stays[indices])
    dt = torch.tensor((rr33_hours[indices]-rr33_hours[prev])*3600,
                      dtype=torch.float32)
    h = torch.tensor(sx[prev, 192:960].reshape(-1, 4, 192))
    h *= torch.exp(-dt[:, None, None]*torch.exp(W0["core.flow.a"]))
    histories = [groups59[rr33_stays[i]][max(0, position59[i]-128):position59[i]]
                 for i in indices]
    L = max(map(len, histories))
    memory = torch.zeros((len(indices), L, 192))
    mask = torch.zeros((len(indices), L), dtype=torch.bool)
    ages = torch.zeros((len(indices), L, 1))
    for j, (i, rows) in enumerate(zip(indices, histories)):
        n = len(rows)
        memory[j, :n] = rr33_e_all[rows]
        mask[j, :n] = True
        ages[j, :n, 0] = torch.tensor(
            (rr33_hours[i]-rr33_hours[rows])*3600, dtype=torch.float32)
    e = rr33_e_all[indices]
    if u is None:
        u = rr33_u_all[indices]
    ctx = torch.tensor(contexts_audit[indices], dtype=torch.float32)
    z = mlp0(torch.cat([ctx, e, u], -1), "z_post")[:, :32]
    a, b = repaired_core_local(h, e, u, z, memory, mask, ages)
    return (a, b) if a.ndim == 2 else (b, a)

oc59, hc59 = r59_core(idx56)
print("Expanded-core agreement:",
      float((oc59-o56).abs().max()), float((hc59-h56).abs().max()))
assert float((oc59-o56).abs().max()) < 0.001

beta59 = torch.tensor(np.linspace(-1, 1, 17), dtype=torch.float32)
beta59[0] = 0

def r59_features(out, h, indices):
    captures, probabilities = [], []
    previous_mlp, previous_softmax = globals()["mlp0"], torch.softmax
    def capture_mlp(x, prefix, *args, **kwargs):
        y = previous_mlp(x, prefix, *args, **kwargs)
        if prefix == "attr_heads.value_bin":
            captures.append(F.gelu(F.linear(
                x, W0["attr_heads.value_bin.0.weight"],
                W0["attr_heads.value_bin.0.bias"])))
        return y
    def capture_softmax(x, *args, **kwargs):
        y = previous_softmax(x, *args, **kwargs)
        probabilities.append(y.clone())
        return y
    globals()["mlp0"], torch.softmax = capture_mlp, capture_softmax
    try:
        means = r49_readouts(out, h, indices)
    finally:
        globals()["mlp0"], torch.softmax = previous_mlp, previous_softmax
    H = torch.stack(captures, 1)
    probabilities = torch.stack(probabilities, 1)
    centres = r49_centres[None, :, None, :]
    sample_mean = (probabilities*centres).sum(-1, keepdim=True)
    gain = (probabilities*(centres-sample_mean)*beta59).sum(-1)
    G = (H*gain[..., None]).mean(2)
    return means.numpy(), G.numpy(), H, probabilities

mean_parts59, G_parts59, H_parts59, prob_parts59 = [], [], [], []
for start59 in range(0, len(sx), 128):
    ix59 = np.arange(start59, min(start59+128, len(sx)))
    m59, g59, hh59, pp59 = r59_features(
        torch.tensor(sx[ix59, :192]),
        torch.tensor(sx[ix59, 192:960].reshape(-1, 4, 192)), ix59)
    mean_parts59.append(m59); G_parts59.append(g59)
    H_parts59.append(hh59); prob_parts59.append(pp59)
M59 = np.concatenate(mean_parts59)
G59 = np.concatenate(G_parts59)
H59 = torch.cat(H_parts59)
Pr59 = torch.cat(prob_parts59)

next59 = np.arange(1, len(sx))
valid59 = ((rr33_stays[next59] == rr33_stays[next59-1])
           & (np.abs(rr33_hours[next59]-np.round(rr33_hours[next59])) < 1e-5)
           & (rr33_hours[next59] >= 1) & (rr33_hours[next59] <= 24))
fidx59 = next59[valid59]-1
future_index59 = pd.MultiIndex.from_arrays(
    [rr33_stays[next59[valid59]], np.round(rr33_hours[next59[valid59]]).astype(int)],
    names=P.index.names)
Y59 = P.reindex(future_index59)[r49_sensors].to_numpy()
bins59 = np.zeros(Y59.shape, dtype=int)
for j59, sensor59 in enumerate(r49_sensors):
    bins59[:, j59] = np.clip(1+np.searchsorted(
        num_spec0[(sensor59, "value")]["fine_bin_edges"], Y59[:, j59],
        side="left"), 1, 16)
observed59 = np.isfinite(Y59)
p_true59 = np.take_along_axis(Pr59[fidx59].mean(2).numpy(),
                             bins59[..., None], -1)[..., 0]
print("Factual readings:", int(observed59.sum()),
      "local baseline NLL:", float(-np.log(p_true59[observed59]).mean()))

CF59 = []
for drug59, j59 in (("insulin", 0), ("norepinephrine", 1)):
    active59 = np.flatnonzero(rr39_rates[:, j59] > 0)
    ix59 = np.sort(rng59.choice(active59, min(400, len(active59)), replace=False))
    for fac59 in (0., 0.5, 1.5, 2., 3.):
        items59 = []
        for start59 in range(0, len(ix59), 128):
            ib59 = ix59[start59:start59+128]
            ub59 = rr35_scaled_action_encoding(ib59, drug59, fac59)
            ob59, hb59 = r59_core(ib59, ub59)
            items59.append(r59_features(ob59, hb59, ib59))
        cm59 = np.concatenate([v[0] for v in items59])
        cg59 = np.concatenate([v[1] for v in items59])
        ch59 = torch.cat([v[2] for v in items59])
        cp59 = torch.cat([v[3] for v in items59])
        dr59 = rr39_rates[ix59, j59]*(fac59-1)
        CF59.append(dict(drug=drug59, factor=fac59, indices=ix59,
                         dr=dr59, mean=cm59, G=cg59, H=ch59, P=cp59))
print("Factual feature table:", G59.shape,
      "counterfactual rows:", sum(len(c["indices"]) for c in CF59))
print("Seconds:", round(time.time()-t59, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 59: preservation-constrained head dataset\n"
            "- Reconstructed all 100 training stays; checked core against validated probes.\n"
            "- Built exact value-head features and local dose counterfactuals at "
            "multipliers 0, 0.5, 1.5, 2, and 3, without modifying weights.\n"
            "- Python runs: 59; forward passes: 4.\n")
```
