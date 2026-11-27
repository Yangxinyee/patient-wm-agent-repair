The larger calibration set confirms substantial under-response, especially for norepinephrine. It also reconstructs the readout query accurately. I’ll now fit closed-form head corrections and measure their trade-off against held-stay factual predictions, including prior sampling. A useful candidate must improve dose responses without materially shifting factual distributions.

```python
import pandas as pd

start183 = time.time()
state_rows183 = {}
for stay in np.unique(stays156):
    ii = np.flatnonzero(stays156 == stay)
    ii = ii[np.argsort(hours156[ii])]
    state_rows183[int(stay)] = ii

fact_rows183 = []
for row in factual157.itertuples(index=False):
    if row.event not in events182 or int(row.stay_id) not in state_rows183:
        continue
    ii = state_rows183[int(row.stay_id)]
    pos = np.searchsorted(hours156[ii], float(row.hour), side="left") - 1
    if pos >= 0:
        fact_rows183.append((row, int(ii[pos])))

nf183 = len(fact_rows183)
fact_ctx183 = torch.empty(nf183, 192)
fact_reg183 = torch.empty(nf183, 192)
fact_trans183 = torch.empty(nf183, 17)
fact_stays183 = np.empty(nf183, dtype=np.int64)
fact_bins183 = np.empty(nf183, dtype=int)
fact_events183 = []
for j, (row, i) in enumerate(fact_rows183):
    hh = state_cal161[i:i+1, 192:960].reshape(1, 4, 192)
    oo = state_cal161[i:i+1, :192]
    _, cap = capture159(hh, oo, int(row.stay_id), float(hours156[i]), row.event)
    xx = F.gelu(linear136(cap["with_code"], "attr_heads.value_bin.0"))
    fact_ctx183[j] = cap["ctx"][0]
    fact_reg183[j] = (cap["with_code"]-cap["q"])[0]
    fact_trans183[j] = (cap["logits"]-linear136(xx, "attr_heads.value_bin.2"))[0]
    fact_stays183[j] = int(row.stay_id)
    edges = numeric[(row.event, "value")]["fine_bin_edges"]
    fact_bins183[j] = 1 + np.searchsorted(edges, float(row.observed), side="left")
    fact_events183.append(row.event)

prior183 = mlp136(fact_ctx183, "z_prior")
gen183 = torch.Generator().manual_seed(183)
eps183 = torch.randn(2, nf183, 32, generator=gen183)
eps183 = torch.cat([torch.zeros(1, nf183, 32), eps183, -eps183], 0)
zs183 = prior183[None, :, :32] + torch.exp(prior183[None, :, 32:].clamp(-10, 5)) * eps183
views183 = fact_ctx183[None] + linear136(zs183, "z_head")
queries183 = norm136(views183 + W["type_emb.weight"][1], "query_norm")
fact_x183 = F.gelu(linear136(queries183 + fact_reg183[None], "attr_heads.value_bin.0"))
fact_logits183 = linear136(fact_x183, "attr_heads.value_bin.2") + fact_trans183[None]
fact_p183 = torch.softmax(fact_logits183, -1).numpy()
fact_xa183 = np.concatenate([
    fact_x183.numpy(), np.ones((5, nf183, 1), dtype=np.float32)
], -1).astype(np.float64)

unique_stays183 = np.sort(np.unique(stays156))
held_stays183 = unique_stays183[np.arange(len(unique_stays183)) % 4 == 3]
fact_hold183 = np.isin(fact_stays183, held_stays183)
cal_hold183 = np.isin(stays182, held_stays183)
xf183 = fact_xa183[:, ~fact_hold183].reshape(-1, 193)
CX183 = xf183.T @ xf183 / len(xf183)

dx_parts183, y_parts183 = [], []
for item in contrasts182:
    use = (drugs182 == item["drug"]) & ~cal_hold183
    dx = item["dx"][use].reshape(-1, 192)
    dx_parts183.append(np.column_stack([dx, np.zeros(len(dx))]))
    y_parts183.append(item["target"][use].reshape(-1))
DX183 = np.concatenate(dx_parts183).astype(np.float64)
Y183 = np.concatenate(y_parts183).astype(np.float64)
CD183 = DX183.T @ DX183 / len(DX183)
BD183 = DX183.T @ Y183 / len(DX183)
ridge183 = 1e-5 * np.trace(CX183) / 193
pold183 = fact_p183[1:].mean(0)
bin_index183 = np.arange(nf183)
old_nll183 = -np.log(pold183[bin_index183, fact_bins183].clip(1e-30))
tilt_np183 = tilt182.numpy()

def evaluate_head183(vector):
    scores = fact_xa183 @ vector
    new_logits = fact_logits183.numpy() + scores[:, :, None] * tilt_np183
    new_logits -= new_logits.max(-1, keepdims=True)
    pp = np.exp(new_logits)
    pp /= pp.sum(-1, keepdims=True)
    pn = pp[1:].mean(0)
    kl = (pold183 * (np.log(pold183.clip(1e-30)) - np.log(pn.clip(1e-30)))).sum(-1)
    dnll = -np.log(pn[bin_index183, fact_bins183].clip(1e-30)) - old_nll183
    result = {"held_KL": float(kl[fact_hold183].mean()),
              "held_dNLL": float(dnll[fact_hold183].mean()),
              "train_KL": float(kl[~fact_hold183].mean()),
              "norm": float(np.linalg.norm(vector))}
    for item in contrasts182:
        if item["scale"] not in [.5, 2.]:
            continue
        sel = (drugs182 == item["drug"]) & cal_hold183
        own = slice(0, 2) if item["drug"] == "insulin" else slice(2, 4)
        xb = base_x182.numpy()
        sb = xb @ vector[:192] + vector[192]
        sn = (xb+item["dx"]) @ vector[:192] + vector[192]
        lb = base_logits182.numpy() + sb[:, :, None] * tilt_np183
        ln = (base_logits182.numpy()
              + F.linear(torch.as_tensor(item["dx"]),
                         W["attr_heads.value_bin.2.weight"]).numpy()
              + sn[:, :, None] * tilt_np183)
        def mean_values(ll):
            ll = ll-ll.max(-1, keepdims=True)
            p = np.exp(ll); p /= p.sum(-1, keepdims=True)
            return (p*centres182.numpy()[None]).sum(-1)
        shift = mean_values(ln)-mean_values(lb)
        result[f'{item["drug"]}_{item["scale"]}'] = float(shift[sel, own].mean())
    return result

fits183 = {}
print("factual readings / held:", nf183, int(fact_hold183.sum()))
print("reconstructed original factual NLL:", round(float(old_nll183.mean()), 5))
for alpha in [1., 10., 100., 1e3, 1e4, 1e5, 1e6, 1e7]:
    vector = np.linalg.solve(CX183 + alpha*CD183 + ridge183*np.eye(193),
                             alpha*BD183)
    result = evaluate_head183(vector)
    fits183[alpha] = (vector, result)
    print("alpha", alpha, {k: round(v, 5) for k, v in result.items()})

print("held calibration targets:")
for item in contrasts182:
    if item["scale"] in [.5, 2.]:
        sel = (drugs182 == item["drug"]) & cal_hold183
        own = slice(0, 2) if item["drug"] == "insulin" else slice(2, 4)
        print(item["drug"], item["scale"],
              round(float(item["true_shift"][sel, own].mean()), 4))
print("elapsed seconds:", round(time.time()-start183, 1))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 183: factual-preserving head regression\n"
            "- Fitted rank-one ordinal-logit corrections to anchored drug contrasts by closed-form ridge.\n"
            "- Factual preservation includes mean-prior and four antithetic sampled-prior head inputs.\n"
            "- Stays are separated for training and internal validation. No weights installed yet.\n")
```
