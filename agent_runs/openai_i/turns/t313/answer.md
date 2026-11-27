The corrected posterior inputs are available, and the routing helper uses only 258 action contexts—not the full compensation sample. I’ll rebuild the full sample with the corrected inputs and compare all four attention heads using closed-form regressions. I’ll provisionally install a candidate only if its held-out reconstruction error improves; GPU factual results will determine whether it is acceptable.

```python
import torch.nn.functional as F
from scipy.linalg import solve
from scipy.spatial import cKDTree

torch.set_num_threads(8)
def arr275(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
def ten275(x):
    return torch.as_tensor(x, dtype=torch.float32)

# Match the full compensation sample to the retained posterior input rows.
ot275 = arr275(orig_token222).astype(np.float32)
sample275 = arr275(orig_tokens222[:, 6]).astype(np.float32)
lookup275 = {row.tobytes(): j for j, row in enumerate(ot275)}
ix275 = np.array([lookup275.get(row.tobytes(), -1) for row in sample275])
missing275 = ix275 < 0
if missing275.any():
    dist275, nearest275 = cKDTree(ot275).query(sample275[missing275])
    assert float(np.max(dist275)) < 1e-4, "Original-token sample alignment failed"
    ix275[missing275] = nearest275

x275 = ten275(actualx269)
z275 = F.linear(F.gelu(F.linear(x275, ten275(c271["z_post.0.weight"]),
                               ten275(c271["z_post.0.bias"]))),
                ten275(c271["z_post.2.weight"]),
                ten275(c271["z_post.2.bias"]))[:, :32]
tok275 = orig_tokens222.clone()
tok275[:, 6] = F.linear(z275, ten275(c271["z_in.weight"]),
                        ten275(c271["z_in.bias"]))[ix275]

# Split by training stay, never by individual slot.
sid275 = np.asarray(stays156)[np.asarray(ids182, dtype=int)[ix275]]
uniq275 = np.unique(sid275)
assert len(uniq275) >= 20, "Stay-level split unavailable"
hold275 = np.isin(sid275, uniq275[::5])
train275 = np.repeat(~hold275, 4)
test275 = ~train275
print("Aligned bundles:", len(ix275), "stays:", len(uniq275),
      "held stays:", len(np.unique(sid275[hold275])))

oq275 = np.asarray(load_original("core.prelude.0.attn.q.weight"))
okv275 = np.asarray(load_original("core.prelude.0.attn.kv.weight"))
oout275 = np.asarray(load_original("core.prelude.0.attn.out.weight"))
ofw275 = np.asarray(load_original("core.prelude.0.ffn.2.weight"))
ofb275 = np.asarray(load_original("core.prelude.0.ffn.2.bias"))
slots275 = arr275(orig_tokens222[:, :4]).reshape(-1, 192)

def ff275(x, fw=ofw275, fb=ofb275):
    t = ten275(x)
    h = F.gelu(F.linear(F.layer_norm(
        t, (192,), orig220["core.prelude.0.norm3.weight"],
        orig220["core.prelude.0.norm3.bias"]),
        orig220["core.prelude.0.ffn.0.weight"],
        orig220["core.prelude.0.ffn.0.bias"]))
    y = t + F.linear(h, ten275(fw), ten275(fb))
    return arr275(h).astype(np.float64), arr275(y).astype(np.float64)

fo275 = arr275(att_features220(orig_tokens222, oq275, okv275)).reshape(-1, 192)
oa275 = fo275 @ oout275.T
_, target275 = ff275(slots275 + oa275)

def ridge275(features, target, lambdas):
    a = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    m, s = a[train275].mean(0), a[train275].std(0) + 1e-4
    ym = y[train275].mean(0)
    ax = (a - m) / s
    g = ax[train275].T @ ax[train275]
    b = ax[train275].T @ (y[train275] - ym)
    best = None
    for lam in lambdas:
        co = solve(g + np.eye(g.shape[0]) * (train275.sum() * lam),
                   b, assume_a="pos", check_finite=False)
        coef = co / s[:, None]
        bias = ym - m @ coef
        err = float(np.sqrt(np.mean((a[test275] @ coef + bias - y[test275]) ** 2)))
        if best is None or err < best[0]:
            best = (err, lam, coef, bias)
    return best

# Baseline uses the current compensation, with response columns silenced.
baseout275 = c271["core.prelude.0.attn.out.weight"].copy()
baseout275[:, 48:50] = 0
basefeat275 = arr275(att_features220(
    tok275, c271["core.prelude.0.attn.q.weight"],
    c271["core.prelude.0.attn.kv.weight"])).reshape(-1, 192)
_, basey275 = ff275(slots275 + basefeat275 @ baseout275.T,
                    c271["core.prelude.0.ffn.2.weight"],
                    c271["core.prelude.0.ffn.2.bias"])
baseerr275 = float(np.sqrt(np.mean((basey275[test275] - target275[test275]) ** 2)))
print("Current corrected zero-response held RMS:", baseerr275)

candidates275 = {}
scores275 = {}
for head in range(4):
    start = head * 48
    q = oq275.copy()
    kv = okv275.copy()
    q[start:start+48] = c271["core.prelude.0.attn.q.weight"][48:96]
    kv[start:start+48] = c271["core.prelude.0.attn.kv.weight"][48:96]
    kv[192+start:192+start+3] = c271["core.prelude.0.attn.kv.weight"][240:243]
    features = arr275(att_features220(tok275, q, kv)).reshape(-1, 192)
    keep = np.setdiff1d(np.arange(192), np.arange(start, start+3))
    const = features[:, start+2]
    assert np.std(const) < 1e-4 and abs(np.mean(const)) > 1e-5
    ae, al, ac, ab = ridge275(features[:, keep], oa275,
                              [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    out = np.zeros_like(oout275)
    out[:, keep] = ac.T.astype(np.float32)
    out[:, start+2] = (ab / np.mean(const)).astype(np.float32)
    hidden, y = ff275(slots275 + features @ out.T)
    fe, fl, fc, fb = ridge275(hidden, target275 - y,
                              [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    candidate = {k: v.copy() for k, v in c271.items()}
    candidate["core.prelude.0.attn.q.weight"] = q
    candidate["core.prelude.0.attn.kv.weight"] = kv
    candidate["core.prelude.0.attn.out.weight"] = out
    candidate["core.prelude.0.ffn.2.weight"] = (ofw275 + fc.T).astype(np.float32)
    candidate["core.prelude.0.ffn.2.bias"] = (ofb275 + fb).astype(np.float32)
    candidate["core.prelude.0.attn.out.weight"][:, start:start+2] = \
        c271["core.prelude.0.attn.out.weight"][:, 48:50]
    candidates275[head] = candidate
    scores275[head] = (fe, ae, fl, al)
    np.savez(f"/work/head275_{head}.npz", **candidate)
    print("head", head, "held final RMS", round(fe, 6),
          "attention RMS", round(ae, 6), "FFN ridge", fl, "attention ridge", al)

besthead275 = min(scores275, key=lambda h: scores275[h][0])
installed275 = scores275[besthead275][0] < baseerr275
if installed275:
    for k in list(edited()):
        drop_edit(k)
    for k, v in candidates275[besthead275].items():
        save_edit(k, v.astype(np.float32))
print("Best head:", besthead275, "provisionally installed:", installed275)
print("Tuning validation stays:", tuning_ids269[:100])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 275 — corrected-input alternative-head regressions\n"
            f"- Current zero-response held slot RMS {baseerr275:.6f}; new scores {scores275}.\n"
            f"- Best head {besthead275}; provisionally installed: {installed275}.\n"
            "- Used closed-form ridge regressions and stay-held selection, no gradients or optimizer.\n"
            "- Drug response directions retained at gain 1; GPU factual and rollout validation remain required.\n")
```